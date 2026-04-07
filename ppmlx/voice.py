"""Voice I/O for ppmlx — STT via Whisper, TTS via mlx-audio.

Provides push-to-talk voice input and streaming TTS output,
all running locally on Apple Silicon via MLX.

Dependencies (optional extras):
    pip install ppmlx[voice]
    # or manually: pip install mlx-whisper mlx-audio sounddevice soundfile pynput
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("ppmlx.voice")


@dataclass
class VoiceConfig:
    """Configuration for voice I/O."""
    stt_model: str = "mlx-community/whisper-large-v3-turbo-q4"
    tts_model: str = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
    tts_voice: str | None = None  # None = default voice
    sample_rate: int = 16000  # Recording sample rate (STT)
    tts_sample_rate: int = 24000  # Playback sample rate (TTS)
    tts_speed: float = 1.0  # TTS playback speed multiplier
    silence_threshold: float = 0.01  # RMS threshold for silence detection
    silence_duration: float = 1.5  # Seconds of silence to stop recording
    max_record_seconds: float = 30.0  # Maximum recording duration
    tts_volume: float = 1.10  # Target peak level for normalization
    # Push-to-talk: hold ptt_key while speaking, release to transcribe.
    ptt_mode: bool = False
    ptt_key: str = "space"  # Accepts 'space', 'f5', single chars, …


def _ensure_hf_token() -> None:
    """Set HF_TOKEN env var from ppmlx config if not already set.

    Third-party libs (mlx_whisper, mlx_audio) use huggingface_hub which
    only reads the env var, not our config.toml.
    """
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from ppmlx.models import _get_hf_token
        token = _get_hf_token()
        if token:
            os.environ["HF_TOKEN"] = token
    except Exception:
        pass


class VoiceInput:
    """Push-to-talk or auto-silence voice input using the microphone."""

    def __init__(self, config: VoiceConfig | None = None):
        self.config = config or VoiceConfig()
        self._whisper = None
        _ensure_hf_token()

    def _load_whisper(self) -> None:
        """Lazy-load the Whisper model."""
        if self._whisper is not None:
            return
        try:
            import mlx_whisper
            self._whisper = mlx_whisper
            log.info("STT model: %s", self.config.stt_model)
        except ImportError:
            raise ImportError(
                "mlx-whisper is required for voice input. "
                "Install with: pip install mlx-whisper"
            )

    def preload_stt(self) -> None:
        """Pre-download and load the STT model into memory.

        Suppresses noisy huggingface_hub progress bars so the caller can
        show its own loading animation instead.
        """
        self._load_whisper()
        import io, contextlib
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            with contextlib.redirect_stderr(io.StringIO()), \
                 contextlib.redirect_stdout(io.StringIO()):
                from mlx_whisper.load_models import load_model
                load_model(self.config.stt_model)
        finally:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

    @staticmethod
    def _check_ffmpeg() -> None:
        """Raise a clear error if ffmpeg is not on PATH (required for file transcription)."""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg is required to transcribe audio files but was not found on PATH.\n"
                "Install it with:  brew install ffmpeg"
            )

    def record_and_transcribe(self) -> str:
        """Record from microphone, then transcribe.

        Uses push-to-talk (hold key, release to send) when ptt_mode is True,
        otherwise records until silence is detected.

        Returns the transcribed text, or "" if nothing was captured.
        """
        self._load_whisper()
        if self.config.ptt_mode:
            audio = self._record_ptt()
        else:
            audio = self._record_until_silence()
        if audio is None or len(audio) == 0:
            return ""
        # Skip clips that are too short to contain real speech (< 0.3 s)
        if len(audio) < self.config.sample_rate * 0.3:
            return ""
        # Trim leading/trailing silence so Whisper doesn't hallucinate
        audio = self._trim_silence(audio)
        if audio is None or len(audio) < self.config.sample_rate * 0.3:
            return ""
        return self._transcribe(audio)

    def _record_ptt(self) -> Any | None:
        """Record audio while the configured PTT key is held down.

        Blocks until the key is first pressed, then records until it is released.
        Returns a 1-D float32 numpy array at self.config.sample_rate Hz, or None.
        """
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            raise ImportError(
                "sounddevice is required for microphone recording. "
                "Install with: brew install portaudio && pip install sounddevice"
            )
        try:
            from pynput import keyboard as kb
        except ImportError:
            raise ImportError(
                "pynput is required for push-to-talk. "
                "Install with: pip install pynput  (or reinstall ppmlx[voice])"
            )

        target_key = _parse_pynput_key(self.config.ptt_key, kb)

        press_event = threading.Event()
        release_event = threading.Event()

        def on_press(key: Any) -> None:
            if _key_matches(key, target_key):
                press_event.set()

        def on_release(key: Any) -> None:
            if _key_matches(key, target_key):
                release_event.set()

        listener = kb.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        try:
            press_event.wait()  # block until key goes down

            sr = self.config.sample_rate
            chunk_size = int(sr * 0.05)  # 50 ms chunks
            chunks: list[Any] = []

            with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                blocksize=chunk_size) as stream:
                while not release_event.is_set():
                    data, _ = stream.read(chunk_size)
                    chunks.append(data.copy())
        finally:
            listener.stop()

        if not chunks:
            return None
        return np.concatenate(chunks, axis=0).flatten()

    def transcribe_file(self, path: str | Path) -> str:
        """Transcribe an audio file."""
        self._load_whisper()
        self._check_ffmpeg()  # mlx_whisper shells out to ffmpeg for file decoding
        result = self._whisper.transcribe(
            str(path),
            path_or_hf_repo=self.config.stt_model,
        )
        return result.get("text", "").strip()

    def _record_until_silence(self) -> Any:
        """Record audio until silence is detected."""
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            raise ImportError(
                "sounddevice is required for microphone recording. "
                "Install with: brew install portaudio && pip install sounddevice"
            )

        sr = self.config.sample_rate
        chunk_duration = 0.1  # 100ms chunks
        chunk_size = int(sr * chunk_duration)
        max_chunks = int(self.config.max_record_seconds / chunk_duration)

        chunks: list[Any] = []
        silent_chunks = 0
        speech_detected = False
        silence_limit = int(self.config.silence_duration / chunk_duration)

        log.debug("Recording... (silence threshold=%.3f, max=%.0fs)",
                  self.config.silence_threshold, self.config.max_record_seconds)

        try:
            with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                blocksize=chunk_size) as stream:
                for _ in range(max_chunks):
                    data, _ = stream.read(chunk_size)
                    chunks.append(data.copy())

                    # Check for silence
                    rms = float(np.sqrt(np.mean(data ** 2)))
                    if rms < self.config.silence_threshold:
                        silent_chunks += 1
                        # Only stop on silence AFTER speech was detected
                        if speech_detected and silent_chunks >= silence_limit:
                            break
                    else:
                        silent_chunks = 0
                        if not speech_detected:
                            speech_detected = True
                            log.debug("Speech detected (RMS=%.4f)", rms)
        except KeyboardInterrupt:
            pass

        if not speech_detected:
            return None

        if not chunks:
            return None

        import numpy as np
        return np.concatenate(chunks, axis=0).flatten()

    def _trim_silence(self, audio: Any) -> Any:
        """Trim leading and trailing silence from audio."""
        import numpy as np
        threshold = self.config.silence_threshold
        frame_len = int(self.config.sample_rate * 0.05)  # 50 ms frames
        # Find first frame above threshold
        start = 0
        for i in range(0, len(audio) - frame_len, frame_len):
            rms = float(np.sqrt(np.mean(audio[i:i + frame_len] ** 2)))
            if rms >= threshold:
                start = max(0, i - frame_len)  # keep one frame of lead-in
                break
        else:
            return None
        # Find last frame above threshold
        end = len(audio)
        for i in range(len(audio) - frame_len, start, -frame_len):
            rms = float(np.sqrt(np.mean(audio[i:i + frame_len] ** 2)))
            if rms >= threshold:
                end = min(len(audio), i + 2 * frame_len)
                break
        return audio[start:end]

    # Known Whisper hallucinations on silence/noise
    _HALLUCINATION_PHRASES: frozenset[str] = frozenset({
        "thank you", "thanks for watching", "thanks for listening",
        "thank you for watching", "thank you for listening",
        "you", "bye", "bye bye", "goodbye",
        "the end", "see you next time", "see you",
        "subscribe", "like and subscribe",
        "dziękuję", "dzięki", "do widzenia",
    })

    def _transcribe(self, audio: Any) -> str:
        """Transcribe a numpy audio array.

        Uses hallucination_silence_threshold and condition_on_previous_text=False
        to minimize Whisper hallucinations on short/quiet clips.
        """
        import io, contextlib
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            with contextlib.redirect_stderr(io.StringIO()), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = self._whisper.transcribe(
                    audio,
                    path_or_hf_repo=self.config.stt_model,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    hallucination_silence_threshold=2.0,
                    temperature=(0.0, 0.2, 0.4),
                )
        finally:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        text = result.get("text", "").strip()
        # Filter known hallucination phrases
        if text.lower().rstrip(".!?,") in self._HALLUCINATION_PHRASES:
            log.debug("Filtered hallucination: %r", text)
            return ""
        return text


class VoiceOutput:
    """Text-to-speech output using mlx-audio."""

    def __init__(self, config: VoiceConfig | None = None):
        self.config = config or VoiceConfig()
        self._tts_model = None
        _ensure_hf_token()

    def _load_tts(self) -> None:
        """Lazy-load the TTS model."""
        if self._tts_model is not None:
            return
        try:
            from mlx_audio.tts.utils import load_model
            import io, contextlib
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            try:
                with contextlib.redirect_stderr(io.StringIO()), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self._tts_model = load_model(self.config.tts_model)
            finally:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            log.info("TTS model: %s", self.config.tts_model)
        except ImportError:
            raise ImportError(
                "mlx-audio is required for TTS. "
                "Install with: pip install mlx-audio"
            )

    def _normalize_audio(self, audio: Any) -> Any:
        """Normalize audio to target peak level and remove DC offset."""
        import numpy as np
        a = np.asarray(audio, dtype=np.float32)
        # Remove DC offset
        a = a - np.mean(a)
        # Normalize to target peak
        peak = np.max(np.abs(a))
        target = self.config.tts_volume
        if peak > 1e-6:
            a = a * (target / peak)
        return a

    @staticmethod
    def clean_text_for_speech(text: str) -> str:
        """Strip markdown, code blocks, and other non-speech content."""
        import re
        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Remove inline code
        text = re.sub(r"`[^`]+`", "", text)
        # Remove markdown links, keep text: [text](url) → text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove markdown headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove bold/italic markers
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
        # Remove bullet points
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        # Remove numbered lists prefix
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        # Collapse whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for incremental TTS."""
        import re
        # Split on sentence-ending punctuation followed by space or end
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in parts if s.strip()]

    def _generate_audio(self, text: str) -> tuple[Any, int] | None:
        """Generate speech for a text chunk. Returns (audio_np, sample_rate) or None."""
        import numpy as np
        kwargs: dict[str, Any] = {"text": text}
        if self.config.tts_voice:
            kwargs["voice"] = self.config.tts_voice
        chunks = []
        sr = self.config.tts_sample_rate
        for result in self._tts_model.generate(**kwargs):
            if result.audio is not None:
                chunks.append(np.asarray(result.audio, dtype=np.float32))
                sr = getattr(result, "sample_rate", sr)
        if not chunks:
            return None
        audio = self._normalize_audio(np.concatenate(chunks))
        # Apply speed multiplier via sample rate
        playback_sr = int(sr * self.config.tts_speed)
        return audio, playback_sr

    def speak(self, text: str) -> None:
        """Generate speech sentence by sentence, overlapping generation and playback."""
        self._load_tts()
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for audio playback. "
                "Install with: brew install portaudio && pip install sounddevice"
            )

        text = self.clean_text_for_speech(text)
        if not text:
            return

        sentences = self._split_sentences(text)
        if not sentences:
            return

        # Generate first sentence synchronously
        result = self._generate_audio(sentences[0])
        if result is None:
            return

        for i in range(len(sentences)):
            audio, sr = result  # type: ignore[misc]

            # Start playback of current sentence
            sd.play(audio, samplerate=sr)

            # Generate next sentence in parallel while audio plays
            next_result = None
            if i + 1 < len(sentences):
                next_result = self._generate_audio(sentences[i + 1])

            # Wait for current playback to finish
            sd.wait()
            result = next_result
            if result is None and i + 1 < len(sentences):
                break

    def save(self, text: str, path: str | Path) -> Path:
        """Generate speech and save to a WAV file."""
        self._load_tts()
        import soundfile as sf
        import numpy as np

        text = self.clean_text_for_speech(text)
        path = Path(path)
        all_audio: list[Any] = []
        sr = self.config.tts_sample_rate

        for result in self._tts_model.generate(text=text):
            if result.audio is not None:
                all_audio.append(np.asarray(result.audio, dtype=np.float32))
                sr = getattr(result, "sample_rate", sr)

        if all_audio:
            combined = self._normalize_audio(np.concatenate(all_audio))
            sf.write(str(path), combined, sr)

        return path


# ---------------------------------------------------------------------------
# Push-to-talk helpers (module level so they can be unit-tested independently)
# ---------------------------------------------------------------------------

def _parse_pynput_key(key_str: str, kb: Any) -> Any:
    """Translate a human-friendly key name to a pynput Key or KeyCode.

    Supported formats
    -----------------
    - Named special keys: 'space', 'enter', 'tab', 'esc'/'escape',
      'backspace', 'delete', 'home', 'end', 'up', 'down', 'left', 'right',
      'ctrl'/'ctrl_l'/'ctrl_r', 'shift'/'shift_l'/'shift_r',
      'alt'/'alt_l'/'alt_r', 'cmd'/'super', 'caps_lock', 'insert',
      'page_up', 'page_down', 'num_lock', 'scroll_lock', 'pause', 'menu'
    - Function keys: 'f1' … 'f20'
    - Single printable character: 'r', 'g', '0', …
    """
    import re

    key_str = key_str.strip().lower()

    _NAMED: dict[str, Any] = {
        "space": kb.Key.space,
        "enter": kb.Key.enter, "return": kb.Key.enter,
        "tab": kb.Key.tab,
        "esc": kb.Key.esc, "escape": kb.Key.esc,
        "backspace": kb.Key.backspace,
        "delete": kb.Key.delete,
        "home": kb.Key.home, "end": kb.Key.end,
        "up": kb.Key.up, "down": kb.Key.down,
        "left": kb.Key.left, "right": kb.Key.right,
        "ctrl": kb.Key.ctrl_l, "ctrl_l": kb.Key.ctrl_l, "ctrl_r": kb.Key.ctrl_r,
        "shift": kb.Key.shift_l, "shift_l": kb.Key.shift_l, "shift_r": kb.Key.shift_r,
        "alt": kb.Key.alt_l, "alt_l": kb.Key.alt_l, "alt_r": kb.Key.alt_r,
        "cmd": kb.Key.cmd, "super": kb.Key.cmd,
        "caps_lock": kb.Key.caps_lock,
        "insert": kb.Key.insert,
        "page_up": kb.Key.page_up, "page_down": kb.Key.page_down,
        "num_lock": kb.Key.num_lock,
        "scroll_lock": kb.Key.scroll_lock,
        "pause": kb.Key.pause,
        "menu": kb.Key.menu,
    }

    if key_str in _NAMED:
        return _NAMED[key_str]

    # Function keys f1–f20
    m = re.match(r"^f(\d{1,2})$", key_str)
    if m:
        n = int(m.group(1))
        fkey = getattr(kb.Key, f"f{n}", None)
        if fkey is not None:
            return fkey

    # Single printable character
    if len(key_str) == 1:
        return kb.KeyCode.from_char(key_str)

    raise ValueError(
        f"Unknown PTT key: {key_str!r}. "
        "Use 'space', 'f5', 'ctrl', or a single character like 'r'. "
        "Set [voice] ptt_key in ~/.ppmlx/config.toml."
    )


def _key_matches(pressed: Any, target: Any) -> bool:
    """Return True if the pynput key event matches the target key."""
    return pressed == target
