"""Voice I/O for ppmlx — STT via Whisper, TTS via mlx-audio.

Provides push-to-talk voice input and streaming TTS output,
all running locally on Apple Silicon via MLX.

Dependencies (optional extras):
    pip install ppmlx[voice]
    # or manually: pip install mlx-whisper mlx-audio sounddevice soundfile
"""
from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("ppmlx.voice")


@dataclass
class VoiceConfig:
    """Configuration for voice I/O."""
    stt_model: str = "mlx-community/whisper-large-v3-turbo"
    tts_model: str = "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"
    tts_voice: str | None = None  # None = default voice
    sample_rate: int = 16000  # Recording sample rate (STT)
    tts_sample_rate: int = 24000  # Playback sample rate (TTS)
    silence_threshold: float = 0.01  # RMS threshold for silence detection
    silence_duration: float = 1.5  # Seconds of silence to stop recording
    max_record_seconds: float = 30.0  # Maximum recording duration


class VoiceInput:
    """Push-to-talk or auto-silence voice input using the microphone."""

    def __init__(self, config: VoiceConfig | None = None):
        self.config = config or VoiceConfig()
        self._whisper = None

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

    def record_and_transcribe(self) -> str:
        """Record from microphone until silence, then transcribe.

        Returns the transcribed text.
        """
        self._load_whisper()
        audio = self._record_until_silence()
        if audio is None or len(audio) == 0:
            return ""
        return self._transcribe(audio)

    def transcribe_file(self, path: str | Path) -> str:
        """Transcribe an audio file."""
        self._load_whisper()
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
                        if silent_chunks >= silence_limit and len(chunks) > silence_limit:
                            break
                    else:
                        silent_chunks = 0
        except KeyboardInterrupt:
            pass

        if not chunks:
            return None

        import numpy as np
        return np.concatenate(chunks, axis=0).flatten()

    def _transcribe(self, audio: Any) -> str:
        """Transcribe a numpy audio array."""
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, self.config.sample_rate)
            tmp_path = f.name

        try:
            result = self._whisper.transcribe(
                tmp_path,
                path_or_hf_repo=self.config.stt_model,
            )
            return result.get("text", "").strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class VoiceOutput:
    """Text-to-speech output using mlx-audio."""

    def __init__(self, config: VoiceConfig | None = None):
        self.config = config or VoiceConfig()
        self._tts_model = None

    def _load_tts(self) -> None:
        """Lazy-load the TTS model."""
        if self._tts_model is not None:
            return
        try:
            from mlx_audio.tts.utils import load_model
            self._tts_model = load_model(self.config.tts_model)
            log.info("TTS model: %s", self.config.tts_model)
        except ImportError:
            raise ImportError(
                "mlx-audio is required for TTS. "
                "Install with: pip install mlx-audio"
            )

    def speak(self, text: str) -> None:
        """Generate speech and play it through speakers."""
        self._load_tts()
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for audio playback. "
                "Install with: brew install portaudio && pip install sounddevice"
            )

        kwargs: dict[str, Any] = {"text": text}
        if self.config.tts_voice:
            kwargs["voice"] = self.config.tts_voice

        for result in self._tts_model.generate(**kwargs):
            audio = result.audio
            if audio is not None:
                sr = getattr(result, "sample_rate", self.config.tts_sample_rate)
                sd.play(audio, samplerate=sr)
                sd.wait()

    def speak_streamed(self, text: str) -> None:
        """Generate speech with streaming playback (lower latency)."""
        self._load_tts()
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            raise ImportError(
                "sounddevice and numpy required. "
                "Install with: pip install sounddevice numpy"
            )

        kwargs: dict[str, Any] = {"text": text}
        if self.config.tts_voice:
            kwargs["voice"] = self.config.tts_voice

        for result in self._tts_model.generate(**kwargs):
            audio = result.audio
            if audio is not None:
                sr = getattr(result, "sample_rate", self.config.tts_sample_rate)
                sd.play(audio, samplerate=sr)
                sd.wait()

    def save(self, text: str, path: str | Path) -> Path:
        """Generate speech and save to a WAV file."""
        self._load_tts()
        import soundfile as sf

        path = Path(path)
        all_audio = []
        sr = self.config.tts_sample_rate

        for result in self._tts_model.generate(text=text):
            if result.audio is not None:
                all_audio.append(result.audio)
                sr = getattr(result, "sample_rate", sr)

        if all_audio:
            import numpy as np
            combined = np.concatenate(all_audio)
            sf.write(str(path), combined, sr)

        return path
