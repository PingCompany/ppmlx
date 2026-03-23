class PpLlm < Formula
  include Language::Python::Virtualenv

  desc "Ollama-style CLI for running LLMs on Apple Silicon via MLX"
  homepage "https://github.com/<org>/pp-llm"
  url "https://files.pythonhosted.org/packages/source/p/pp-llm/pp_llm-0.1.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "MIT"

  depends_on "python@3.11"
  depends_on :macos
  # Apple Silicon (arm64) is required for MLX
  on_arm do
    # MLX only runs on Apple Silicon
  end

  resource "typer" do
    url "https://files.pythonhosted.org/packages/source/t/typer/typer-0.12.3.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/source/r/rich/rich-13.7.1.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "fastapi" do
    url "https://files.pythonhosted.org/packages/source/f/fastapi/fastapi-0.115.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "uvicorn" do
    url "https://files.pythonhosted.org/packages/source/u/uvicorn/uvicorn-0.30.6.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "huggingface-hub" do
    url "https://files.pythonhosted.org/packages/source/h/huggingface_hub/huggingface_hub-0.24.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/source/p/pydantic/pydantic-2.7.1.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "httpx" do
    url "https://files.pythonhosted.org/packages/source/h/httpx/httpx-0.27.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  resource "sse-starlette" do
    url "https://files.pythonhosted.org/packages/source/s/sse_starlette/sse_starlette-2.1.0.tar.gz"
    sha256 "PLACEHOLDER"
  end

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      pp-llm requires Apple Silicon (M1/M2/M3/M4) and macOS 13+.

      MLX dependencies (mlx-lm, mlx-vlm, mlx-embeddings) must be installed
      separately as they require a running macOS ARM64 environment:

        pip install mlx-lm mlx-vlm mlx-embeddings

      Or install pp-llm via uv for full dependency resolution:

        uv tool install pp-llm

      Quick start:
        pp-llm pull llama3
        pp-llm run llama3
        pp-llm serve          # OpenAI-compatible API on :6767
    EOS
  end

  test do
    assert_match "pp-llm", shell_output("#{bin}/pp-llm --version")
    assert_match "pp-llm", shell_output("#{bin}/pp-llm --help")
  end
end
