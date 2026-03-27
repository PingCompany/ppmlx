class Ppmlx < Formula
  include Language::Python::Virtualenv

  desc "CLI for running LLMs on Apple Silicon via MLX"
  homepage "https://github.com/PingCompany/ppmlx"
  url "https://files.pythonhosted.org/packages/source/p/ppmlx/ppmlx-0.2.0.tar.gz"
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
      ppmlx requires Apple Silicon (M1/M2/M3/M4) and macOS 13+.

      MLX dependencies (mlx-lm, mlx-vlm, mlx-embeddings) must be installed
      separately as they require a running macOS ARM64 environment:

        pip install mlx-lm mlx-vlm mlx-embeddings

      Or install ppmlx via uv for full dependency resolution:

        uv tool install ppmlx

      Quick start:
        ppmlx pull llama3
        ppmlx run llama3
        ppmlx serve          # OpenAI-compatible API on :6767
    EOS
  end

  test do
    assert_match "ppmlx", shell_output("#{bin}/ppmlx --version")
    assert_match "ppmlx", shell_output("#{bin}/ppmlx --help")
  end
end
