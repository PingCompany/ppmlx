class Ppmlx < Formula
  include Language::Python::Virtualenv

  desc "CLI for running LLMs on Apple Silicon via MLX"
  homepage "https://github.com/the-focus-company/ppmlx"
  url "https://files.pythonhosted.org/packages/ee/0a/433431922f521f2bab089399880f59768cc3b1ecc4d53848be287bfdb26b/ppmlx-0.1.0.tar.gz"
  sha256 "22f94d51c01930f8f2dd865bca022cbdb7711afc70fdc87e663b61121edbecff"
  license "MIT"

  depends_on "python@3.11"
  depends_on :macos

  resource "typer" do
    url "https://files.pythonhosted.org/packages/ac/0a/d55af35db5f50f486e3eda0ada747eed773859e2699d3ce570b682a9b70a/typer-0.12.3.tar.gz"
    sha256 "49e73131481d804288ef62598d97a1ceef3058905aa536a1134f90891ba35482"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/b3/01/c954e134dc440ab5f96952fe52b4fdc64225530320a910473c1fe270d9aa/rich-13.7.1.tar.gz"
    sha256 "9be308cb1fe2f1f57d67ce99e95af38a1e2bc71ad9813b0e247cf7ffbcc3a432"
  end

  resource "fastapi" do
    url "https://files.pythonhosted.org/packages/7b/5e/bf0471f14bf6ebfbee8208148a3396d1a23298531a6cc10776c59f4c0f87/fastapi-0.115.0.tar.gz"
    sha256 "f93b4ca3529a8ebc6fc3fcf710e5efa8de3df9b41570958abf1d97d843138004"
  end

  resource "uvicorn" do
    url "https://files.pythonhosted.org/packages/5a/01/5e637e7aa9dd031be5376b9fb749ec20b86f5a5b6a49b87fabd374d5fa9f/uvicorn-0.30.6.tar.gz"
    sha256 "4b15decdda1e72be08209e860a1e10e92439ad5b97cf44cc945fcbee66fc5788"
  end

  resource "huggingface-hub" do
    url "https://files.pythonhosted.org/packages/66/84/9240cb3fc56112c7093ef84ece44a555386263e7a19c81a4c847fd7e2bba/huggingface_hub-0.24.0.tar.gz"
    sha256 "6c7092736b577d89d57b3cdfea026f1b0dc2234ae783fa0d59caf1bf7d52dfa7"
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/1f/74/0d009e056c2bd309cdc053b932d819fcb5ad3301fc3e690c097e1de3e714/pydantic-2.7.1.tar.gz"
    sha256 "e9dbb5eada8abe4d9ae5f46b9939aead650cd2b68f249bb3a8139dbe125803cc"
  end

  resource "httpx" do
    url "https://files.pythonhosted.org/packages/5c/2d/3da5bdf4408b8b2800061c339f240c1802f2e82d55e50bd39c5a881f47f0/httpx-0.27.0.tar.gz"
    sha256 "a0cb88a46f32dc874e04ee956e4c2764aba2aa228f650b06788ba6bda2962ab5"
  end

  resource "sse-starlette" do
    url "https://files.pythonhosted.org/packages/2a/dc/f2c00e9bd7355134a88cd83ed984b664d0f187ea36f1bd47b310ae731e17/sse_starlette-2.1.0.tar.gz"
    sha256 "ffff6e7d948f925f347e662be77af5783a6b93efce15d42c03004dcd7d6d91d3"
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
