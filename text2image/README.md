# Quickstart for GLM-Image with uv

This guide helps you set up the environment and run the GLM-Image text-to-image generation sample using `uv`.

## Prerequisites

- **uv**: An extremely fast Python package and project manager.
  - Install via curl: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Or via pip: `pip install uv`
  - Or via Homebrew (macOS): `brew install uv`

## 1. Setup Project and Dependencies

Initialize the uv project and create the virtual environment with Python 3.12.
**Note**: We use Python 3.12 because Python 3.13/3.14 often lack pre-built binaries (wheels) for PyTorch, leading to build failures.

```bash
# cd $HOME/Code/VCS/ai/model-inference-samples
cd text2image
# Initialize project
uv init --python 3.12
uv venv --python 3.12 && source .venv/bin/activate && uv sync
```

Add the dependencies. We use `--prerelease=allow` to support the git dependencies.

```bash
# add packages to prod group
uv add --prerelease=allow -r requirements.txt
# add packages to dev group
uv add --dev pytest -r requirements_dev.txt
```
Reference:
* https://stackoverflow.com/questions/78902565/how-do-i-install-python-dev-dependencies-using-uv

This command will automatically:
- Update `pyproject.toml`
- Create the `.venv` virtual environment (if missing)
- Install all dependencies

Activate the virtual environment:

- **macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```
- **Windows**:
  ```powershell
  .venv\Scripts\activate
  ```

## 3. Run the Sample

The `glm.py` script downloads the model (if not already cached) and generates an image based on the prompt.

```bash
python glm_mps_dev.py
```

## Monitor MPS utilization
```bash
brew install asitop
sudo asitop
```

## Troubleshooting

- **Memory Issues**: If you run into OOM (Out Of Memory) errors, try reducing the image size in `glm.py` or enabling model offloading (though `enable_model_cpu_offload` might need adjustments for this specific pipeline).
- **Device Support**: The script attempts to auto-detect CUDA or MPS (macOS). If you are on a CPU-only machine, generation will be very slow.
