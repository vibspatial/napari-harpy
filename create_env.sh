set -e

uv venv .venv --python 3.13
source .venv/bin/activate

uv pip install -e '.[dev]'
uv pip install jupyter

echo "Launch napari with: source .venv/bin/activate && napari"
echo "Open the widget from: Plugins -> napari-harpy -> Object Classifier"
