# On macOS, LightGBM and PyTorch each ship their own OpenMP (threading) library, and the two
# clash inside one process:
#   - PyTorch imported first  -> loading a LightGBM model crashes
#   - LightGBM used first     -> PyTorch's multi-threaded CPU code freezes or crashes
# Fix for the tests: import LightGBM first and run PyTorch on a single thread.
# The app is not affected: it never imports PyTorch (it runs the CNN with ONNX Runtime).
import lightgbm  # noqa: F401, I001
import torch

torch.set_num_threads(1)
