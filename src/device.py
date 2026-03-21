import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6"

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"
