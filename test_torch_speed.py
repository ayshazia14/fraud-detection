import numpy as np
import torch
import time

print("Starting...")
start = time.time()
X = np.random.rand(227845, 30).astype('float64')
print(f"NumPy array created: {time.time()-start:.4f}s")

start = time.time()
X32 = X.astype('float32')
print(f"Dtype conversion: {time.time()-start:.4f}s")

start = time.time()
X_t = torch.tensor(X32)
print(f"Torch tensor: {time.time()-start:.4f}s")
