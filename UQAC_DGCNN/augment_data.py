import os
import math
import numpy as np
import torch
from torch_geometric.datasets import ModelNet
from torch_cluster import fps
import h5py
from tqdm import tqdm

# Parameters
root = './data/ModelNet40'
out_dir = './augmented_datasets'
os.makedirs(out_dir, exist_ok=True)
factors = [2, 5, 10, 100]
num_points = 1024
trunc_thresh = 10000   # maximum points before FPS
BATCH = 512            # batch size for HDF5 writes
early_flush = True     # flush in batches if True
use_lzf = True         # switch to faster LZF compression or None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) Load dataset and build class_to_points
print("Loading ModelNet40...")
dataset = ModelNet(root=root, name='40', train=True, transform=None)
class_to_points = {}
for data in dataset:
    pts = data.pos.numpy()
    if pts.shape[0] < num_points * 2:
        continue
    lbl = int(data.y)
    class_to_points.setdefault(lbl, []).append(pts)
orig_counts = {lbl: len(lst) for lbl, lst in class_to_points.items()}
max_orig = max(orig_counts.values())
print(f"Found {len(class_to_points)} classes; max_orig={max_orig}")

# 2) GPU-based FPS + normalization
def sample_and_normalize(points_np):
    # Truncate on CPU
    if points_np.shape[0] > trunc_thresh:
        idx = np.random.choice(points_np.shape[0], trunc_thresh, replace=False)
        points_np = points_np[idx]
    pts = torch.from_numpy(points_np).float().to(device)
    N = pts.size(0)
    idx_pt = fps(
        pts,
        torch.zeros(N, dtype=torch.long, device=device),
        ratio=float(num_points) / N,
        random_start=True
    )[:num_points]
    sampled = pts[idx_pt]
    # Zero-center & unit-scale
    sampled = sampled - sampled.mean(dim=0, keepdim=True)
    sampled = sampled / sampled.norm(dim=1).max()
    return sampled  # CUDA tensor

# 3) On-the-fly augmentation on GPU
def augment_gpu(pc_tensor):
    # pc_tensor: (num_points,3) on CUDA
    theta_val = (torch.rand(1, device=device) * 2 * math.pi).item()
    c = math.cos(theta_val)
    s = math.sin(theta_val)
    # Build rotation matrix from Python floats (avoiding mismatched tensor shapes)
    R = torch.tensor(
        [[c, -s, 0.0],
         [s,  c, 0.0],
         [0.0, 0.0, 1.0]],
        device=device,
        dtype=pc_tensor.dtype
    )
    pc = pc_tensor @ R.T
    scale = torch.empty(1, device=device).uniform_(0.9, 1.1).item()
    pc = pc * scale
    noise = torch.randn_like(pc) * 0.005
    return pc + noise

# 4) Buffered HDF5 writes with a single tqdm over all samples
for f in factors:
    target = f * max_orig
    total_samples = len(class_to_points) * target
    comp = 'lzf' if use_lzf else None
    out_path = os.path.join(out_dir, f'modelnet40_bal_x{f}.h5')
    print(f"=== Augmenting ×{f} -> {out_path} ===")

    with h5py.File(out_path, 'w') as hf, tqdm(total=total_samples, desc=f"×{f} samples") as pbar:
        dset_data = hf.create_dataset(
            'data', (total_samples, num_points, 3),
            dtype='f4', compression=comp,
            chunks=(BATCH, num_points, 3)
        )
        dset_label = hf.create_dataset(
            'label', (total_samples,),
            dtype='i8', compression=comp
        )
        ptr = 0
        buf_tensors = []
        buf_labels = []

        def flush(curr_ptr):
            batch_size = len(buf_labels)
            arr = torch.stack(buf_tensors).cpu().numpy()
            dset_data[curr_ptr:curr_ptr+batch_size] = arr
            dset_label[curr_ptr:curr_ptr+batch_size] = buf_labels
            del buf_tensors[:]
            del buf_labels[:]
            pbar.update(batch_size)
            return curr_ptr + batch_size

        for lbl, pts_list in sorted(class_to_points.items()):
            orig_count = len(pts_list)
            needed = target - orig_count

            # Write original samples
            for pts in pts_list:
                buf_tensors.append(sample_and_normalize(pts))
                buf_labels.append(lbl)
                if early_flush and len(buf_labels) >= BATCH:
                    ptr = flush(ptr)

            # Write augmented samples
            for _ in range(needed):
                base = pts_list[np.random.randint(orig_count)]
                sampled = sample_and_normalize(base)
                buf_tensors.append(augment_gpu(sampled))
                buf_labels.append(lbl)
                if early_flush and len(buf_labels) >= BATCH:
                    ptr = flush(ptr)

        # Flush any remaining
        if buf_labels:
            ptr = flush(ptr)

    print(f"Saved {ptr}/{total_samples} samples to {out_path}")

print("All done!")
