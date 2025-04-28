#!/usr/bin/env python3
import argparse
import h5py
import numpy as np


def split_h5(input_path, train_path, val_path, test_path,
             val_fraction=0.20, test_fraction=0.01, seed=42):
    # 1) load entire dataset
    with h5py.File(input_path, 'r') as hf:
        data   = hf['data'][:]    # (N,1024,3)
        labels = hf['label'][:]   # (N,)
    N = data.shape[0]

    # 2) shuffle indices
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)

    # 3) compute split sizes (at least 1 sample each)
    n_test = max(int(N * test_fraction), 1)
    n_val  = max(int(N * val_fraction), 1)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    # 4) helper to write
    def write_split(path, indices):
        with h5py.File(path, 'w') as out:
            out.create_dataset('data',  data=data[indices],  compression='gzip')
            out.create_dataset('label', data=labels[indices], compression='gzip')
        print(f"Wrote {path}: {len(indices)} samples")

    # 5) write all three
    write_split(train_path, train_idx)
    write_split(val_path,   val_idx)
    write_split(test_path,  test_idx)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input',      required=True,
                   help='original .h5 (modelnet40_bal_x2.h5)')
    p.add_argument('--train-out',  default='train.h5')
    p.add_argument('--val-out',    default='val.h5')
    p.add_argument('--test-out',   default='test.h5')
    p.add_argument('--val-frac',   type=float, default=0.20)
    p.add_argument('--test-frac',  type=float, default=0.01)
    p.add_argument('--seed',       type=int,   default=42)
    args = p.parse_args()

    split_h5(args.input,
             args.train_out,
             args.val_out,
             args.test_out,
             val_fraction=args.val_frac,
             test_fraction=args.test_frac,
             seed=args.seed)
