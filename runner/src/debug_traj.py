import numpy as np
import os

traj_path = r"D:\desktop\code\conditional-flow-matching\runner\logs\1.26\eb_phate\2026-01-22_23-13-02\figs\trajs.npy"
trajs = np.load(traj_path, allow_pickle=True)
print(f'Shape: {trajs.shape}')
print(f'Dtype: {trajs.dtype}')
print(f'First 3 trajectories:')
print(trajs[:3])
