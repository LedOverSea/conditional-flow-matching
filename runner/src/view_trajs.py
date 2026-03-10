"""
查看轨迹可视化脚本
参考: src/models/components/plotting.py 中的 plot_trajectories 和 plot_trajectory 函数
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端进行绘图
import matplotlib.pyplot as plt
import os
import scprep  # 用于高质量散点图绘制

# 设置轨迹文件目录
traj_dir = r"D:\desktop\code\conditional-flow-matching\runner\logs\1.26\eb_phate"

# 要查看的模型文件夹列表
traj_files = [
    "2026-01-22_23-13-02",  
]

# 遍历每个模型的轨迹文件
for traj_file in traj_files:
    traj_path = os.path.join(traj_dir, traj_file, "figs", "trajs.npy")
    
    # 检查文件是否存在
    if not os.path.exists(traj_path):
        print(f"File not found: {traj_path}")
        continue
        
    # 加载轨迹数据
    # 数据形状: (n_trajs, n_times, dim) = (轨迹数, 时间步数, 维度)
    trajs = np.load(traj_path, allow_pickle=True)
    
    print(f"\n=== {traj_file} ===")
    print(f"Shape: {trajs.shape}")
    
    # 解析数据维度
    n_trajs, n_times, dim = trajs.shape
    print(f"Trajectories: {n_trajs}, Time points: {n_times}, Dimensions: {dim}")
    
    # 取前200条轨迹进行可视化
    n = min(200, n_trajs)
    
    # 准备绘图数据
    obs = trajs
    batch_size = n_trajs
    ts = n_times
    
    # 将轨迹数据展平，用于绘制散点图
    # obs_flat: (n_trajs * n_times, dim)
    obs_flat = obs.reshape(-1, dim)
    
    # 创建时间标签，用于着色
    # tts: 重复每个时间点 n_trajs 次
    tts = np.tile(np.arange(ts), batch_size)
    
    # 创建图形，参考 plot_trajectory 函数
    plt.figure(figsize=(6, 6))
    
    # 使用 scprep 绘制散点图，颜色表示时间点
    scprep.plot.scatter2d(obs_flat, c=tts, ax=plt.gca())
    
    # 绘制所有轨迹点（黑色半透明）
    # trajs[:, :n, 0] 取前 n 条轨迹的所有时间点的第0维
    plt.scatter(trajs[:, :n, 0], trajs[:, :n, 1], s=0.3, alpha=0.2, c="black", label="Flow")
    
    # 绘制终点（紫色 X 标记）
    # trajs[-1, :, :] 取最后时间步的所有轨迹
    plt.scatter(trajs[-1, :n, 0], trajs[-1, :n, 1], s=6, alpha=1, c="purple", marker="x", label="End")
    
    # 绘制20条代表性轨迹路径（红色线）
    for i in range(20):
        # trajs[:, i, :] 取第 i 条轨迹的所有时间点
        plt.plot(trajs[:, i, 0], trajs[:, i, 1], c="red", alpha=0.5, label="Trajectory" if i == 0 else "")
    
    # 添加图例
    # 参考 plotting.py 中的 plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.legend(["Prior sample z(S)", "Flow", "End", "Trajectory"], loc='best')
    
    # 保存图片到 figs 目录
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{traj_file}_traj.png")
    
    # 显示图形
    plt.show()
