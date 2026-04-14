#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
# 导入你项目的论文绘图模板（保持风格统一）
from template.plot_journal_template import set_paper_style, export_figure

# 初始化论文绘图风格
set_paper_style()

# ===================== 1. 定义 ODE 向量场（核心：dx/dt = v(x,t)）=====================
def ode_vector_field(x, t, target):
    """
    流模型的核心 ODE：dx/dt = v(x,t)
    这里用极简线性向量场（方便可视化，可替换为你的模型向量场）
    x: 当前点坐标 [x, y]
    t: 时间步
    target: 目标点坐标
    """
    # 向量场：向目标点收敛（还原路径的物理意义）
    v = -(x - target)  
    return v

# ===================== 2. 数值积分解 ODE（还原路径）=====================
# 超参数
t_start = 0.0    # 起始时间
t_end = 1.0      # 终止时间
num_steps = 20   # 积分步数
x0 = np.array([-2.5, 2.5])  # 初始点 (t=0)
target = np.array([1.5, -1.5])  # 目标点 (t=1)
t = np.linspace(t_start, t_end, num_steps)  # 时间序列

# 解 ODE：数值积分得到完整路径
path = odeint(ode_vector_field, x0, t, args=(target,))

# ===================== 3. 绘制 ODE 积分还原路径示意图 =====================
fig, ax = plt.subplots(figsize=(4, 3))  # 论文单栏尺寸

# 1. 绘制向量场（背景：展示驱动力）
x_grid, y_grid = np.meshgrid(np.linspace(-3, 2, 15), np.linspace(-2, 3, 15))
vx = -(x_grid - target[0])
vy = -(y_grid - target[1])
ax.quiver(x_grid, y_grid, vx, vy, color='gray', alpha=0.3, width=0.003)

# 2. 绘制 ODE 积分还原的核心路径
ax.plot(path[:, 0], path[:, 1], 
        color='#2E86AB', linewidth=2.5, label='ODE 积分还原路径')

# 3. 标记起点、终点
ax.scatter(x0[0], x0[1], color='#A23B72', s=80, zorder=5, label='初始点 $t=0$')
ax.scatter(target[0], target[1], color='#F18F01', s=80, zorder=5, label='目标点 $t=1$')

# 4. 绘制时间箭头（标注积分方向）
for i in range(0, len(path)-1, 4):
    ax.arrow(path[i,0], path[i,1], 
             path[i+1,0]-path[i,0], path[i+1,1]-path[i,1],
             head_width=0.1, color='#2E86AB', zorder=4)

# 5. 图表美化
ax.set_xlabel('$x_1$', fontsize=10)
ax.set_ylabel('$x_2$', fontsize=10)
ax.set_title('ODE 数值积分 · 样本路径还原', fontsize=11)
ax.legend(fontsize=8, frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.set_xlim(-3, 2)
ax.set_ylim(-2, 3)
plt.tight_layout()

# 6. 导出图片（适配你的模板）
# export_figure(fig, 'ode_integration_path', outdir='./plots')
plt.show()