#!/usr/bin/env python3
"""
Script for plotting m sensitivity analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 导入模板配置
import os
import sys

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from template.plot_journal_template import set_paper_style, get_palette_colors, DOUBLE_COLUMN, add_panel_label, export_figure, TRIPLE_ROW

# 设置论文作图风格
set_paper_style()

data = {
    'm/M': [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00],
    'W1': [0.381, 0.356, 0.334, 0.2888, 0.323, 0.331, 0.342],
    'W2': [0.524, 0.481, 0.436, 0.398, 0.412, 0.427, 0.451],
    'MSE': [0.0094, 0.0086, 0.0079, 0.0072, 0.0074, 0.0077, 0.0081]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 找到最优值的位置
optimal_idx = df['W1'].idxmin()
optimal_m = df.loc[optimal_idx, 'm/M']

# 创建3张子图的布局
fig, axes = plt.subplots(1, 3, figsize=TRIPLE_ROW)

# 获取模板中的颜色方案
colors = get_palette_colors()

# 绘制W1指标
ax = axes[0]
sns.lineplot(data=df, x='m/M', y='W1', marker='o', ax=ax, color=colors[0], linewidth=1.5, markersize=6)
# 标记最优值
# ax.scatter([optimal_m], [df.loc[optimal_idx, 'W1']], color='red', s=100, zorder=5)
# ax.axvline(x=optimal_m, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('$m/M$')
ax.set_ylabel('$W_1$')
#ax.grid(True, alpha=0.3)
add_panel_label(ax, index=0, style='paren')

# 绘制W2指标
ax = axes[1]
sns.lineplot(data=df, x='m/M', y='W2', marker='o', ax=ax, color=colors[1], linewidth=1.5, markersize=6)
# 标记最优值
# ax.scatter([optimal_m], [df.loc[optimal_idx, 'W2']], color='red', s=100, zorder=5)
# ax.axvline(x=optimal_m, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('$m/M$')
ax.set_ylabel('$W_2$')
# ax.grid(True, alpha=0.3)
add_panel_label(ax, index=1, style='paren')

# 绘制MSE指标
ax = axes[2]
sns.lineplot(data=df, x='m/M', y='MSE', marker='o', ax=ax, color=colors[2], linewidth=1.5, markersize=6)
# 标记最优值
# ax.scatter([optimal_m], [df.loc[optimal_idx, 'MSE']], color='red', s=100, zorder=5)
# ax.axvline(x=optimal_m, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('$m/M$')
ax.set_ylabel('MSE')
#ax.grid(True, alpha=0.3)
add_panel_label(ax, index=2, style='paren')

# 调整布局
plt.tight_layout()

# 导出图表
output_dir = r"d:\desktop\code\conditional-flow-matching\runner\src\plots"
export_figure(fig, 'm_sensitivity_analysis', outdir=output_dir)
print(f"敏感性分析图表已保存到: {output_dir}")

plt.show()
