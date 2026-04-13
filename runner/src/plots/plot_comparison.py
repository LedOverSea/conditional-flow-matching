"""
绘制rf模型性能对比柱形图
输入: 4个模型的5个指标数据
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import sys

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模板配置
from template.plot_journal_template import set_paper_style, get_palette_colors, SINGLE_COLUMN, DOUBLE_COLUMN, export_figure

# 设置论文作图风格
set_paper_style()

# 模拟输入数据
# 模型数据: (4, 5) - 4个模型, 5个指标
# model_data = [
#     [0.89602294,    0.970474617,    0.367643494,    0.506564893,    0.501846562
# ],  
#     [0.535054004,    0.636390438,    0.128863793,    0.283978006,    0.244753775
# ],   
#     [1.460404134,    1.579791264,    0.192723316,    0.421085011,    0.381166165
# ],  
#     [1.496125926,    1.614741789,    0.313336757,    0.502493915,    0.444758058],   
# ]
model_data = [
    [2.183429976,    2.183429978,    3.471384163,    1.543918113,    1.482632498
],  
    [0.519673833,    0.650997909,    0.069637249,    0.257417902,    0.24246722
],     
]

# 模型名字列表
# model_names = ["eb(phate) unmodified", "eb(phate) modified", "eb(pca) unmodified", "eb(pca) modified"]
model_names = ["batch size = 1", "batch size = 128"]

# 指标名字列表
metric_names = ["1-Wasserstein", "2-Wasserstein", "MSE", "L2", "L1"]

# 转换为numpy数组便于处理
data = np.array(model_data)

# 设置图形大小
fig, ax = plt.subplots(figsize=(10, 6))

# 设置柱形图参数
x = np.arange(len(model_names))  # 横坐标位置
width = 0.12  # 柱形宽度，适当减小以避免拥挤

# 使用模板中的颜色方案
colors = get_palette_colors()

# 绘制每个指标的柱形
for i, metric_name in enumerate(metric_names):
    offset = (i - len(metric_names) / 2 + 0.5) * width
    bars = ax.bar(x + offset, data[:, i], width, label=metric_name, color=colors[i])

# 设置标签
ax.set_xlabel('Batch Size')
ax.set_ylabel('Value')
ax.set_title('Model Performance Comparison')

# 设置x轴刻度标签
ax.set_xticks(x)
ax.set_xticklabels(model_names)

# 添加图例，放置在图外以避免遮挡
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

# 添加网格线，使用较浅的颜色
ax.grid(axis='y', alpha=0.2, linestyle='--')

# 调整布局，为图例预留空间
plt.subplots_adjust(right=0.85)
plt.tight_layout()

# 保存图片到当前目录，使用模板中的导出函数
export_figure(fig, "model_comparison_academic", outdir=os.path.dirname(__file__))

print(f"图片已保存到: {os.path.dirname(__file__)}")

# 显示图形
plt.show()
