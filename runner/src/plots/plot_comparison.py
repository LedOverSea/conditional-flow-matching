"""
绘制rf模型性能对比柱形图
输入: 4个模型的5个指标数据
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# 设置全局字体为 Times New Roman，符合学术论文要求
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

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

# 学术论文友好的配色方案 - 使用柔和且专业的颜色
# 参考 ColorBrewer 配色方案，适合学术论文
colors = [
    '#0072B2',  # 深蓝色
    '#D55E00',  # 橙红色
    '#009E73',  # 深绿色
    '#CC79A7',  # 淡紫色
    '#F0E442',  # 淡黄色
]

# 绘制每个指标的柱形
for i, metric_name in enumerate(metric_names):
    offset = (i - len(metric_names) / 2 + 0.5) * width
    bars = ax.bar(x + offset, data[:, i], width, label=metric_name, color=colors[i])

# 设置标签
ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')

# 设置x轴刻度标签
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)

# 添加图例，放置在图外以避免遮挡
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, frameon=False)

# 添加网格线，使用较浅的颜色
ax.grid(axis='y', alpha=0.2, linestyle='--')

# 调整布局，为图例预留空间
plt.subplots_adjust(right=0.85)
plt.tight_layout()

# 保存图片到当前目录
save_path = os.path.join(os.path.dirname(__file__), "model_comparison_academic.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"图片已保存到: {save_path}")

# 显示图形
plt.show()
