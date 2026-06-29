
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
import os
import sys
from pathlib import Path

# 添加项目根目录到Python模块搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

# 导入模板配置
from template.plot_journal_template import set_paper_style, get_palette, DOUBLE_COLUMN, add_panel_label, export_figure
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

# -------------------------- 论文绘图风格 --------------------------
set_paper_style()

# -------------------------- 读取数据 --------------------------
data_path = Path("runner") / "data" / "Processed Single-cell RNA Time Series DAta" / "ebdata_v3.h5ad"
adata = sc.read_h5ad(data_path)

# -------------------------- 关键修复1：动态适配配色 --------------------------
# 获取样本组数量 & 生成适配的配色
sample_labels = adata.obs['sample_labels'].unique()
n_groups = len(sample_labels)
palette = get_palette()
# print(adata.obs['sample_labels'].unique())
# 如果类别数超过默认配色长度，扩展配色（使用seaborn的tab10调色板）
if n_groups > len(palette['colors']):
    custom_colors = sns.color_palette("tab10", n_groups).as_hex()  # 转为16进制字符串（匹配模板格式）
else:
    custom_colors = palette['colors'][:n_groups]  # 仅取需要的数量

# -------------------------- 画图：2张并排 --------------------------
# 使用模板中定义的DOUBLE_COLUMN尺寸，适合1*2布局
fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.20))  

# ====================== 子图1：每组细胞数量柱状图 ======================
sample_counts = adata.obs['sample_labels'].value_counts().sort_index()
# 使用适配后的配色
sns.barplot(
    x=sample_counts.index,
    y=sample_counts.values,
    ax=axes[0],
    palette=custom_colors,
    width=0.6  # 调整柱形宽度，默认是0.8
)
axes[0].set_xlabel('样本组')
axes[0].set_ylabel('细胞数量')
axes[0].tick_params(axis='x', rotation=30)

# 柱子上标数字
for i, v in enumerate(sample_counts.values):
    axes[0].text(i, v + 200, str(v), ha='center', fontweight='normal')

# 添加子图标签
add_panel_label(axes[0], index=0, style='paren')

# ====================== 子图2：时间点分布密度图 ======================
# 关键修复3：显式控制图例的生成与位置
sns.histplot(
    data=adata.obs,
    x='1d-phate-normalized',
    hue='sample_labels',
    ax=axes[1],
    kde=True,
    stat='density',
    palette=custom_colors,
    alpha=0.6,
    bins=30
)
axes[1].set_xlabel('归一化伪时间')
axes[1].set_ylabel('密度')

# 关键修复4：手动调整图例位置 & 样式，确保显示完整
# legend = axes[1].legend(
#     title='group',
#     bbox_to_anchor=(0.9, 1),  # 图例放在子图右侧（不挤压绘图区域）
#     loc='upper left',
#     borderaxespad=0.0,
#     frameon=False  # 匹配模板的无框样式
# )
# 可选：如果标签太长，设置图例文字换行/缩小
# plt.setp(legend.get_texts(), fontsize=8)  # 微调字体大小
# plt.setp(legend.get_title(), fontsize=9)

# 添加子图标签
add_panel_label(axes[1], index=1, style='paren')

# -------------------------- 布局保存 --------------------------
# 关键修复5：调整tight_layout的padding，给图例留空间
plt.tight_layout(rect=[0, 0, 0.9, 1])  # 右侧留10%空间给图例

# 导出图表（确保导出时包含图例）
export_figure(fig, "dataset_summary", outdir="figures", dpi=600)
print("图表已保存到 figures 目录")

plt.show()