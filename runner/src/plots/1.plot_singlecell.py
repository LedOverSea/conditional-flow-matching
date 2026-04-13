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

# 忽略所有警告
warnings.filterwarnings("ignore")

# -------------------------- 论文绘图风格 --------------------------
set_paper_style()

# -------------------------- 读取数据 --------------------------
data_path = Path("runner") / "data" / "Processed Single-cell RNA Time Series DAta" / "ebdata_v3.h5ad"

adata = sc.read_h5ad(data_path)  # 替换你的路径

# -------------------------- 画图：2张并排 --------------------------
fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN)  # 1行2列

# 获取配色
palette = get_palette()

# ====================== 子图1：每组细胞数量柱状图 ======================
sample_counts = adata.obs['sample_labels'].value_counts().sort_index()
sns.barplot(
    x=sample_counts.index,
    y=sample_counts.values,
    ax=axes[0],
    palette=palette['colors']
)
# axes[0].set_title('每组细胞数量')
axes[0].set_xlabel('样本组')
axes[0].set_ylabel('细胞数量')
axes[0].tick_params(axis='x', rotation=30)  # 防止标签重叠

# 柱子上标数字
for i, v in enumerate(sample_counts.values):
    axes[0].text(i, v + 200, str(v), ha='center', fontweight='normal')

# 添加子图标签
add_panel_label(axes[0], index=0, style='paren')

# ====================== 子图2：时间点分布密度图 ======================
sns.histplot(
    data=adata.obs,
    x='1d-phate-normalized',
    hue='sample_labels',
    ax=axes[1],
    kde=True,
    stat='density',
    palette=palette['colors'],
    alpha=0.6,
    bins=30
)
# axes[1].set_title('时间点分布 (1D-PHATE)')
axes[1].set_xlabel('归一化伪时间')
axes[1].set_ylabel('密度')
axes[1].legend(title='组', bbox_to_anchor=(1.02, 1), loc='upper left')

# 添加子图标签
add_panel_label(axes[1], index=1, style='paren')

# -------------------------- 布局保存 --------------------------
plt.tight_layout()

# 导出图表
export_figure(fig, "dataset_summary", outdir="figures")
print("图表已保存到 figures 目录")

plt.show()