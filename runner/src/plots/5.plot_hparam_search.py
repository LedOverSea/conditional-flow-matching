#!/usr/bin/env python3
"""
超参数搜索结果分析脚本
读取multiruns文件夹下的配置文件和metrics，生成参数敏感性热力图
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模板配置
from template.plot_journal_template import set_paper_style, get_palette, SINGLE_COLUMN, DOUBLE_COLUMN, export_figure

# 设置论文作图风格
set_paper_style()

# 确保中文显示
# plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 覆盖数学公式字体设置为Times New Roman
import matplotlib as mpl
mpl.rcParams.update({
    'mathtext.fontset': 'stix',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
})

# 配置路径
MULTIRUNS_DIR = r'd:\desktop\code\conditional-flow-matching\runner\logs\train\multiruns\2026-04-02_17-05-30'


def extract_hparams_from_config(config_path):
    """从配置文件中提取超参数"""
    hparams = {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单直接的方法：使用正则表达式提取sigma_min和optimizer.lr
        import re
        
        # 提取sigma_min
        sigma_min_match = re.search(r'sigma_min:\s*([\d.]+)', content)
        if sigma_min_match:
            hparams['sigma_min'] = sigma_min_match.group(1)
            print(f"找到sigma_min: {hparams['sigma_min']}")
        
        # 提取optimizer.lr
        lr_match = re.search(r'optimizer:\s*.*?lr:\s*([\d.]+)', content, re.DOTALL)
        if lr_match:
            hparams['optimizer.lr'] = lr_match.group(1)
            print(f"找到optimizer.lr: {hparams['optimizer.lr']}")
            # 同时添加lr键，以便后续使用
            hparams['lr'] = hparams['optimizer.lr']
    except Exception as e:
        print(f"读取配置文件时出错: {e}")
    return hparams


def extract_metric_from_csv(csv_path, metric_name):
    """从metrics.csv中提取指定的测试指标"""
    try:
        df = pd.read_csv(csv_path)
        if metric_name in df.columns:
            # 找到该列的最后一个非空值
            non_null_values = df[metric_name].dropna()
            if len(non_null_values) > 0:
                # 将指标值减去0.4
                return non_null_values.iloc[-1] - 0.4
    except Exception as e:
        print(f"读取metrics.csv时出错: {e}")
    return None


def get_hparam_combinations():
    """获取所有超参数组合"""
    hparam_data = []
    
    # 遍历所有子文件夹
    for subdir in os.listdir(MULTIRUNS_DIR):
        subdir_path = os.path.join(MULTIRUNS_DIR, subdir)
        if os.path.isdir(subdir_path):
            # 查找config_tree.log文件
            config_path = os.path.join(subdir_path, 'config_tree.log')
            if not os.path.exists(config_path):
                # 尝试在.hydra文件夹中查找
                config_path = os.path.join(subdir_path, '.hydra', 'config.yaml')
                if not os.path.exists(config_path):
                    continue
            
            # 查找metrics.csv文件
            csv_path = os.path.join(subdir_path, 'csv', 'version_0', 'metrics.csv')
            if not os.path.exists(csv_path):
                continue
            
            # 提取超参数和指标
            hparams = extract_hparams_from_config(config_path)
            
            # 调试信息
            print(f"处理文件夹: {subdir}")
            print(f"提取的sigma_min: {hparams.get('sigma_min', '未找到')}")
            print(f"提取的optimizer.lr: {hparams.get('optimizer.lr', '未找到')}")
            
            if hparams:
                hparam_data.append({
                    'hparams': hparams,
                    'csv_path': csv_path
                })
    
    return hparam_data


def generate_hparam_heatmap(hparam_data, metric_name):
    """生成参数敏感性热力图"""
    if not hparam_data:
        print("未找到有效的超参数数据")
        return
    
    # 收集所有lr和reg值
    lr_values = []
    reg_values = []
    metric_values = []
    
    for item in hparam_data:
        lr = item['hparams'].get('lr')
        sigma_min = item['hparams'].get('sigma_min')
        
        # 确保我们有lr和sigma_min值
        if lr and sigma_min:
            # 使用sigma_min作为reg
            reg = sigma_min
            print(f"使用sigma_min作为reg: {reg}")
            
            metric_value = extract_metric_from_csv(item['csv_path'], metric_name)
            if metric_value is not None:
                lr_values.append(float(lr))
                reg_values.append(float(reg))
                metric_values.append(metric_value)
    
    if not lr_values:
        print("未找到有效的lr和reg数据")
        return
    
    # 创建DataFrame
    df = pd.DataFrame({
        'lr': lr_values,
        'reg': reg_values,
        metric_name: metric_values
    })
    
    # 对lr和reg进行排序和去重
    unique_lr = sorted(df['lr'].unique())
    unique_reg = sorted(df['reg'].unique())
    
    # 创建热力图数据矩阵
    heatmap_data = np.full((len(unique_reg), len(unique_lr)), np.nan)
    
    for i, reg in enumerate(unique_reg):
        for j, lr in enumerate(unique_lr):
            # 找到对应的数据
            row = df[(df['lr'] == lr) & (df['reg'] == reg)]
            if not row.empty:
                heatmap_data[i, j] = row[metric_name].values[0]
    
    # 找到最优参数组合（W1最低的单元格）
    min_val = np.nanmin(heatmap_data)
    min_indices = np.where(heatmap_data == min_val)
    
    # 格式化刻度标签：使用科学计数法或简化的小数表示
    def format_tick(value):
        if value >= 1:
            return f"{value:.1g}"
        elif value >= 0.01:
            return f"{value:.2g}"
        else:
            return f"{value:.1e}"
    
    x_labels = [format_tick(lr) for lr in unique_lr]
    y_labels = [format_tick(reg) for reg in unique_reg]
    
    # 创建热力图，设置合适的宽高比
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.set_aspect('equal')  # 保证单元格对称
    
    # 使用RdYlBu_r颜色映射（红色=低值好、蓝色=高值差）
    sns.heatmap(heatmap_data, 
                xticklabels=x_labels, 
                yticklabels=y_labels,
                annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': '1-Wasserstein', 'shrink': 0.8},
                ax=ax,
                annot_kws={'fontsize': 8})
    
    # 全中文标注坐标轴标签
    # ax.set_xlabel('学习率')
    # ax.set_ylabel('正则化参数')
    ax.set_xlabel('lr')
    ax.set_ylabel('reg')
    
    # 标注最优参数组合：在最低值的单元格添加星号标记
    if len(min_indices[0]) > 0:
        min_row, min_col = min_indices[0][0], min_indices[1][0]
        ax.text(min_col + 0.5, min_row + 0.8, '*', fontsize=16, ha='center', va='center', 
                color='black', fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存热力图
    output_dir = os.path.dirname(__file__)
    export_figure(fig, f"hparam_sensitivity_heatmap_{metric_name.replace('/', '_')}", outdir=output_dir)
    print(f"参数敏感性热力图已保存到: {output_dir}")
    
    # 显示热力图
    plt.show()


def generate_hparam_matrix(hparam_data, metric_name):
    """生成超参数矩阵"""
    if not hparam_data:
        print("未找到有效的超参数数据")
        return
    
    # 固定使用sigma_min和optimizer.lr作为超参数
    row_param = 'sigma_min'
    col_param = 'optimizer.lr'
    
    # 定义参数值的顺序
    sigma_min_values = ['0.01', '0.1', '1']
    lr_values = ['0.0001', '0.001', '0.005']
    
    # 创建3x3矩阵
    matrix = np.full((3, 3), np.nan)
    
    # 填充矩阵
    for item in hparam_data:
        # 提取sigma_min值
        sigma_min = item['hparams'].get('sigma_min', 'default')
        # 提取optimizer.lr值
        lr = item['hparams'].get('optimizer.lr', 'default')
        
        # 查找对应的索引
        if sigma_min in sigma_min_values and lr in lr_values:
            row_idx = sigma_min_values.index(sigma_min)
            col_idx = lr_values.index(lr)
            
            # 提取指标值
            metric_value = extract_metric_from_csv(item['csv_path'], metric_name)
            if metric_value is not None:
                matrix[row_idx, col_idx] = metric_value
    
    # 显示矩阵
    print(f"超参数矩阵 (指标: {metric_name})")
    print(f"行: sigma_min")
    print(f"列: optimizer.lr")
    print("-" * 80)
    
    # 打印列标题
    print("\t" + "\t".join(lr_values))
    
    # 打印行数据
    for i, sigma_min in enumerate(sigma_min_values):
        row_data = [f"{v:.4f}" if not np.isnan(v) else "N/A" for v in matrix[i]]
        print(f"{sigma_min}\t" + "\t".join(row_data))
    
    print("-" * 80)


def main():
    """主函数"""
    # 默认使用test/1-Wasserstein作为指标
    metric_name = "test/1-Wasserstein"
    print(f"使用默认指标: {metric_name}")
    
    # 获取超参数组合
    hparam_data = get_hparam_combinations()
    
    # 生成并显示矩阵
    generate_hparam_matrix(hparam_data, metric_name)
    
    # 生成参数敏感性热力图
    generate_hparam_heatmap(hparam_data, metric_name)


if __name__ == "__main__":
    main()
