#!/usr/bin/env python3
"""
超参数搜索结果分析脚本
读取multiruns文件夹下的配置文件和metrics，生成超参数矩阵
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
from collections import defaultdict

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
                return non_null_values.iloc[-1]
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
    if len(sys.argv) != 2:
        print("用法: python matrix_hparam_search.py <metric_name>")
        print("例如: python matrix_hparam_search.py test/1-Wasserstein")
        return
    
    metric_name = sys.argv[1]
    
    # 获取超参数组合
    hparam_data = get_hparam_combinations()
    
    # 生成并显示矩阵
    generate_hparam_matrix(hparam_data, metric_name)


if __name__ == "__main__":
    main()
