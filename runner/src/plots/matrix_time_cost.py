#!/usr/bin/env python3
"""
统计模型运行时间的脚本
读取multiruns文件夹下的配置文件，提取exec_time和模型信息，生成运行时间矩阵
"""

import os
import re
import numpy as np
import pandas as pd

# 配置路径
MULTIRUNS_DIR = r'd:\desktop\code\conditional-flow-matching\runner\logs\3.29'


def get_model_name_from_config(config_path):
    """从配置文件中提取模型名称"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找ot_sampler值
        ot_sampler_match = re.search(r'ot_sampler:\s*(\w+)', content)
        if ot_sampler_match:
            ot_sampler_value = ot_sampler_match.group(1)
            
            # 根据ot_sampler值生成模型名称
            if ot_sampler_value == 'null':
                return 'I-CFM'  # 不使用OT采样
            elif ot_sampler_value == 'exact':
                return 'OT-CFM'
            elif ot_sampler_value == 'sinkhorn':
                return 'EOT-CFM'
            elif ot_sampler_value == 'unbalanced':
                return 'UOT-CFM'
            elif ot_sampler_value == 'partial':
                return 'POT-CFM'
            else:
                return f'OT-CFM ({ot_sampler_value})'
    except Exception as e:
        print(f"读取配置文件时出错: {e}")
    return 'Unknown'


def get_datamodule_from_path(path):
    """从路径中提取数据模块信息"""
    try:
        # 从路径中提取数据模块名称
        import os
        # 获取3.29文件夹下的子文件夹名称
        parts = path.split(os.sep)
        # 找到3.29文件夹的索引
        if '3.29' in parts:
            idx = parts.index('3.29')
            if idx + 1 < len(parts):
                return parts[idx + 1]
    except Exception as e:
        print(f"从路径提取数据模块时出错: {e}")
    return 'Unknown'


def get_sigma_min_from_config(config_path):
    """从配置文件中提取sigma_min值"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找sigma_min值
        sigma_min_match = re.search(r'sigma_min:\s*([\d.]+)', content)
        if sigma_min_match:
            return sigma_min_match.group(1)
    except Exception as e:
        print(f"读取配置文件时出错: {e}")
    return 'Unknown'


def get_lr_from_config(config_path):
    """从配置文件中提取optimizer.lr值"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找optimizer.lr值
        lr_match = re.search(r'optimizer:\s*.*?lr:\s*([\d.]+)', content, re.DOTALL)
        if lr_match:
            return lr_match.group(1)
    except Exception as e:
        print(f"读取配置文件时出错: {e}")
    return 'Unknown'


def get_exec_time_from_log(log_path):
    """从exec_time.log中提取执行时间"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找执行时间
        exec_time_match = re.search(r"execution time:\s*([\d.]+)\s*\(s\)", content)
        if exec_time_match:
            return float(exec_time_match.group(1))
    except Exception as e:
        print(f"读取exec_time.log时出错: {e}")
    return None


def collect_hparam_combinations():
    """收集所有子文件夹的超参数组合"""
    hparam_combinations = []
    
    # 遍历所有子文件夹，包括嵌套的子文件夹
    for root, dirs, files in os.walk(MULTIRUNS_DIR):
        # 查找配置文件
        config_path = os.path.join(root, '.hydra', 'config.yaml')
        if not os.path.exists(config_path):
            continue
        
        # 获取模型名称、数据模块和超参数
        model_name = get_model_name_from_config(config_path)
        datamodule = get_datamodule_from_path(root)
        sigma_min = get_sigma_min_from_config(config_path)
        lr = get_lr_from_config(config_path)
        
        # 查找exec_time.log文件
        exec_time_log = os.path.join(root, 'exec_time.log')
        exec_time = None
        
        if os.path.exists(exec_time_log):
            # 从exec_time.log中提取执行时间
            exec_time = get_exec_time_from_log(exec_time_log)
        
        if exec_time is None:
            # 如果找不到exec_time.log文件，使用文件夹的修改时间作为近似
            exec_time = os.path.getmtime(root)
        
        hparam_combinations.append({
            'datamodule': datamodule,
            'model': model_name,
            'sigma_min': sigma_min,
            'lr': lr,
            'exec_time': exec_time
        })
    
    return hparam_combinations


def generate_time_matrix(hparam_combinations):
    """生成运行时间矩阵"""
    if not hparam_combinations:
        print("未找到有效的超参数组合数据")
        return
    
    # 提取所有唯一的datamodule和模型名称
    datamodules = list(set(item['datamodule'] for item in hparam_combinations))
    models = list(set(item['model'] for item in hparam_combinations))
    
    # 按照字母顺序排序
    datamodules.sort()
    models.sort()
    
    # 创建矩阵，行是datamodule，列是模型
    matrix = np.full((len(datamodules), len(models)), np.nan)
    
    # 填充矩阵
    for item in hparam_combinations:
        datamodule = item['datamodule']
        model = item['model']
        exec_time = item['exec_time']
        
        if datamodule in datamodules and model in models:
            row_idx = datamodules.index(datamodule)
            col_idx = models.index(model)
            matrix[row_idx, col_idx] = exec_time
    
    # 创建DataFrame用于显示
    matrix_df = pd.DataFrame(matrix, index=datamodules, columns=models)
    matrix_df.index.name = 'datamodule'
    matrix_df.columns.name = 'model'
    
    # 添加平均值行
    matrix_df.loc['Average'] = matrix_df.mean()
    
    # 显示矩阵
    print("运行时间矩阵 (单位: 秒)")
    print("-" * 80)
    print(matrix_df)
    print("-" * 80)


def main():
    """主函数"""
    # 收集超参数组合
    hparam_combinations = collect_hparam_combinations()
    
    # 生成矩阵
    generate_time_matrix(hparam_combinations)


if __name__ == "__main__":
    main()
