#!/usr/bin/env python3
"""
Script for analyzing m sensitivity results
Reads all CSV files in m sensitivity folder and generates a summary CSV
"""

import os
import pandas as pd
import glob
from pathlib import Path

# 定义路径
logs_dir = r"d:\desktop\code\conditional-flow-matching\runner\logs\m sensitivity"
output_file = os.path.join(logs_dir, "m_sensitivity_summary.csv")

# 收集所有子文件夹（如m=0.25, m=0.5等）
subfolders = [f for f in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, f))]
subfolders.sort()  # 按顺序排序

# 存储所有数据
all_data = []

for subfolder in subfolders:
    # 构建子文件夹路径
    subfolder_path = os.path.join(logs_dir, subfolder)
    
    # 查找csv文件
    csv_files = glob.glob(os.path.join(subfolder_path, "**/metrics.csv"), recursive=True)
    
    if csv_files:
        # 取第一个找到的csv文件
        csv_file = csv_files[0]
        print(f"Processing {subfolder}: {csv_file}")
        
        try:
            # 读取csv文件
            df = pd.read_csv(csv_file)
            
            # 提取含test的列
            test_columns = [col for col in df.columns if 'test' in col.lower()]
            
            if test_columns:
                # 提取最后一行的测试数据
                test_data = {}
                for col in test_columns:
                    # 找到该列的最后一个非空值
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        test_data[col] = non_null_values.iloc[-1]
                    else:
                        test_data[col] = None
                
                # 添加文件夹名称作为标识
                test_data['m_value'] = subfolder
                all_data.append(test_data)
            else:
                print(f"No test columns found in {csv_file}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    else:
        print(f"No metrics.csv files found in {subfolder}")

# 生成汇总DataFrame
if all_data:
    summary_df = pd.DataFrame(all_data)
    
    # 将m_value列移到第一列
    cols = ['m_value'] + [col for col in summary_df.columns if col != 'm_value']
    summary_df = summary_df[cols]
    
    # 保存为csv文件
    summary_df.to_csv(output_file, index=False)
    print(f"Summary CSV generated at: {output_file}")
    print(f"\nGenerated summary:")
    print(summary_df)
else:
    print("No data found to generate summary")
