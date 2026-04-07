import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def get_model_name_from_config(model_dir):
    model_name_map = {
        "CFMLitModule": "CFM",
        "RectifiedFlowLitModule": "RectifiedFlow",
        "ActionMatchingLitModule": "ActionMatching",
        "VariancePreservingCFM": "VP-CFM",
        "SBCFMLitModule": "SBCFM",
        "SF2MLitModule": "SF2M"
    }

    model_root_dir = os.path.dirname(os.path.dirname(model_dir))
    config_yaml_path = os.path.join(model_root_dir, '.hydra', 'config.yaml')

    if not os.path.exists(config_yaml_path):
        return get_model_name_from_path(model_dir)

    try:
        with open(config_yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        ot = True

        for line in lines:
            if 'ot_sampler:' in line and ('null' in line or '!!' in line):
                ot = False

        for line in lines:
            if '_target_' in line and 'src.models.' in line:
                target_value = line.split(':', 1)[1].strip()
                if target_value:
                    model_name = target_value.split('.')[-1]
                    model_name = model_name_map.get(model_name, model_name)
                    if model_name == "CFM":
                        model_name = 'OT-CFM' if ot else 'I-CFM'
                    return model_name

    except Exception as e:
        print(f"  警告：解析 {config_yaml_path} 时出错: {e}")

    return get_model_name_from_path(model_dir)

def get_model_name_from_path(model_dir):
    parent_dir = os.path.dirname(model_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    basename = os.path.basename(grandparent_dir)

    if 'actionmatching' in basename.lower():
        return 'ActionMatching'
    elif 'sbcfm' in basename.lower():
        return 'SBCFM'
    elif 'rectifiedflow' in basename.lower():
        return 'RectifiedFlow'
    elif 'sf2m' in basename.lower():
        return 'SF2M'
    elif 'vp-' in basename.lower():
        return 'VP-CFM'
    elif 'cfm-' in basename.lower():
        return 'CFM'
    else:
        parts = basename.split('-')
        return parts[0] if parts else 'Unknown'

def plot_loss_for_experiment(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    train_df = df[['epoch', 'train/loss']].dropna()
    val_df = df[['epoch', 'val/loss']].dropna()

    if len(train_df) == 0 and len(val_df) == 0:
        print(f"  警告：没有找到有效数据，跳过")
        return False

    model_dir = os.path.dirname(csv_path)
    model_name = get_model_name_from_config(model_dir)

    print(f"  绘制模型: {model_name}")

    plt.figure(figsize=(10, 6))

    if len(train_df) > 0:
        plt.plot(train_df['epoch'], train_df['train/loss'], label='train/loss', alpha=0.8)

    if len(val_df) > 0:
        plt.plot(val_df['epoch'], val_df['val/loss'], label='val/loss', alpha=0.8)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{model_name}_loss_curves.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  已保存: {output_path}")
    return True

def main():
    logs_dir = r"C:\Users\Administrator\Desktop\code\conditional-flow-matching\runner\logs\3.3 am"

    csv_files = glob.glob(os.path.join(logs_dir, "**/metrics.csv"), recursive=True)

    if not csv_files:
        print(f"在 {logs_dir} 中未找到 metrics.csv 文件")
        return

    print(f"找到 {len(csv_files)} 个 metrics.csv 文件")

    success_count = 0
    for csv_file in csv_files:
        print(f"\n处理: {csv_file}")
        model_dir = os.path.dirname(csv_file)
        if plot_loss_for_experiment(csv_file, model_dir):
            success_count += 1

    print(f"\n完成！成功绘制 {success_count}/{len(csv_files)} 个实验的损失曲线图")

if __name__ == "__main__":
    main()
