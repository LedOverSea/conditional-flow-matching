import matplotlib.pyplot as plt
from PIL import Image
import os

# 图片路径
img1_path = r'd:\desktop\code\conditional-flow-matching\notes\experiment\imgs\eb umap.png'
img2_path = r'd:\desktop\code\conditional-flow-matching\notes\experiment\imgs\umap ot traj.png'

# 读取图片
img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

# ===================== 修复1：统一两张图片高度 =====================
target_height = 400  # 统一高度，可自行修改
img1 = img1.resize((int(img1.width * target_height / img1.height), target_height), Image.Resampling.LANCZOS)
img2 = img2.resize((int(img2.width * target_height / img2.height), target_height), Image.Resampling.LANCZOS)

# 创建画布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ===================== 修复2：标号放在图外侧，不覆盖图片 =====================
# 子图1
axes[0].imshow(img1)
axes[0].axis('off')
axes[0].set_title('(a)', fontsize=16, fontweight='bold', pad=10)  # 放在上方，不覆盖

# 子图2
axes[1].imshow(img2)
axes[1].axis('off')
axes[1].set_title('(b)', fontsize=16, fontweight='bold', pad=10)  # 放在上方，不覆盖

# 紧凑布局
plt.tight_layout()
plt.subplots_adjust(wspace=0.1)  # 控制两张图间距

# 保存（bbox_inches 保证不裁切）
output_path = r'd:\desktop\code\conditional-flow-matching\combined_images.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"已生成整齐排列的组合图：{output_path}")