import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# ---------- 1. 合并所有CSV标签文件 (与之前相同) ----------
data_dir = "data_more"
csv_files = glob.glob(os.path.join(data_dir, "labels_*.csv"))
df_list = []
for file in csv_files:
    image_folder_name = "images_" + os.path.basename(file).split('_')[1].split('.')[0]
    df_temp = pd.read_csv(file)
    df_temp['filename'] = df_temp['filename'].apply(lambda x: os.path.join(image_folder_name, x))
    df_list.append(df_temp)

master_df = pd.concat(df_list, ignore_index=True)
print(f"成功合并 {len(csv_files)} 个CSV文件, 总计 {len(master_df)} 条数据。")

# 可视化原始数据分布
print("正在分析【原始】转向角分布...")
plt.figure(figsize=(10, 6))
plt.hist(master_df['steer'], bins=50, color='red', alpha=0.7)
plt.title('Original Steering Angle Distribution (Before Balancing)')
plt.xlabel('Steering Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# ---------- 2. 【新增】数据过滤与平衡 ----------
print("-" * 30)
print("开始进行数据过滤与平衡...")

# 设定一个阈值，小于这个绝对值的转向都算作“直行”
# 这个值可以微调，0.01 到 0.05 之间都是合理的选择
steer_threshold = 0.02

# 将数据分为“直行”和“转弯”两部分
turning_data = master_df[abs(master_df['steer']) > steer_threshold]
straight_data = master_df[abs(master_df['steer']) <= steer_threshold]

print(f"原始数据中，转弯数据: {len(turning_data)} 条, 直行数据: {len(straight_data)} 条")

# 对直行数据进行随机欠采样（undersampling），比如只保留其中的 25%
 # frac=0.1 意味着只保留10%的直行数据，转弯样本占比更高
straight_data_sampled = straight_data.sample(frac=0.1, random_state=42)

print(f"从直行数据中采样后，保留: {len(straight_data_sampled)} 条")

# 将保留的直行数据和所有的转弯数据合并成我们最终的平衡数据集
balanced_df = pd.concat([turning_data, straight_data_sampled], ignore_index=True)

# 【重要】最后一定要打乱整个数据集的顺序，否则模型会先学转弯再学直行
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"数据平衡后，总数据量为: {len(balanced_df)} 条")
print("-" * 30)


# 可视化平衡后的数据分布
print("正在分析【平衡后】的转向角分布...")
plt.figure(figsize=(10, 6))
plt.hist(balanced_df['steer'], bins=50, color='blue', alpha=0.7)
plt.title('Balanced Steering Angle Distribution (After Filtering)')
plt.xlabel('Steering Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# ---------- 3. 划分训练集和验证集 (使用平衡后的数据) ----------
print("正在使用【平衡后】的数据集划分训练集和验证集...")

# 使用我们新创建的 balanced_df 进行划分
train_df, validation_df = train_test_split(
    balanced_df,
    test_size=0.2,
    random_state=42
)

# 保存文件
train_csv_path = os.path.join(data_dir, "train_labels.csv")
validation_csv_path = os.path.join(data_dir, "validation_labels.csv")
train_df.to_csv(train_csv_path, index=False)
validation_df.to_csv(validation_csv_path, index=False)

print(f"数据集划分完成:")
print(f"训练集大小: {len(train_df)} 条数据")
print(f"验证集大小: {len(validation_df)} 条数据")
print(f"训练集标签已保存至: {train_csv_path}")
print(f"验证集标签已保存至: {validation_csv_path}")