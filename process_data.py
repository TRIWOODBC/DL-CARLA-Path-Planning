"""
数据处理脚本 - 合并多个数据目录，平衡数据，划分训练/验证/测试集
"""
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from config import (
    STEER_THRESHOLD, STRAIGHT_SAMPLE_RATIO, 
    TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO
)

# ==================== 配置：要合并的数据目录 ====================
DATA_DIRS = [
    "data_v3",           # Town01 数据
    "data_v3_town2_3",   # Town02/03 数据
    "data_town04",       # Town04 弯道数据（采集后添加）
    # "data_more",       # 旧数据（如需要可取消注释）
]

# 输出目录（合并后的数据保存位置）
OUTPUT_DIR = "data_merged"

def load_data_from_dir(data_dir):
    """从单个目录加载所有CSV标签"""
    csv_files = glob.glob(os.path.join(data_dir, "labels_*.csv"))
    if not csv_files:
        print(f"  ⚠️ {data_dir} 中没有找到标签文件")
        return pd.DataFrame()
    
    df_list = []
    for file in csv_files:
        # 提取场景编号
        scenario_id = os.path.basename(file).split('_')[1].split('.')[0]
        image_folder = f"images_{scenario_id}"
        
        df_temp = pd.read_csv(file)
        # 添加完整的图像路径（包含数据目录）
        df_temp['filename'] = df_temp['filename'].apply(
            lambda x: os.path.join(data_dir, image_folder, x)
        )
        df_temp['source_dir'] = data_dir  # 记录数据来源
        df_list.append(df_temp)
    
    return pd.concat(df_list, ignore_index=True)


def balance_data(df, steer_threshold, straight_ratio):
    """平衡转向数据，减少直行数据占比"""
    # 分离转弯和直行数据
    turning_data = df[abs(df['steer']) > steer_threshold]
    straight_data = df[abs(df['steer']) <= steer_threshold]
    
    # 进一步细分：左转和右转
    left_turn = df[df['steer'] < -steer_threshold]
    right_turn = df[df['steer'] > steer_threshold]
    
    print(f"  原始数据分布:")
    print(f"    - 左转: {len(left_turn)} ({100*len(left_turn)/len(df):.1f}%)")
    print(f"    - 直行: {len(straight_data)} ({100*len(straight_data)/len(df):.1f}%)")
    print(f"    - 右转: {len(right_turn)} ({100*len(right_turn)/len(df):.1f}%)")
    
    # 对直行数据欠采样
    straight_sampled = straight_data.sample(frac=straight_ratio, random_state=42)
    
    # 合并
    balanced = pd.concat([turning_data, straight_sampled], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱
    
    # 统计平衡后
    left_bal = balanced[balanced['steer'] < -steer_threshold]
    right_bal = balanced[balanced['steer'] > steer_threshold]
    straight_bal = balanced[abs(balanced['steer']) <= steer_threshold]
    
    print(f"  平衡后数据分布:")
    print(f"    - 左转: {len(left_bal)} ({100*len(left_bal)/len(balanced):.1f}%)")
    print(f"    - 直行: {len(straight_bal)} ({100*len(straight_bal)/len(balanced):.1f}%)")
    print(f"    - 右转: {len(right_bal)} ({100*len(right_bal)/len(balanced):.1f}%)")
    
    return balanced


def plot_distribution(df, title, color='blue'):
    """绘制转向分布图"""
    plt.figure(figsize=(10, 6))
    plt.hist(df['steer'], bins=50, color=color, alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel('Steering Value')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("数据处理脚本 - 合并多数据目录")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ========== 1. 合并所有数据目录 ==========
    print("\n【步骤1】合并数据目录...")
    all_dfs = []
    for data_dir in DATA_DIRS:
        if os.path.exists(data_dir):
            print(f"  加载: {data_dir}")
            df = load_data_from_dir(data_dir)
            if len(df) > 0:
                print(f"    → {len(df)} 条数据")
                all_dfs.append(df)
        else:
            print(f"  ⚠️ 目录不存在: {data_dir}")
    
    if not all_dfs:
        print("❌ 没有找到任何数据！")
        return
    
    master_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  ✅ 总计: {len(master_df)} 条数据")
    
    # 按来源统计
    print("\n  数据来源统计:")
    for src, count in master_df['source_dir'].value_counts().items():
        print(f"    - {src}: {count} 条")
    
    # ========== 2. 可视化原始分布 ==========
    print("\n【步骤2】分析原始数据分布...")
    plot_distribution(master_df, 'Original Steering Distribution (Before Balancing)', 'red')
    
    # ========== 3. 数据平衡 ==========
    print("\n【步骤3】数据平衡...")
    balanced_df = balance_data(master_df, STEER_THRESHOLD, STRAIGHT_SAMPLE_RATIO)
    print(f"\n  ✅ 平衡后总数据: {len(balanced_df)} 条")
    
    # ========== 4. 可视化平衡后分布 ==========
    print("\n【步骤4】分析平衡后数据分布...")
    plot_distribution(balanced_df, 'Balanced Steering Distribution (After Filtering)', 'blue')
    
    # ========== 5. 划分训练/验证/测试集 ==========
    print("\n【步骤5】划分训练集/验证集/测试集...")
    print(f"  比例: 训练={TRAIN_RATIO:.0%}, 验证={VALIDATION_RATIO:.0%}, 测试={TEST_RATIO:.0%}")
    
    # 先分出测试集
    train_val_df, test_df = train_test_split(
        balanced_df, 
        test_size=TEST_RATIO, 
        random_state=42
    )
    
    # 再从剩余数据中分出验证集
    # 验证集占剩余数据的比例 = VALIDATION_RATIO / (TRAIN_RATIO + VALIDATION_RATIO)
    val_ratio_adjusted = VALIDATION_RATIO / (TRAIN_RATIO + VALIDATION_RATIO)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_ratio_adjusted, 
        random_state=42
    )
    
    # 保存到输出目录
    train_path = os.path.join(OUTPUT_DIR, "train_labels.csv")
    val_path = os.path.join(OUTPUT_DIR, "validation_labels.csv")
    test_path = os.path.join(OUTPUT_DIR, "test_labels.csv")
    master_path = os.path.join(OUTPUT_DIR, "master_labels.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    balanced_df.to_csv(master_path, index=False)
    
    # ========== 6. 输出总结 ==========
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"原始数据总量: {len(master_df)}")
    print(f"平衡后数据量: {len(balanced_df)}")
    print(f"训练集大小: {len(train_df)} ({100*len(train_df)/len(balanced_df):.0f}%)")
    print(f"验证集大小: {len(val_df)} ({100*len(val_df)/len(balanced_df):.0f}%)")
    print(f"测试集大小: {len(test_df)} ({100*len(test_df)/len(balanced_df):.0f}%)")
    print(f"\n输出文件:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")
    print(f"  - {master_path}")
    
    # 提示下一步
    print("\n" + "=" * 60)
    print("下一步: 运行 train.py 训练模型")
    print("注意: 测试集只在最终评估时使用一次！")
    print("=" * 60)


if __name__ == '__main__':
    main()
