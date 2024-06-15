import re
import csv

# 定义输入和输出文件路径
input_file_path = "result/loss_feature.txt"
output_file_path = "result/output.csv"

# 打开输入文件读取数据
with open(input_file_path, "r") as infile:
    lines = infile.readlines()

# 准备要写入CSV的数据
data_to_write = []

# 解析每一行
for line in lines:
    # 使用正则表达式匹配和提取数据
    match = re.match(
        r"(GCN) (\w+) (\w+) feature_removed (\w+) run_seed_feature (\d+) remove_feature_ratio ([0-9.]+) loss0: ([0-9.]+)",
        line,
    )
    if match:
        (
            method,
            partition_method,
            dataset,
            feature_removed,
            run_seed_feature,
            remove_feature_ratio,
            loss0,
        ) = match.groups()
        data_to_write.append(
            [
                method,
                partition_method,
                dataset,
                feature_removed,
                run_seed_feature,
                remove_feature_ratio,
                loss0,
            ]
        )

# 写入CSV文件
with open(output_file_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # 写入标题行
    writer.writerow(
        [
            "Method",
            "Partition Method",
            "Dataset",
            "Feature Removed",
            "Run Seed Feature",
            "Remove Feature Ratio",
            "Loss0",
        ]
    )
    # 写入数据
    writer.writerows(data_to_write)

print(f"Data has been successfully written to {output_file_path}")
