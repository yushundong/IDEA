import csv
import pdb

# 文本文件路径
input_file_path = "result/loss_feature.txt"
# 输出CSV文件路径
output_file_path = "result/output_feature.csv"

# 打开文本文件和CSV文件
with open(input_file_path, "r") as txt_file, open(
    output_file_path, "w", newline=""
) as csv_file:
    csv_writer = csv.writer(csv_file)
    # 写入列头
    csv_writer.writerow(
        ["Seed", "Method", "Dataset", "Ratio", "Model", "Loss_w", "Loss_wo"]
    )

    # 逐行读取文本文件
    for line in txt_file:
        # 解析行以提取数据
        parts = line.split()
        id_and_feature = parts[0].split("_")
        seed = id_and_feature[0]
        method = id_and_feature[1]
        dataset = id_and_feature[2]
        ratio = id_and_feature[-1]
        model = id_and_feature[6]
        loss_w = float(parts[2].strip(","))
        loss_wo = float(parts[4].strip(","))
        # 写入数据到CSV
        csv_writer.writerow(
            [
                seed,
                method,
                dataset,
                ratio,
                model,
                loss_w,
                loss_wo,
            ]
        )


print("finished")
