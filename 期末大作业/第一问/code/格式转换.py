import csv
import chardet
import sys


def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as file:
        raw_data = file.read(10000)  # 读取前10000字节来检测编码
        result = chardet.detect(raw_data)
        return result['encoding']


def convert_txt_to_csv(input_file, output_file, swap_columns=False):
    # 首先检测输入文件的编码
    detected_encoding = detect_encoding(input_file)
    print(f"检测到的文件编码: {detected_encoding}")

    # 尝试多种编码方式
    encodings_to_try = [detected_encoding, 'utf-8', 'utf-8-sig', 'gbk', 'cp1252', 'iso-8859-1']

    lines = None
    used_encoding = None

    # 尝试不同的编码读取文件
    for encoding in encodings_to_try:
        if encoding is None:
            continue
        try:
            with open(input_file, 'r', encoding=encoding, errors='ignore') as file:
                lines = file.readlines()
                used_encoding = encoding
                print(f"成功使用编码读取文件: {encoding}")
                break
        except (UnicodeDecodeError, LookupError) as e:
            print(f"编码 {encoding} 失败: {e}")
            continue

    if lines is None:
        print("无法读取文件，尝试所有编码都失败了")
        return

    try:
        # 获取原始表头并按tab分割
        headers = lines[0].strip().split('\t')
        print(f"检测到的列: {headers}")

        # 使用UTF-8编码写入CSV文件，并添加BOM以确保Excel等软件正确显示
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            # 如果需要交换列顺序
            if swap_columns and len(headers) >= 5:
                # 交换表头中 username 和 image_id 的位置（索引3和4）
                headers[3], headers[4] = headers[4], headers[3]

            writer.writerow(headers)  # 写入分割后的表头

            processed_count = 0
            error_count = 0

            # 从第二行开始处理数据
            for i, line in enumerate(lines[1:], 2):  # 从第2行开始计数
                try:
                    # 清理行数据
                    clean_line = line.strip()
                    if not clean_line:
                        continue

                    # 按tab分割主要字段
                    main_parts = clean_line.split('\t')

                    if len(main_parts) >= len(headers):
                        # 只取前面对应表头数量的列
                        row_data = main_parts[:len(headers)]

                        if swap_columns and len(row_data) >= 5:
                            # 交换数据中 username 和 image_id 的位置
                            row_data[3], row_data[4] = row_data[4], row_data[3]

                        # 清理每个字段中可能的问题字符
                        cleaned_row = []
                        for field in row_data:
                            # 移除可能导致CSV解析问题的字符
                            cleaned_field = field.replace('\n', ' ').replace('\r', ' ')
                            cleaned_row.append(cleaned_field)

                        writer.writerow(cleaned_row)
                        processed_count += 1
                    else:
                        print(
                            f"警告: 第{i}行字段数量不匹配 (期望{len(headers)}，实际{len(main_parts)}): {clean_line[:100]}...")
                        error_count += 1

                except Exception as e:
                    print(f"处理第{i}行时出错: {str(e)}")
                    print(f"问题行内容: {line.strip()[:100]}...")
                    error_count += 1
                    continue

            print(f"转换完成!")
            print(f"- 成功处理: {processed_count} 行")
            print(f"- 错误行数: {error_count} 行")
            print(f"- 输入文件编码: {used_encoding}")
            print(f"- 输出文件: {output_file} (UTF-8编码)")

    except Exception as e:
        print(f"转换过程中发生错误: {e}")


def validate_csv_output(csv_file):
    """验证生成的CSV文件"""
    try:
        with open(csv_file, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            headers = next(reader)
            print(f"CSV文件验证 - 列头: {headers}")

            # 读取前几行数据进行验证
            for i, row in enumerate(reader):
                if i >= 3:  # 只检查前3行数据
                    break
                print(f"第{i + 1}行数据: {row[:3]}...")  # 只显示前3个字段

    except Exception as e:
        print(f"验证CSV文件时出错: {e}")


# 使用示例
if __name__ == "__main__":
    # 转换文件 - posts.csv 需要交换列顺序
    print("=== 转换训练集 ===")
    convert_txt_to_csv(
        r'D:\大三下\云计算与大数据分析\大作业\期末大作业\twitter_dataset\devset\posts.txt',
        r'D:\大三下\云计算与大数据分析\大作业\期末大作业\data\训练集.csv',
        swap_columns=True
    )

    # 验证输出
    validate_csv_output(r'D:\大三下\云计算与大数据分析\大作业\期末大作业\data\训练集.csv')

    print("\n=== 转换测试集 ===")
    # posts_groundtruth.csv 保持原有顺序
    convert_txt_to_csv(
        r'D:\大三下\云计算与大数据分析\大作业\期末大作业\twitter_dataset\testset\posts_groundtruth.txt',
        r'D:\大三下\云计算与大数据分析\大作业\期末大作业\data\测试集.csv',
        swap_columns=False
    )

    # 验证输出
    validate_csv_output(r'D:\大三下\云计算与大数据分析\大作业\期末大作业\data\测试集.csv')
