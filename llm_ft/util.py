import json
import random
import csv
import os
import pandas as pd


def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def dump_json(data, path):
    # 将Python对象写入JSON文件
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def dic2json(data):
    return json.dumps(data, indent=4, ensure_ascii=False)


def list_to_csv(data, filename):
    """
    将二维列表写入CSV文件。

    参数:
    data (list of list): 要写入CSV的二维列表。
    filename (str): 输出的CSV文件名。
    """
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            for row in data:
                csvwriter.writerow(row)
        print(f"数据已成功写入 {filename}")
    except Exception as e:
        print(f"写入CSV文件时出错: {e}")


def list_files_recursive(directory):
    """
    递归遍历目录下的所有文件路径。

    :param directory: 要遍历的目录路径
    :return: 文件路径列表
    """
    files_list = []
    
    # 遍历目录中的所有条目
    for entry in os.listdir(directory):
        # 构建完整路径
        full_path = os.path.join(directory, entry)
        
        # 如果是目录，则递归调用
        if os.path.isdir(full_path):
            files_list.extend(list_files_recursive(full_path))
        else:
            # 如果是文件，则添加到列表
            files_list.append(full_path)
    
    return files_list


def read_file(file_path):
    """
    读取文件内容并返回为字符串。

    :param file_path: 要读取的文件路径
    :return: 文件内容字符串
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except IOError:
        print(f"Error: An error occurred while reading the file {file_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def read_xlsx_with_pandas(file_path):
    """
    使用 pandas 读取 Excel 文件内容。

    :param file_path: 要读取的 Excel 文件路径
    :return: 包含每个工作表数据的字典，键为工作表名称，值为 DataFrame
    """
    try:
        excel_data = pd.read_excel(file_path, sheet_name=None)
        return excel_data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        

def read_ids_to_set(file_path):
    """
    读取一个txt文件中的ID，并将其存储在一个集合中。

    :param file_path: txt文件的路径
    :return: 包含所有ID的集合
    """
    ids_set = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除每行的空白字符（如换行符）并添加到集合中
                id = line.strip()
                if id:  # 确保不添加空行
                    ids_set.add(id)
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    
    return ids_set


def read_ids_from_file(file_path):
    """
    读取一个只有一列ID的文件，并将其存储在一个列表中。

    参数:
    file_path (str): 文件的路径。

    返回:
    list: 包含文件中所有ID的列表。
    """
    ids = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除每行的换行符和空格
                id = line.strip()
                if id:  # 确保不是空行
                    ids.append(id)
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except Exception as e:
        print(f"读取文件时出错: {e}")
    
    return ids



if __name__ == '__main__':
    lis = list_files_recursive('category_output')
    for item in lis:
        print (read_file(item))