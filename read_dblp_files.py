import os
import json
import pandas as pd
import glob

def read_file(file_path):
    """读取文件并返回其内容和格式"""
    file_extension = os.path.splitext(file_path)[1].lower()
    file_size = os.path.getsize(file_path)
    
    # 检查文件大小，如果超过1GB，只读取部分内容
    max_size = 1 * 1024 * 1024 * 1024  # 1GB
    
    try:
        if file_size > max_size:
            print(f"警告: 文件大小为 {file_size / (1024*1024):.2f} MB，超过1GB限制，将只读取部分内容")
            
        if file_extension == '.json':
            if file_size > max_size:
                # 对于大型JSON文件，只读取前10000行
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= 10000:
                            break
                        lines.append(line)
                    data_str = ''.join(lines)
                    try:
                        # 尝试解析可能不完整的JSON
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        # 如果解析失败，则返回前10行作为文本
                        data = lines[:10]
                        file_format = 'JSON (部分内容，未完整解析)'
                        return data, file_format
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            file_format = 'JSON'
        elif file_extension == '.csv':
            if file_size > max_size:
                # 对于大型CSV文件，只读取前1000行
                data = pd.read_csv(file_path, nrows=1000)
            else:
                data = pd.read_csv(file_path)
            file_format = 'CSV'
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_size > max_size:
                    data = [f.readline() for _ in range(1000)]  # 只读取前1000行
                else:
                    data = f.readlines()
            file_format = 'TXT'
        elif file_extension == '.xml':
            # 简单读取XML文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_size > max_size:
                    data = [f.readline() for _ in range(1000)]  # 只读取前1000行
                else:
                    data = f.readlines()
            file_format = 'XML'
        else:
            # 尝试以二进制方式读取
            with open(file_path, 'rb') as f:
                data = f.read(1000)  # 只读取前1000字节用于分析
            file_format = '未知格式 (二进制)'
            
        return data, file_format
    except Exception as e:
        return f"读取文件时出错: {str(e)}", "错误"

def display_data_preview(data, file_format):
    """显示数据的前10行或前10个元素"""
    if file_format == 'JSON':
        if isinstance(data, dict):
            # 如果是字典，显示前10个键值对
            preview = dict(list(data.items())[:10])
            structure = f"字典结构，共有 {len(data)} 个键值对"
        elif isinstance(data, list):
            # 如果是列表，显示前10个元素
            preview = data[:10]
            structure = f"列表结构，共有 {len(data)} 个元素"
        else:
            preview = str(data)[:500] + "..." if len(str(data)) > 500 else str(data)
            structure = f"其他JSON结构: {type(data)}"
    elif file_format == 'CSV':
        preview = data.head(10)
        structure = f"DataFrame结构，形状: {data.shape}, 列: {list(data.columns)}"
    elif file_format in ['TXT', 'XML']:
        preview = data[:10]
        structure = f"文本行列表，共有 {len(data)} 行"
    elif '二进制' in file_format:
        preview = str(data)
        structure = "二进制数据"
    else:
        preview = str(data)
        structure = "未知结构"
        
    return preview, structure

def get_file_size_info(file_path):
    """获取文件大小信息"""
    size_bytes = os.path.getsize(file_path)
    
    if size_bytes < 1024:
        return f"{size_bytes} 字节"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def main():
    # 直接指向正确的DBLP文件夹路径
    dblp_folder = os.path.join('src', 'collective-memory', 'data', 'DBLP')
    
    if not os.path.exists(dblp_folder):
        print(f"错误: 找不到DBLP文件夹 '{dblp_folder}'")
        # 尝试在其他位置查找
        possible_locations = [
            'data',
            'data/processed',
            'src/collective-memory/data',
            '.'
        ]
        
        for location in possible_locations:
            pattern = os.path.join(location, "**", "DBLP", "*")
            files = glob.glob(pattern, recursive=True)
            if files:
                dblp_folder = os.path.dirname(files[0])
                print(f"找到DBLP文件夹: {dblp_folder}")
                break
        else:
            # 如果没有找到DBLP文件夹，尝试直接在data目录下查找文件
            print("尝试在data目录下直接查找文件...")
            dblp_folder = 'data'
    
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(dblp_folder) if os.path.isfile(os.path.join(dblp_folder, f))]
    
    if not files:
        print(f"在 {dblp_folder} 中没有找到文件")
        return
    
    print(f"在 {dblp_folder} 中找到 {len(files)} 个文件")
    
    # 如果超过3个文件，只处理前3个
    if len(files) > 3:
        files = files[:3]
        print(f"只处理前3个文件: {files}")
    
    # 处理每个文件
    for file_name in files:
        file_path = os.path.join(dblp_folder, file_name)
        print("\n" + "=" * 80)
        print(f"文件: {file_name}")
        print("=" * 80)
        
        data, file_format = read_file(file_path)
        print(f"文件格式: {file_format}")
        
        preview, structure = display_data_preview(data, file_format)
        print(f"数据结构: {structure}")
        print("\n前10行/元素预览:")
        print(preview)

if __name__ == "__main__":
    main()