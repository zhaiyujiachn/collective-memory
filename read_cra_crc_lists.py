import json
import os

def read_json_file(file_path):
    """读取JSON文件并返回其内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None

def main():
    # 定义文件路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # APS目录下的文件
    aps_cra_path = os.path.join(base_dir, 'final_code_data', 'APS', 'cra_list.json')
    aps_crc_path = os.path.join(base_dir, 'final_code_data', 'APS', 'crc_list.json')
    
    # Medline目录下的文件
    medline_cra_path = os.path.join(base_dir, 'final_code_data', 'Medline', 'cra_list.json')
    medline_crc_path = os.path.join(base_dir, 'final_code_data', 'Medline', 'crc_list.json')
    
    # 读取APS目录下的文件
    print("\n读取APS目录下的文件:")
    aps_cra_data = read_json_file(aps_cra_path)
    if aps_cra_data:
        print(f"APS cra_list.json 包含 {len(aps_cra_data)} 条记录")
        # 显示前5条记录作为示例
        print("示例数据 (前5条):")
        for i, item in enumerate(aps_cra_data[:5]):
            print(f"  {i+1}. {item}")
    
    aps_crc_data = read_json_file(aps_crc_path)
    if aps_crc_data:
        print(f"\nAPS crc_list.json 包含 {len(aps_crc_data)} 条记录")
        # 显示前5条记录作为示例
        print("示例数据 (前5条):")
        for i, item in enumerate(aps_crc_data[:5]):
            print(f"  {i+1}. {item}")
    
    # 读取Medline目录下的文件
    print("\n读取Medline目录下的文件:")
    medline_cra_data = read_json_file(medline_cra_path)
    if medline_cra_data:
        print(f"Medline cra_list.json 包含 {len(medline_cra_data)} 条记录")
        # 显示前5条记录作为示例
        print("示例数据 (前5条):")
        for i, item in enumerate(medline_cra_data[:5]):
            print(f"  {i+1}. {item}")
    
    medline_crc_data = read_json_file(medline_crc_path)
    if medline_crc_data:
        print(f"\nMedline crc_list.json 包含 {len(medline_crc_data)} 条记录")
        # 显示前5条记录作为示例
        print("示例数据 (前5条):")
        for i, item in enumerate(medline_crc_data[:5]):
            print(f"  {i+1}. {item}")

if __name__ == "__main__":
    main()