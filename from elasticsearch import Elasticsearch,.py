from elasticsearch import Elasticsearch, helpers
import gzip
import json
import os
from tqdm import tqdm
from multiprocessing import Pool, freeze_support

# 将处理函数移到模块顶层
def process_file(args):
    file, file_dir, index = args
    try:
        fast_bulk_import(os.path.join(file_dir, file), index)
        return True
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return False

def fast_bulk_import(file_path, index_name):
    es = get_es_client()  # 使用单独的函数获取客户端
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        batch = []
        for line in f:
            item = json.loads(line)
            doc = {
                "_op_type": "index",
                "_index": index_name,
                "_id": item.get("id"),
                "_source": {k: v for k, v in item.items() 
                          if k in {'fieldsOfStudy', 'authors', 'year', 'inCitations', 'outCitations', 'sources'}}
            }
            batch.append(doc)
            if len(batch) >= 5000:
                helpers.bulk(es, batch, request_timeout=120)
                batch = []
        if batch:
            helpers.bulk(es, batch, request_timeout=120)

# 单独的函数获取ES客户端
def get_es_client():
    return Elasticsearch(
        ["http://172.29.174.49:9200"],
        verify_certs=False,
        request_timeout=300,
        retry_on_timeout=True,
        http_compress=True,
        connections_per_node=25
    )

def main():
    file_dir = "D:\\Dataset\\new_data\\data"
    index = "semantic_scholar"
    
    es = get_es_client()
    if not es.indices.exists(index=index):
        es.indices.create(
            index=index,
            settings={
                "number_of_shards": 4,
                "number_of_replicas": 0,
                "refresh_interval": "-1",
                "index.translog.durability": "async"
            }
        )

    files = os.listdir(file_dir)
    
    # 准备参数
    params = [(file, file_dir, index) for file in files]
    
    # 使用多进程
    with Pool(processes=min(4, os.cpu_count())) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, params), total=len(files)))
    
    # 恢复设置
    es.indices.put_settings(
        index=index,
        body={
            "number_of_replicas": 1,
            "refresh_interval": "1s"
        }
    )

if __name__ == '__main__':
    freeze_support()
    main()