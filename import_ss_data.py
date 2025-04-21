from elasticsearch import Elasticsearch, helpers
import gzip
from tqdm import tqdm
import json
import os


es = Elasticsearch(
    "http://172.29.174.49:9200",
    verify_certs=False  # 如果是自签名证书
)


# new version of import
def bulk_data(es, fuc):
    for success, info in helpers.parallel_bulk(es, fuc, chunk_size=500):
        if not success:
            print('A document failed:', info)
            raise helpers.BulkIndexError


def gen_data(file, id_key, index_name):
    for i in file:
        item = json.loads(i)
        # if "Sociology" in item["fieldsOfStudy"] or "DBLP" in item["sources"] or "Medline" in item["sources"]:
        data = {}
        for k in item:
            data["_index"] = index_name
            if k == id_key:
                data["_id"] = item[id_key]
            elif item[k] and k in {'fieldsOfStudy', 'authors', 'year', 'inCitations', 'outCitations', 'sources'}:
                data[k] = item[k]
        yield data


file_dir = "D:\\Dataset\\new_data\\data"
author_dic = {}
index = "semantic_scholar"

if not es.indices.exists(index=index):
    es.indices.create(index=index, settings={"number_of_replicas": 0, "max_result_window": 2000000000})

for f in tqdm(os.listdir(file_dir)):
    file = os.path.join(file_dir, f)
    with gzip.open(file, 'r') as pf:
        bulk_data(es, gen_data(pf, 'id', index))
