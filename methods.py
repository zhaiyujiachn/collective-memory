from elasticsearch import Elasticsearch
import elasticsearch
from elasticsearch_dsl import Search
from tqdm import tqdm


class SearchData:

    def __init__(self, index_name):

        ELASTIC_PASSWORD = 'yaSHUANG1987'
        self.es = Elasticsearch(
        ["http://172.29.174.49:9200"],
        verify_certs=False,
        request_timeout=300,
        retry_on_timeout=True,
        http_compress=True,
        connections_per_node=25
    )
        self.index = index_name
        

    def get_paper(self, paper_id):
        if not self.es.exists_source(index=self.index, id=paper_id):
            return None
        d = self.es.get(index=self.index, id=paper_id)
        return d

    def get_paper_reference(self, paper_id):
        d = self.es.get(index=self.index, id=paper_id)
        return d["_source"]["outCitations"]

    def get_paper_citation(self, paper_id):
        d = self.es.get(index=self.index, id=paper_id)
        if "inCitations" not in d["_source"].keys():
            return []
        return d["_source"]["inCitations"]

    def get_paper_date(self, paper_id):
        d = self.es.get(index=self.index, id=paper_id)
        if "year" not in d["_source"].keys():
            return ''
        year = str(d["_source"]["year"]) + "-01-01"
        return year

    def get_paper_venue(self, paper_id):
        d = self.es.get(index=self.index, id=paper_id)
        venue = str(d["_source"]["venue"])
        return venue
    
    def get_paper_author(self, paper_id):
        d = self.es.get(index=self.index, id=paper_id)
        return [i['ids'][0] for i in d["_source"]["authors"] if i['ids']]

    def is_paper_has_author(self, paper_id):
        try:
            d = self.es.get(index=self.index, id=paper_id)
        except elasticsearch.NotFoundError:
            return False
        if "authors" in d["_source"]:
            if len(d["_source"]["authors"]) > 0:
                return True
        return False

    def get_author_paper(self, dataset, author_id):
        if dataset in ["DBLP", "Medline"]:
            s = Search(using=self.es, index=self.index).query("match", sources=dataset).query("match", authors__ids=author_id)
        elif dataset == "Sociology":
            s = Search(using=self.es, index=self.index).query("match", fieldsOfStudy=dataset).query("match", authors__ids=author_id)
        s = s[0:999999]
        res = s.execute()
        d = [i['_id'] for i in res.to_dict()['hits']['hits']]
        return d

    def get_all_paper(self, dataset):
        if dataset in ["DBLP", "Medline"]:
            s = Search(using=self.es, index=self.index).query("match", sources=dataset)
        elif dataset == "Sociology":
            s = Search(using=self.es, index=self.index).query("match", fieldsOfStudy=dataset)
        with tqdm(total=s.count()) as pbar:
            for hit in s.scan():
                pbar.update(1)
                yield hit.meta.id

    def get_latest_paper(self, dataset):
        if dataset in ["DBLP", "Medline"]:
            s = Search(using=self.es, index=self.index).query("match", sources=dataset).query("exists",
                                                                                              field="year").sort(
                {"year": {"order": "asc"}})
        elif dataset == "Sociology":
            s = Search(using=self.es, index=self.index).query("match", fieldsOfStudy=dataset).query("exists",
                                                                                                    field="year").sort(
                {"year": {"order": "asc"}})
        latest_paper = s.execute()[0].year
        return str(latest_paper) + "-01-01"

    def get_earliest_paper(self, dataset):
        if dataset in ["DBLP", "Medline"]:
            s = Search(using=self.es, index=self.index).query("match", sources=dataset).query("exists",
                                                                                              field="year").sort(
                {"year": {"order": "desc"}})
        elif dataset == "Sociology":
            s = Search(using=self.es, index=self.index).query("match", fieldsOfStudy=dataset).query("exists",
                                                                                                    field="year").sort(
                {"year": {"order": "desc"}})

        earliest_paper = s.execute()[0].year
        return str(earliest_paper) + "-01-01"


if __name__ == '__main__':
    sd = SearchData("semanticscholar")
    print("数据中发表最早的论文时间是" + sd.get_latest_paper("Medline"))
    print("数据中发表最晚的论文时间是" + sd.get_earliest_paper("Medline"))
    print(sd.get_paper_author("7fa11121cb5a40d0f9dd077887db594d24bb74f8"))
