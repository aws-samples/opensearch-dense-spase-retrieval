import time
import argparse
from tqdm import tqdm
from datasets import load_dataset
from setup_model_and_pipeline import get_aos_client
from search_func import search_by_bm25, search_by_dense, search_by_sparse, search_by_dense_sparse, search_by_dense_bm25
from datasets import load_dataset
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

data_root_dir = "beir_data"

def ingest_dataset(corpus, aos_client,index_name, bulk_size=50):
    i=0
    bulk_body=[]
    for _id , body in tqdm(corpus.items()):
        text=body["title"]+" "+body["text"]
        bulk_body.append({ "index" : { "_index" : index_name, "_id" : _id } })
        bulk_body.append({ "content" : text })
        i+=1
        if i % bulk_size==0:
            response=aos_client.bulk(bulk_body,request_timeout=100)
            try:
                assert response["errors"]==False
            except:
                print("there is errors")
                print(response)
                time.sleep(1)
                response = aos_client.bulk(bulk_body,request_timeout=100)
            bulk_body=[]
        
    response=aos_client.bulk(bulk_body,request_timeout=100)
    assert response["errors"]==False
    aos_client.indices.refresh(index=index_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aos_endpoint', type=str, default='', help='aos endpoint')
    parser.add_argument('--testset_size', type=int, default=1000, help='testset size')
    parser.add_argument('--index_name', type=str, default='', help='index name')
    parser.add_argument('--is_ingest', type=bool, default=False, help='ingest or search')
    parser.add_argument('--topk', type=int, default=4, help='top k')
    parser.add_argument('--dense_model_id', type=str, default='', help='dense_model_id')
    parser.add_argument('--sparse_model_id', type=str, default='', help='sparse_model_id')
    parser.add_argument("--ingest", action="store_true", help="is ingest or search")
    parser.add_argument("--dataset_name", type=str, default='fiqa', help='specify the dataset')
    args = parser.parse_args()
    aos_endpoint = args.aos_endpoint
    aos_domain = '-'.join(aos_endpoint.split('-')[1:3])
    testset_size = args.testset_size
    index_name = args.index_name
    ingest = args.ingest
    topk = args.topk
    dense_model_id = args.dense_model_id
    sparse_model_id = args.sparse_model_id
    dataset_name = args.dataset_name

    aos_client = get_aos_client(aos_endpoint)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, data_root_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    if ingest is True:
        start = time.time()
        ingest_dataset(corpus, aos_client=aos_client, index_name=index_name)
        elpase_time = time.time() - start
        throughput = float(testset_size)/elpase_time
        print(f"[ingest] throughput/s:{throughput}")
    else:
        
        run_res={}
        for _id, query in tqdm(queries.items()):
            hits = search_by_bm25(aos_client, index_name, query, topk)
            run_res[_id]={item["_id"]:item["_score"] for item in hits}
            
        for query_id, doc_dict in tqdm(run_res.items()):
            if query_id in doc_dict:
                doc_dict.pop(query_id)
        res = EvaluateRetrieval.evaluate(qrels, run_res, [1, 4,10])
        print("search_by_bm25:")
        print(res)
        
        run_res={}
        for _id, query in tqdm(queries.items()):
            hits = search_by_dense(aos_client, index_name, query, dense_model_id, topk)
            run_res[_id]={item["_id"]:item["_score"] for item in hits}
            
        for query_id, doc_dict in tqdm(run_res.items()):
            if query_id in doc_dict:
                doc_dict.pop(query_id)
        res = EvaluateRetrieval.evaluate(qrels, run_res, [1, 4,10])
        print("search_by_dense:")
        print(res)
        
        run_res={}
        for _id, query in tqdm(queries.items()):
            hits = search_by_sparse(aos_client, index_name, query, sparse_model_id, topk)
            run_res[_id]={item["_id"]:item["_score"] for item in hits}
            
        for query_id, doc_dict in tqdm(run_res.items()):
            if query_id in doc_dict:
                doc_dict.pop(query_id)
        res = EvaluateRetrieval.evaluate(qrels, run_res, [1, 4,10])
        print("search_by_sparse:")
        print(res)
        
        run_res={}
        for _id, query in tqdm(queries.items()):
            hits = search_by_dense_sparse(aos_client, index_name, query, sparse_model_id, dense_model_id, topk)
            run_res[_id]={item["_id"]:item["_score"] for item in hits}
            
        for query_id, doc_dict in tqdm(run_res.items()):
            if query_id in doc_dict:
                doc_dict.pop(query_id)
        res = EvaluateRetrieval.evaluate(qrels, run_res, [1, 4,10])
        print("search_by_dense_sparse:")
        print(res)
        
        run_res={}
        for _id, query in tqdm(queries.items()):
            hits = search_by_dense_bm25(aos_client, index_name, query, dense_model_id, topk)
            run_res[_id]={item["_id"]:item["_score"] for item in hits}
            
        for query_id, doc_dict in tqdm(run_res.items()):
            if query_id in doc_dict:
                doc_dict.pop(query_id)
        res = EvaluateRetrieval.evaluate(qrels, run_res, [1, 4,10])
        print("search_by_dense_bm25:")
        print(res)