import time
import argparse
from tqdm import tqdm
from datasets import load_dataset
from setup_model_and_pipeline import get_aos_client
from search_func import search_by_bm25, search_by_dense, search_by_sparse, search_by_dense_sparse, search_by_dense_bm25
from datasets import load_dataset

def deduplicate_dataset(dataset):
    context_list = [row["context"] for row in dataset]
    context_set = set(context_list)
    return list(context_set)

def build_bulk_body(index_name,sources_list):
    bulk_body = []
    for source in sources_list:
        bulk_body.append({ "index" : { "_index" : index_name} })
        bulk_body.append(source)
    return bulk_body

def ingest_dataset(dataset,aos_client,index_name, bulk_size=50):
    print("Deduplicating dataset...")
    context_list = deduplicate_dataset(dataset)
    # 19029 for train, 1204 for validation
    print(f"Finished deduplication. Total number of passages: {len(context_list)}")

    for start_idx in tqdm(range(0,len(context_list),bulk_size)):
        contexts = context_list[start_idx:start_idx+bulk_size]
        response = aos_client.bulk(
            build_bulk_body(index_name, [{"content":context} for context in contexts]),
            # set a large timeout because a new sparse encoding endpoint need warm up
            request_timeout=100
        )
        assert response["errors"]==False

    aos_client.indices.refresh(index=index_name,request_timeout=100)

def calc_recall(metric, answer, results):
    if answer in results[:1]:
        metric['hit_1'] += 1
    else:
        metric['miss_1'] += 1

    if answer in results[:4]:
        metric['hit_4'] += 1
    else:
        metric['miss_4'] += 1

    if answer in results[:10]:
        metric['hit_10'] += 1
    else:
        metric['miss_10'] += 1

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
    parser.add_argument("--query_dataset_type", type=str, default='validation', help='use validation set or train set to query')
    parser.add_argument("--dataset_name", type=str, default='squad_v2', help='specify the dataset')
    args = parser.parse_args()
    aos_endpoint = args.aos_endpoint
    aos_domain = '-'.join(aos_endpoint.split('-')[1:3])
    testset_size = args.testset_size
    index_name = args.index_name
    ingest = args.ingest
    topk = args.topk
    dense_model_id = args.dense_model_id
    sparse_model_id = args.sparse_model_id
    query_dataset_type = args.query_dataset_type

    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name)

    aos_client = get_aos_client(aos_endpoint)

    if ingest is True:
        start = time.time()
        ingest_dataset(dataset=dataset["train"],aos_client=aos_client,index_name=index_name)
        ingest_dataset(dataset=dataset["validation"],aos_client=aos_client,index_name=index_name)
        elpase_time = time.time() - start
        throughput = float(testset_size)/elpase_time
        print(f"[ingest] throughput/s:{throughput}")
    else:
        dataset = dataset[query_dataset_type]
        print("start search by bm25")
        metric = {
            "hit_1" : 0,
            "miss_1" : 0,
            "hit_4" : 0,
            "miss_4" : 0,
            "hit_10" : 0,
            "miss_10" : 0
        }
        for idx, item in tqdm(enumerate(dataset.select(range(testset_size)))):
            query = item['question']
            content = item['context']
            response = search_by_bm25(aos_client, index_name, query, topk)
            results = [hit["_source"]['content'] for hit in response ]
            calc_recall(metric, content, results)

        print(f"hit_1:{metric['hit_1']}, miss_1:{metric['miss_1']}, recall@1:{metric['hit_1']/(metric['hit_1']+metric['miss_1'])}")
        print(f"hit_4:{metric['hit_4']}, miss_1:{metric['miss_4']}, recall@4:{metric['hit_4']/(metric['hit_4']+metric['miss_4'])}")
        print(f"hit_10:{metric['hit_10']}, miss_10:{metric['miss_10']}, recall@10:{metric['hit_10']/(metric['hit_10']+metric['miss_10'])}")

        print("start search by dense")
        metric = {
            "hit_1" : 0,
            "miss_1" : 0,
            "hit_4" : 0,
            "miss_4" : 0,
            "hit_10" : 0,
            "miss_10" : 0
        }
        for idx, item in tqdm(enumerate(dataset.select(range(testset_size)))):
            query = item['question']
            content = item['context']
            response = search_by_dense(aos_client, index_name, query, dense_model_id, topk)
            results = [hit["_source"]['content'] for hit in response ]
            calc_recall(metric, content, results)

        print(f"hit_1:{metric['hit_1']}, miss_1:{metric['miss_1']}, recall@1:{metric['hit_1']/(metric['hit_1']+metric['miss_1'])}")
        print(f"hit_4:{metric['hit_4']}, miss_1:{metric['miss_4']}, recall@4:{metric['hit_4']/(metric['hit_4']+metric['miss_4'])}")
        print(f"hit_10:{metric['hit_10']}, miss_10:{metric['miss_10']}, recall@10:{metric['hit_10']/(metric['hit_10']+metric['miss_10'])}")

        print("start search by sparse")
        metric = {
            "hit_1" : 0,
            "miss_1" : 0,
            "hit_4" : 0,
            "miss_4" : 0,
            "hit_10" : 0,
            "miss_10" : 0
        }
        for idx, item in tqdm(enumerate(dataset.select(range(testset_size)))):
            query = item['question']
            content = item['context']
            response = search_by_sparse(aos_client, index_name, query, sparse_model_id, topk)
            results = [hit["_source"]['content'] for hit in response ]
            calc_recall(metric, content, results)

        print(f"hit_1:{metric['hit_1']}, miss_1:{metric['miss_1']}, recall@1:{metric['hit_1']/(metric['hit_1']+metric['miss_1'])}")
        print(f"hit_4:{metric['hit_4']}, miss_1:{metric['miss_4']}, recall@4:{metric['hit_4']/(metric['hit_4']+metric['miss_4'])}")
        print(f"hit_10:{metric['hit_10']}, miss_10:{metric['miss_10']}, recall@10:{metric['hit_10']/(metric['hit_10']+metric['miss_10'])}")

        print("start search by dense-sparse")
        metric = {
            "hit_1" : 0,
            "miss_1" : 0,
            "hit_4" : 0,
            "miss_4" : 0,
            "hit_10" : 0,
            "miss_10" : 0
        }
        for idx, item in tqdm(enumerate(dataset.select(range(testset_size)))):
            query = item['question']
            content = item['context']
            response = search_by_dense_sparse(aos_client, index_name, query, sparse_model_id, dense_model_id, topk)
            results = [hit["_source"]['content'] for hit in response ]
            calc_recall(metric, content, results)

        print(f"hit_1:{metric['hit_1']}, miss_1:{metric['miss_1']}, recall@1:{metric['hit_1']/(metric['hit_1']+metric['miss_1'])}")
        print(f"hit_4:{metric['hit_4']}, miss_1:{metric['miss_4']}, recall@4:{metric['hit_4']/(metric['hit_4']+metric['miss_4'])}")
        print(f"hit_10:{metric['hit_10']}, miss_10:{metric['miss_10']}, recall@10:{metric['hit_10']/(metric['hit_10']+metric['miss_10'])}")

        print("start search by dense-bm25")
        metric = {
            "hit_1" : 0,
            "miss_1" : 0,
            "hit_4" : 0,
            "miss_4" : 0,
            "hit_10" : 0,
            "miss_10" : 0
        }
        for idx, item in tqdm(enumerate(dataset.select(range(testset_size)))):
            query = item['question']
            content = item['context']
            response = search_by_dense_bm25(aos_client, index_name, query, dense_model_id, topk)
            results = [hit["_source"]['content'] for hit in response ]
            calc_recall(metric, content, results)

        print(f"hit_1:{metric['hit_1']}, miss_1:{metric['miss_1']}, recall@1:{metric['hit_1']/(metric['hit_1']+metric['miss_1'])}")
        print(f"hit_4:{metric['hit_4']}, miss_1:{metric['miss_4']}, recall@4:{metric['hit_4']/(metric['hit_4']+metric['miss_4'])}")
        print(f"hit_10:{metric['hit_10']}, miss_10:{metric['miss_10']}, recall@10:{metric['hit_10']/(metric['hit_10']+metric['miss_10'])}")
