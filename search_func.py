import json

def search_by_bm25(aos_client, index_name, query, topk=4):    
    request_body = {
      "size": topk,
      "query": {
        "match": {
            "content" : query
        }
      }
    }

    response = aos_client.transport.perform_request(
        method="GET",
        url=f"/{index_name}/_search",
        body=json.dumps(request_body)
    )

    # docs = [hit["_source"]['content'] for hit in response["hits"]["hits"]]
    return response["hits"]["hits"]

def search_by_dense(aos_client, index_name, query, dense_model_id, topk=4):
    request_body = {
        "query": {
            "neural": {
                "dense_embedding": {
                  "query_text": query,
                  "model_id": dense_model_id,
                  "k": topk
                }
            }
        }
    }

    response = aos_client.transport.perform_request(
        method="GET",
        url=f"/{index_name}/_search",
        body=json.dumps(request_body)
    )

    # docs = [hit["_source"]['content'] for hit in response["hits"]["hits"]]
    return response["hits"]["hits"]

def search_by_sparse(aos_client, index_name, query, sparse_model_id, topk=4):
    request_body = {
      "size": topk,
      "query": {
          "neural_sparse": {
              "sparse_embedding": {
                "query_text": query,
                "model_id": sparse_model_id,
                "max_token_score": 3.5
              }
          }
      }
    }

    response = aos_client.transport.perform_request(
        method="GET",
        url=f"/{index_name}/_search",
        body=json.dumps(request_body)
    )

    # docs = [hit["_source"]['content'] for hit in response["hits"]["hits"]]
    return response["hits"]["hits"]

def search_by_dense_sparse(aos_client, index_name, query, sparse_model_id, dense_model_id, topk=4):
    request_body = {
      "size": topk,
      "query": {
        "hybrid": {
          "queries": [
            {
              "neural_sparse": {
                  "sparse_embedding": {
                    "query_text": query,
                    "model_id": sparse_model_id,
                    "max_token_score": 3.5
                  }
              }
            },
            {
              "neural": {
                  "dense_embedding": {
                      "query_text": query,
                      "model_id": dense_model_id,
                      "k": 10
                    }
                }
            }
          ]
        }
      }
    }

    response = aos_client.transport.perform_request(
        method="GET",
        url=f"/{index_name}/_search?search_pipeline=hybird-search-pipeline",
        body=json.dumps(request_body)
    )

    # docs = [hit["_source"]['content'] for hit in response["hits"]["hits"]]
    return response["hits"]["hits"]

def search_by_dense_bm25(aos_client, index_name, query, dense_model_id, topk=4):
    request_body = {
      "size": topk,
      "query": {
        "hybrid": {
          "queries": [
            {
              "match": {
                "text": {
                  "query": query
                }
              }
            },
            {
              "neural": {
                  "dense_embedding": {
                      "query_text": query,
                      "model_id": dense_model_id,
                      "k": 10
                    }
                }
            }
          ]
        }
      }
    }

    response = aos_client.transport.perform_request(
        method="GET",
        url=f"/{index_name}/_search?search_pipeline=hybird-search-pipeline",
        body=json.dumps(request_body)
    )

    # docs = [hit["_source"]['content'] for hit in response["hits"]["hits"]]
    return response["hits"]["hits"]