import json
import boto3
import requests
from requests_aws4auth import AWS4Auth
import argparse
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers
# pip3 install boto3 requests requests_aws4auth argparse opensearch-py

def create_bedrock_caller_role(domain_name, account_id, region):
    role_policy = {
      "Version": "2012-10-17",
      "Statement": [
          {
              "Sid": "VisualEditor0",
              "Effect": "Allow",
              "Action": [
                  "bedrock:InvokeModel",
                  "bedrock:InvokeModelWithResponseStream",
                  "sagemaker:InvokeEndpointAsync",
                  "sagemaker:InvokeEndpoint"
              ],
              "Resource": "*"
          },
          {
              "Effect": "Allow",
              "Action": "iam:PassRole",
              "Resource": f"arn:aws:iam::{account_id}:role/OpenSearchAndBedrockRole"
          },
          {
              "Effect": "Allow",
              "Action": "es:ESHttpPost",
              "Resource": f"arn:aws:es:{region}:{account_id}:domain/{domain_name}/*"
          }
      ]
    }

    role_assume_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Action": [
                    "sts:AssumeRole"
                ],
                "Effect": "Allow",
                "Principal": {
                    "Service": [
                        "opensearchservice.amazonaws.com"
                    ]
                }
            }
        ]
    }

    iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(role_assume_policy))["Role"]
    policy_name = "aos_call_bedrock_policy"
    policy_arn = iam.create_policy(
        PolicyName=policy_name, PolicyDocument=json.dumps(role_policy)
    )["Policy"]["Arn"]
    iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)


def get_aos_client(aos_endpoint):
    session = boto3.Session()
    credentials = session.get_credentials()
    region = session.region_name
    auth = AWSV4SignerAuth(credentials, region)
    aos_endpoint= aos_endpoint.replace('https://', '') if 'https://' in aos_endpoint else aos_endpoint

    client = OpenSearch(
        hosts = [{'host': aos_endpoint, 'port': 443}],
        http_auth = auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        timeout = 60, # 默认超时时间是10 秒，
        max_retries=5, # 重试次数
        retry_on_timeout=True
    )

    return client

def create_aos_model_group(aos_client):
    # POST /_plugins/_ml/model_groups/_register
    # {
    # "name": "remote_model_group",
    # "description": "A model group for remote models"
    # }

    request_body = {
        "name": "remote_model_group",
        "description": "A model group for remote models"
    }

    # Execute the _predict request
    response = aos_client.transport.perform_request(
        method="POST",
        url=f"/_plugins/_ml/model_groups/_register",
        body=json.dumps(request_body)
    )

    return response['model_group_id']

def register_and_deploy_aos_model(aos_client, model_name, model_group_id, description, connecter_id):
    # POST /_plugins/_ml/models/_register?deploy=true
    # {
    #     "name": "cohere embed-multilingual-v3",
    #     "function_name": "remote",
    #     "model_group_id": "zksL94sB0S9ucTLoj1u0",
    #     "description": "embedding for multilingual",
    #     "connector_id": "3ktHC4wB0S9ucTLoRFvx"
    # }

    request_body = {
        "name": model_name,
        "function_name": "remote",
        "model_group_id": model_group_id,
        "description": description,
        "connector_id": connecter_id
    }

    response = aos_client.transport.perform_request(
        method="POST",
        url=f"/_plugins/_ml/models/_register?deploy=true",
        body=json.dumps(request_body)
    )

    return response

def create_ingest_pipeline(aos_client, sparse_model_id, dense_model_id):
    # PUT /_ingest/pipeline/neural-sparse-pipeline
    # {
    #     "description": "neural sparse encoding pipeline",
    #     "processors" : [
    #         {
    #         "sparse_encoding": {
    #             "model_id": "<nerual_sparse_model_id>",
    #             "field_map": {
    #               "content": "sparse_embedding"
    #             }
    #         }
    #         },
    #         {
    #         "text_embedding": {
    #             "model_id": "<cohere_ingest_model_id>",
    #             "field_map": {
    #               "content": "dense_embedding"
    #             }
    #         }
    #         }
    #     ]
    # }

    request_body = {
        "description": "neural sparse encoding pipeline",
        "processors" : [
            {
            "sparse_encoding": {
                "model_id": sparse_model_id,
                "field_map": {
                    "content": "sparse_embedding"
                }
            }
            },
            {
            "text_embedding": {
                "model_id": dense_model_id,
                "field_map": {
                    "content": "dense_embedding"
                }
            }
            }
        ]
    }

    response = aos_client.transport.perform_request(
        method="PUT",
        url=f"/_ingest/pipeline/neural-sparse-pipeline",
        body=json.dumps(request_body)
    )

    return response

def create_query_pipeline(aos_client, sparse_model_id, dense_model_id):
    # PUT /_search/pipeline/hybird-search-pipeline
    # {
    #   "description": "Post processor for hybrid search",
    #   "phase_results_processors": [
    #     {
    #       "normalization-processor": {
    #         "normalization": {
    #           "technique": "l2"
    #         },
    #         "combination": {
    #           "technique": "arithmetic_mean",
    #           "parameters": {
    #             "weights": [
    #               0.3,
    #               0.7
    #             ]
    #           }
    #         }
    #       }
    #     }
    #   ]
    # }
    
    request_body = {
      "description": "Post processor for hybrid search",
      "phase_results_processors": [
        {
          "normalization-processor": {
            "normalization": {
              "technique": "l2"
            },
            "combination": {
              "technique": "arithmetic_mean",
              "parameters": {
                "weights": [
                  0.5,
                  0.5
                ]
              }
            }
          }
        }
      ]
    }

    response = aos_client.transport.perform_request(
        method="PUT",
        url=f"/_search/pipeline/hybird-search-pipeline",
        body=json.dumps(request_body)
    )

    return response

def create_index(aos_client, index_name="aos-retrieval"):
    index_mapping = {
        "settings" : {
            "index":{
                "number_of_shards" : 1,
                "number_of_replicas" : 0,
                "knn": "true",
                "knn.algo_param.ef_search": 32
            },
            "default_pipeline": "neural-sparse-pipeline"
        },
        "mappings": {
            "properties": {
                "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                "dense_embedding": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "engine": "nmslib",
                        "space_type": "innerproduct",
                        "parameters": {}
                    }
                },
                "sparse_embedding": {
                    "type": "rank_features"
                }
            }
        }
    }

    # 创建索引
    response = aos_client.indices.create(index=index_name, body=index_mapping)
    return response

def create_bedrock_cohere_connector(account_id, aos_endpoint, input_type='search_document'):
    # input_type could be search_document | search_query
    service = 'es'
    session = boto3.Session()
    credentials = session.get_credentials()
    region = session.region_name
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

    path = '/_plugins/_ml/connectors/_create'
    url = 'https://' + aos_endpoint + path

    role_name = "OpenSearchAndBedrockRole"
    role_arn = "arn:aws:iam::{}:role/{}".format(account_id, role_name)
    model_name = "cohere.embed-multilingual-v3"

    bedrock_url = "https://bedrock-runtime.{}.amazonaws.com/model/{}/invoke".format(region, model_name)

    payload = {
      "name": "Amazon Bedrock Connector: Cohere doc embedding",
      "description": "The connector to the Bedrock Cohere multilingual doc embedding model",
      "version": 1,
      "protocol": "aws_sigv4",
      "parameters": {
        "region": region,
        "service_name": "bedrock"
      },
      "credential": {
        "roleArn": role_arn
      },
      "actions": [
        {
          "action_type": "predict",
          "method": "POST",
          "url": bedrock_url,
          "headers": {
            "content-type": "application/json",
            "x-amz-content-sha256": "required"
          },
          "request_body": "{ \"texts\": ${parameters.texts}, \"input_type\": \"search_document\" }",
          "pre_process_function": "connector.pre_process.cohere.embedding",
          "post_process_function": "connector.post_process.cohere.embedding"
        }
      ]
    }
    headers = {"Content-Type": "application/json"}

    r = requests.post(url, auth=awsauth, json=payload, headers=headers)
    return json.loads(r.text)["connector_id"]

if __name__ == '__main__':
    '''
    Usage : python3 create_cohere_model.py --aos_endpoint "https://vpc-domain66ac69e0-2m4jji7cweof-4fefsofiqdzu3hxammxwq5hth4.us-west-2.es.amazonaws.com"
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--aos_endpoint', type=str, default='', help='aos endpoint')
    parser.add_argument('--sparse_model_id', type=str, default='', help='you can found it in the output of cloudformation')
    parser.add_argument('--index_name', type=str, default='', help='index name')
    args = parser.parse_args()
    aos_endpoint = args.aos_endpoint
    aos_domain = '-'.join(aos_endpoint.split('-')[1:3])
    sparse_model_id = args.sparse_model_id
    index_name = args.index_name

    iam = boto3.client('iam')
    session = boto3.session.Session()

    region = session.region_name
    print("Current Region:", region)

    sts_client = session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    print("Account ID:", account_id)

    role_name = 'OpenSearchAndBedrockRole'
    create_bedrock_caller_role(aos_domain, account_id, region)

    aos_client = get_aos_client(aos_endpoint)
    model_group_id = create_aos_model_group(aos_client)
    print(f"model_group_id:{model_group_id}")

    response = create_index(aos_client, index_name)
    print(f"index:{response}")

    cohere_doc_emb_connector_id = create_bedrock_cohere_connector(account_id, aos_endpoint, 'search_document')
    cohere_query_emb_connector_id = create_bedrock_cohere_connector(account_id, aos_endpoint, 'search_query')
    print(f"cohere_doc_emb_connector_id: {cohere_doc_emb_connector_id}")
    print(f"cohere_query_emb_connector_id: {cohere_query_emb_connector_id}")

    response1 = register_and_deploy_aos_model(aos_client, model_name='cohere multilingual doc', model_group_id=model_group_id, description="embed-multilingual-v3 for doc", connecter_id=cohere_doc_emb_connector_id)
    response2 = register_and_deploy_aos_model(aos_client, model_name='cohere multilingual query', model_group_id=model_group_id, description="embed-multilingual-v3 for query", connecter_id=cohere_query_emb_connector_id)

    print("response1:")
    print(response1)
    print("response2:")
    print(response2)

    doc_dense_model_id = response1['model_id']
    print(f"doc_dense_model_id:{doc_dense_model_id}")
    query_dense_model_id = response2['model_id']
    print(f"query_dense_model_id:{query_dense_model_id}")

    # pipeline = neural-sparse-pipeline
    response = create_ingest_pipeline(aos_client, sparse_model_id, doc_dense_model_id)
    print("create_ingest_pipeline:")
    print(response)

    response = create_query_pipeline(aos_client, sparse_model_id, query_dense_model_id)
    print("create_query_pipeline:")
    print(response)