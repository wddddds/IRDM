import pandas as pd
import numpy as np

from math import log
from collections import defaultdict


def DCG(result, ndeval, K, alpha):

    first_k_doc = result['document_id'][0:K]
    DCG = 0
    i = 1
    indent_dict = defaultdict(lambda: 0)
    for doc in first_k_doc:
        docs = ndeval[ndeval['document_id'] == doc]
        if len(docs) != 0:
            for ind, d in docs.iterrows():
                DCG += ((1 - alpha) ** indent_dict[d['indent']])/log(i+1, 2)
                indent_dict[d['indent']] += 1
        i += 1

    return DCG


def IDCG(ndeval, K, alpha):

    docum = defaultdict(lambda: 0)

    for doc in ndeval['document_id']:
        sum = ndeval[ndeval['document_id'] == doc]['relevance'].sum()
        docum[doc] = sum

    ndeval['sum'] = ndeval['document_id'].apply(lambda x: docum[x])
    sorted_ndeval = ndeval.sort_values(['sum'], ascending=0)

    ideal_rank = sorted_ndeval['document_id'].drop_duplicates()

    IDCG = 0
    i = 1
    indent_dict = defaultdict(lambda: 0)
    for doc in ideal_rank:
        docs = ndeval[ndeval['document_id'] == doc]
        if len(docs) != 0:
            for ind, d in docs.iterrows():
                IDCG += ((1 - alpha) ** indent_dict[d['indent']])/log(i+1, 2)
                indent_dict[d['indent']] += 1
        i += 1

    return IDCG


if __name__ == '__main__':
    alpha = 0.9

    data = pd.read_csv('qrels.adhoc.txt', header=None)

    ndeval = pd.DataFrame()
    ndeval['id'] = data[0].apply(lambda x:x.split(' ')[0])
    ndeval['document_id'] = data[0].apply(lambda x:x.split(' ')[2])
    ndeval['relevance'] = data[0].apply(lambda x:int(x.split(' ')[3]))
    ndeval['relevance'] = ndeval['relevance'].apply(lambda x: 1 if x >= 1 else 0)
    ndeval['indent'] = data[0].apply(lambda x:int(x.split(' ')[1]))


    # print adhoc.head(5)

    re = pd.read_csv('Protfolio_ranking_b_-4.txt', header=None)

    res = pd.DataFrame()
    res['id'] = re[0].apply(lambda x:x.split(' ')[0])
    res['document_id'] = re[0].apply(lambda x:x.split(' ')[1])
    res['rank'] = re[0].apply(lambda x:int(x.split(' ')[2]))


    id_set = res['id'].drop_duplicates()
    K_set = [1, 5, 10, 20, 30, 40, 50]
    for K in K_set:

        NDCG = 0
        num_id = len(id_set)
        for id in id_set:

            result = res[res['id'] == id]
            nd = ndeval[ndeval['id'] == id]

            NDCG += DCG(result, nd, K, alpha)/IDCG(nd,K,alpha)

        ndcg = NDCG/num_id

        print K, '|', ndcg