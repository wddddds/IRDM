import pandas as pd
import numpy as np

from math import log


def DCG(result, adhoc, K):

    first_k_doc = result['document_id'][0:K]
    DCG = 0
    i = 1
    for doc in first_k_doc:
        docs = adhoc[adhoc['document_id'] == doc]
        if len(docs) != 0:
            relevance = docs.reset_index()
            relevance = relevance['relevance'][0]
            DCG += (2 ** relevance)/log(i+1, 2)
        i += 1

    return DCG


def IDCG(adhoc, K):

    sorted_adhoc = adhoc.sort_values(['relevance'], ascending=0)
    rel = sorted_adhoc['relevance'][0:K]

    IDCG = 0
    i = 1
    for relevance in rel:
        IDCG += (2 ** relevance)/log(i+1, 2)
        i += 1

    return IDCG
if __name__ == '__main__':

    data = pd.read_csv('qrels.adhoc.txt', header=None)

    adhoc = pd.DataFrame()
    adhoc['id'] = data[0].apply(lambda x:x[0:3])
    adhoc['document_id'] = data[0].apply(lambda x:x[6:31])
    adhoc['relevance'] = data[0].apply(lambda x:int(x[31::]))


    # print adhoc.head(5)

    re = pd.read_csv('BM25b0.75_0.res', header=None)

    res = pd.DataFrame()
    res['id'] = re[0].apply(lambda x:x[0:3])
    res['document_id'] = re[0].apply(lambda x:x[7:32])
    res['rank'] = re[0].apply(lambda x:int(x.split(' ')[3]))


    print 'bm25'

    id_set = res['id'].drop_duplicates()
    K_set = [1, 5, 10, 20, 30, 40, 50]
    for K in K_set:

        NDCG = 0
        num_id = len(id_set)
        for id in id_set:

            result = res[res['id'] == id]
            ad = adhoc[adhoc['id'] == id]

            NDCG += DCG(result, ad, K)/IDCG(ad,K)

        ndcg = NDCG/num_id

        print K, '|', ndcg