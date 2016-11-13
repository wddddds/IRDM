import pandas as pd
import math
import sys

from collections import defaultdict


def content_to_vector(content):
    vector = defaultdict(int)
    word = content.split(' ')
    word.pop()
    for w in word:
        w = w.split(':')
        vector[w[0]] = int(w[1])

    return vector


def cos_similarity(q, D):
    intersection = set(q.keys()) & set(D.keys())
    numerator = sum([q[x] * D[x] for x in intersection])

    sum1 = sum([q[x]**2 for x in q.keys()])
    sum2 = sum([D[x]**2 for x in D.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


if __name__ == '__main__':

    re = pd.read_csv('MyBM25.txt', header=None)
    res = pd.DataFrame()
    res['id'] = re[0].apply(lambda x:x[0:3])
    res['document_id'] = re[0].apply(lambda x:x[7:32])
    res['rank'] = re[0].apply(lambda x:int(x.split(' ')[3]))


    q = pd.read_csv('query_term_vectors.dat',header=None)
    query = pd.DataFrame()
    query['id'] = q[0].apply(lambda x: x[0:3])
    query['content'] = q[0].apply(lambda x: x[4::])

    d = pd.read_csv('document_term_vectors.dat', header=None)
    document = pd.DataFrame()
    document['id'] = d[0].apply(lambda x: x[0:25])
    document['content'] = d[0].apply(lambda x: x[26::])

    MMR_rank = pd.DataFrame()

    for i, rows in query.iterrows():
        lamb = 0.50

        query_id = rows['id']
        rank = res.loc[res['id'] == query_id]
        rank = rank.loc[rank['rank'] < 100]
        # print rank

        rank['content'] = rank['document_id'].apply(lambda x: document.loc[document['id']==x].iloc[0]['content'])
        rank['query'] = rank['id'].apply(lambda x: query.loc[query['id']==query_id].iloc[0]['content'])
        rank['word_vector'] = rank['content'].apply(lambda x: content_to_vector(x))
        rank['query_vector'] = rank['query'].apply(lambda x: content_to_vector(x))

        current_best = rank.loc[rank['rank'] == 0]
        new_rank = current_best
        new_rank['query_similarity'] = 0
        new_rank['document_similarity'] = 0
        new_rank['MMR_score'] = 0


        document_left = rank.loc[rank['rank'] != 0]
        document_left = document_left.reset_index(drop=True)

        while len(document_left) != 0:
            print 'processin',i+1,'th query, ' 'document left: ', len(document_left)
            document_left['query_similarity'] = 0

            document_left['document_similarity'] = 0
            for j, document_to_be_ranked in document_left.iterrows():

                similarity_with_q = cos_similarity(document_to_be_ranked['query_vector'],document_to_be_ranked['word_vector'])
                document_left.loc[j, 'query_similarity'] = similarity_with_q

                content_to_be_ranked = document_to_be_ranked['word_vector']
                cos_sim = []
                for k, document_ranked in new_rank.iterrows():
                    content_ranked = document_ranked['word_vector']
                    cos_sim.append(cos_similarity(content_ranked,content_to_be_ranked))
                similarity_with_D = max(cos_sim)
                document_left.loc[j, 'document_similarity'] = similarity_with_D

            document_left['MMR_score'] = lamb * document_left['query_similarity'] - (1-lamb)*document_left['document_similarity']
            max_index = document_left['MMR_score'].argmax()
            current_best = document_left.loc[[max_index]]

            frame = [new_rank, current_best]
            new_rank = pd.concat(frame)
            index_of_current_best = current_best.index[0]
            document_left = document_left[document_left['document_id'] != current_best['document_id'].iloc[0]]
            document_left.reset_index(drop=True)

        print new_rank

        MMR_rank = pd.concat([MMR_rank, new_rank])

    count = 1
    for index, row in MMR_rank.iterrows():
        orig_stdout = sys.stdout
        f = file('MMR_ranking.txt', 'a')
        sys.stdout = f
        if count%100 != 0:
            Rank = count%100
        else:
            Rank = 100
        print row['id'],row['document_id'],Rank
        sys.stdout = orig_stdout
        f.close()
        count += 1



