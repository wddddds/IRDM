import pandas as pd
import numpy as np
import sys

from math import log


if __name__ == '__main__':

    q = pd.read_csv('query_term_vectors.dat',header=None)
    query = pd.DataFrame()
    query['id'] = q[0].apply(lambda x: x[0:3])
    query['content'] = q[0].apply(lambda x: x[4::])

    d = pd.read_csv('document_term_vectors.dat', header=None)
    document = pd.DataFrame()
    document['id'] = d[0].apply(lambda x: x[0:25])
    document['content'] = d[0].apply(lambda x: x[26::])
    document = document.loc[0:200]

    docs_tf = {}
    idf = {}
    doc_length = {}
    vocab = set()

    count = 0
    c = 0
    total_length = 0

    # compute tf, idf for each documents and words.
    for index, rows in document.iterrows():
        dd = {}
        total_words = 0

        words = rows['content'].split(' ')
        words.pop()
        for word in words:
            w = word.split(':')
            vocab.add(w[0])
            dd.setdefault(w[0], 0)
            dd[w[0]] = w[1]
            total_words += int(w[1])

        for k, v in dd.iteritems():
            dd[k] = 1.* int(v) / total_words

        docs_tf[rows['id']] = dd
        doc_length[rows['id']] = total_words

        count += 1
        if count % 500 == 0:
            print 'document processed: ', count

        total_length += total_words

    for w in list(vocab):
        docs_with_w = 0
        for path, doc_tf in docs_tf.iteritems():
            if w in doc_tf:
                docs_with_w += 1
        idf[w] = log((len(docs_tf) - docs_with_w + 0.5)/(docs_with_w + 0.5))

        c += 1
        if c % 500 == 0:
            print 'word processed ', c

    ave_length = total_length/count

    res = []

    total_docs = len(docs_tf)
    print len(docs_tf)

    for i, rows in query.iterrows():

        words = rows['content'].split(' ')
        words.pop()
        BM25_score = []
        doc_count = 0

        result = []
        for doc, doc_tf in docs_tf.iteritems():

            score = 0
            for word in words:
                w = word.split(':')
                if w[0] in doc_tf:
                    score += idf[w[0]]*((doc_tf[w[0]] * 2.5)/(doc_tf[w[0]] + 1.5 *
                                                                   (1 - 0.75 + 0.75 * doc_length[doc]/ave_length)))

            BM25_score.append(score)
            s = -np.asarray(BM25_score)
            order = s.argsort()
            rank = order.argsort()

            result.append((rows.id, 'Q0', doc, score, 'bm25'))
            result = sorted(result, key=lambda r : r[3], reverse=True)

            # print rows.id, 'Q0', doc, rank[doc_count], score, 'bm25'
            # doc_count +=1

        n = 0
        for r in result:
            if r[3] != 0:
                print " ".join(str(x) for x in r[0:3]), n, " ".join(str(x) for x in r[3:5])
                orig_stdout = sys.stdout
                f = file('MyBM25.txt', 'a')
                sys.stdout = f
                print " ".join(str(x) for x in r[0:3]), n, " ".join(str(x) for x in r[3:5])
                sys.stdout = orig_stdout
                f.close()
                # print r[0:3],n, r[3:5]
                n += 1

        res = res + result