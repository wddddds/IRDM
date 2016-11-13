import pandas as pd
import pylab as pl
import numpy as np
from collections import defaultdict


if __name__ == '__main__':
    data = pd.read_csv('part-00000.txt', header=None)

    df = pd.DataFrame()
    df['first'] = data[0].apply(lambda x: x.split(' ')[0])
    df['second'] = data[0].apply(lambda x: x.split(' ')[1].split('\t')[0])
    df['frequence'] = data[0].apply(lambda x: int(x.split(' ')[1].split('\t')[1]))

    juliet = df[df['first'] == 'juliet']
    juliet = juliet.sort_values(['frequence'], ascending=False)
    # romeo = romeo.reset_index(drop=True)

    hist = defaultdict(lambda :0)

    for i, d in juliet.iterrows():
        hist[d['frequence']] += 1

    X = np.arange(len(hist))
    pl.bar(X, hist.values())
    pl.xticks(X, hist.keys())
    ymax = max(hist.values()) + 1
    pl.ylim(0, ymax)
    pl.show()

    martino = df[df['first'] == 'martino']

    print martino
