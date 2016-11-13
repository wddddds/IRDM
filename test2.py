import pandas as pd


data = pd.read_csv('test.txt', header=None)

df = pd.DataFrame()
df['first'] = data[0].apply(lambda x: x.split(' ')[0])
df['second'] = data[0].apply(lambda x: x.split(' ')[1].split('\t')[0])
df['frequence'] = data[0].apply(lambda x: int(x.split(' ')[1].split('\t')[1]))
df['cumulative_frequence'] = 0

c = 0
for i, d in df.iterrows():
    c += d['frequence']
    df['cumulative_frequence'].loc[i] = c

print df