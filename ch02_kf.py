# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

def top_counts(count_dict,n=10):
    value_key_pairs = [(count,tz) for tz,count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

def add_prop(group):
    # integer division floors
    births = group.births.astype(float)
    group['prop'] = births/births.sum()
    return group

path = 'ch02/usagov_bitly_data2012-03-16-1331923249.txt'
# open(path).readline()
records = [json.loads(line) for line in open(path)]

time_zones = [rec['tz'] for rec in records if 'tz' in rec]

counts = get_counts(time_zones)

top = top_counts(counts, 15)

"""
another way of doing this
"""
from collections import Counter
counts = Counter(time_zones)
counts.most_common(10)


"""
Counting time zones with Pandas
"""
from pandas import DataFrame, Series

frame = DataFrame(records)
frame

tz_counts = frame['tz'].value_counts()

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()

tz_counts[:10].plot(kind='barh',rot=0)

results = Series([x.split()[0] for x in frame.a.dropna()])
cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows')
by_tz_os =cframe.groupby(['tz',operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
indexer = agg_counts.sum(1).argsort()
count_subset = agg_counts.take(indexer)[-10:]
count_subset.plot(kind='barh',stacked=True)
normed_subset = count_subset.div(count_subset.sum(1),axis=0)
plt.show()

"""
working on the MovieLens data set
"""

path = 'ch02/movielens/'

unames = ['user_id', 'gender', 'age', 'occupation','zip']
users = pd.read_table(path+'users.dat', sep='::',header=None,names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(path+'ratings.dat', sep='::',header=None,names=rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table(path+'movies.dat', sep='::',header=None,names=mnames)

data = pd.merge(pd.merge(ratings,users),movies)

mean_ratings = data.pivot_table(values='rating',index='title',columns='gender',aggfunc='mean')

ratings_by_title = data.groupby('title').size()
active_titles = ratings_by_title.index[ratings_by_title >= 250]
mean_ratings = mean_ratings.ix[active_titles]
top_female_ratings = mean_ratings.sort_values(by='F',ascending=False)

mean_ratings['diff'] = mean_ratings['M']-mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff')

rating_std_by_title = data.groupby('title')['rating'].std()
rating_std_by_title = rating_std_by_title.ix[active_titles]


"""
US BABY NAMES
"""

path = 'ch02/names/'

names1880 = pd.read_csv(path+'yob1880.txt',names=['name','sex','births'])
names1880_grouped = names1880.groupby('sex').births.sum()

years = range(1880,2011)
pieces=[]
columns = ['name', 'sex','births']

for year in years:
    path = 'ch02/names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)
    
#concatenate all of the pieces into a single dataframe
names = pd.concat(pieces, ignore_index=True)

total_births = names.pivot_table(values='births',index='year',columns='sex',aggfunc=sum)
total_births.tail()

total_births.plot(title='Total births by sex and year')

#uses user defined function add_prop
names = names.groupby(['year','sex']).apply(add_prop)

#sanity check
np.allclose(names.groupby(['year','sex']).prop.sum(),1)

def get_top1000(group):
    return group.sort_values(by='births',ascending=False)[:1000]

grouped = names.groupby(['year','sex'])
top1000 = grouped.apply(get_top1000)
top1000

#analyzing naming trends
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']

