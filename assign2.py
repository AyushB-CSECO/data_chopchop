import pandas as pd
import math
import random
import numpy as np 
from numpy.linalg import norm
from operator import itemgetter
import nltk
from nltk.corpus import stopwords

def cos_sim(a,b):
	a = np.array(a)
	b = np.array(b)
	cos_sim = abs(np.dot(a,b))/(norm(a)*norm(b))
	return(cos_sim)

stop_words = set(stopwords.words('english'))

# Julius Caesar
words_caesar = nltk.Text(nltk.corpus.gutenberg.words('shakespeare-caesar.txt'))
words_caesar = [word.lower() for word in words_caesar if word.isalpha()]
words_caesar = [word for word in words_caesar if word not in stop_words]
# print('Julius Caeser')
# print(words_caesar[0:100])
# print("==================================================")

# Hamlet
words_hamlet = nltk.Text(nltk.corpus.gutenberg.words('shakespeare-hamlet.txt'))
words_hamlet = [word.lower() for word in words_hamlet if word.isalpha()]
words_hamlet = [word for word in words_hamlet if word not in stop_words]
# print(' ')
# print('hamlet')
# print(words_hamlet[0:100])
# print("=================================================")

# Macbeth
words_macbeth = nltk.Text(nltk.corpus.gutenberg.words('shakespeare-macbeth.txt'))
words_macbeth = [word.lower() for word in words_macbeth if word.isalpha()]
words_macbeth = [word for word in words_macbeth if word not in stop_words]
# print(' ')
# print('macbeth')
# print(words_macbeth[0:100])

V = list(set(words_macbeth + words_hamlet + words_caesar))
# print(len(V))
# print(V[0:100])

# Q1: Construct a tf-idf matrix using log weighting for the corpus Shakespeare play ?
column_header = ['Caesar', 'Hamlet', 'Macbeth']
bin_inc_mat = pd.DataFrame(columns = column_header)

for word in V:
	bin_inc_mat.loc[word] = [0,0,0]
	if word in words_caesar:
		bin_inc_mat.loc[word, 'Caesar'] = 1
	if word in words_hamlet:
		bin_inc_mat.loc[word,'Hamlet'] = 1
	if word in words_macbeth:
		bin_inc_mat.loc[word, 'Macbeth'] = 1

# print(bin_inc_mat)

caesar_count = {}
for word in words_caesar:
	count = caesar_count.get(word,0)
	caesar_count[word] = count + 1
caesar_count = dict(reversed(sorted(caesar_count.items(), key = itemgetter(1))))
# print(caesar_count)

hamlet_count = {}
for word in words_hamlet:
	count = hamlet_count.get(word,0)
	hamlet_count[word] = count + 1
hamlet_count = dict(reversed(sorted(hamlet_count.items(), key = itemgetter(1))))
# print(hamlet_count)

macbeth_count = {}
for word in words_macbeth:
	count = macbeth_count.get(word,0)
	macbeth_count[word] = count + 1
macbeth_count = dict(reversed(sorted(macbeth_count.items(), key = itemgetter(1))))
# print(macbeth_count)

tfidf_mat = pd.DataFrame(columns = column_header)
for word in V:
	tfidf_mat.loc[word, 'Caesar'] = round(caesar_count.get(word,0)*math.log(3/sum(bin_inc_mat.loc[word]),10),2)
	tfidf_mat.loc[word, 'Hamlet'] = round(hamlet_count.get(word,0)*math.log(3/sum(bin_inc_mat.loc[word]),10),2)
	tfidf_mat.loc[word, 'Macbeth'] = round(macbeth_count.get(word,0)*math.log(3/sum(bin_inc_mat.loc[word]),10),2)
# print(tf_mat)

# Q2: Construct a query vector consisting of terms from the vocabulary and find the ranks of
# 	  the plays with respect to the query

query_doc = random.choices(V, k = 20000)

# Modify binary incidence matrix defined above
for word in bin_inc_mat.index:
	if word in query_doc:
		bin_inc_mat.loc[word, 'Query'] = 1
	else:
		bin_inc_mat.loc[word, 'Query'] = 0
# print(bin_inc_mat)

query_count = {}
for word in query_doc:
	count = query_count.get(word,0)
	query_count[word] = count + 1
query_count = dict(reversed(sorted(query_count.items(), key = itemgetter(1))))
# print(query_count)

for word in tfidf_mat.index:
	tfidf_mat.loc[word,'Caesar'] = round(caesar_count.get(word,0)*math.log(4/sum(bin_inc_mat.loc[word]),10),2)
	tfidf_mat.loc[word,'Hamlet'] = round(hamlet_count.get(word,0)*math.log(4/sum(bin_inc_mat.loc[word]),10),2)
	tfidf_mat.loc[word,'Macbeth'] = round(macbeth_count.get(word,0)*math.log(4/sum(bin_inc_mat.loc[word]),10),2)
	tfidf_mat.loc[word,'Query'] = round(query_count.get(word,0)*math.log(4/sum(bin_inc_mat.loc[word]),10),2)
# print(tfidf_mat)

query_cos_sim = pd.DataFrame([0,0,0])
query_cos_sim.index = column_header
for colname in query_cos_sim.index:
	query_cos_sim.loc[colname] = round(cos_sim(tfidf_mat.loc[:,colname], tfidf_mat.loc[:,'Query']),2)
query_cos_sim = query_cos_sim.sort_values(0, ascending = False)
query_cos_sim.rename({0:'cos_sim'}, inplace = True)
query_cos_sim.loc[:,'rank'] = range(1,len(query_cos_sim.index)+1)
# print(query_cos_sim)
