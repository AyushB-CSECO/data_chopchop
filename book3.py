import pandas as pd
import gensim 
from gensim import corpora
from gensim import models
from gensim import similarities as sm 
from gensim.parsing.preprocessing import strip_numeric, stem_text


physics_corp_dir = 'https://raw.githubusercontent.com/Ramaseshanr/anlp/master/corpus/phy_corpus.txt'

corpus = pd.read_csv(physics_corp_dir, sep = '\n', header = None)[0]
# print(corpus)

def preprocessing():
	for doc in corpus:
		doc_new = strip_numeric(stem_text(doc))
		yield gensim.utils.tokenize(doc_new, lower = True)

text = preprocessing()
dictionary = corpora.Dictionary(text)
dictionary.filter_extremes(no_below = 1, keep_n = 700)
# print(dictionary)

doc_term_mat = [dictionary.doc2bow(tokens) for tokens in preprocessing()]
tfidf = models.TfidfModel(doc_term_mat)
corpus_tfidf = tfidf[doc_term_mat]
# print(corpus_tfidf)

lsi = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics = 1) # initialize an LSI transformation
doc = 'car acceleration speed far'
vec_bow = dictionary.doc2bow(doc.lower().split())

vec_lsi = lsi[vec_bow]
index = sm.MatrixSimilarity(lsi[doc_term_mat])
unsorted_sm = index[vec_lsi]
sorted_sm = sorted(enumerate(unsorted_sm), key = lambda item: -item[1])
for index, similarity in sorted_sm:
	print(similarity, corpus[index])
	