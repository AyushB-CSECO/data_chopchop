import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from operator import itemgetter
import pandas as pd

def running_ttr(corpus,k):
	i = 0
	while i+k <= len(corpus):
		ttr_list = list([])
		sample = corpus[i:i+k]
		sample_ttr = len(set(sample))/k
		ttr_list.append(sample_ttr)
		i = i+1
	result = round(sum(ttr_list)/len(ttr_list),4)
	return(result)

stop_words = set(stopwords.words('english'))

#read the corpus
# bryant
words_bryant = nltk.Text(nltk.corpus.gutenberg.words('bryant-stories.txt'))
words_bryant = [word.lower() for word in words_bryant if word.isalpha()]		# Converting alphabets to lower 
words_bryant = [word for word in words_bryant if word not in stop_words]		# Removing stopwords from words
# print(words_bryant)

#emma
words_emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
words_emma = [word.lower() for word in words_emma if word.isalpha()]
words_emma = [word for word in words_emma if word not in stop_words]
# print(words_emma)

fdist = FreqDist(words_emma)
print('No of words = {}'.format(len(words_emma)))
print('No of uniques words = {}'.format(len(set(words_emma))))
# for x,v in fdist.most_common(10):
# 	print(x,round(v/len(words_bryant),4))

# TTR
ttr_bryant = round(len(set(words_bryant))/len(words_bryant),4)	#static
ttr_emma = round(len(set(words_emma))/len(words_emma),4)
print('TTR_bryant = {}'.format(ttr_bryant))
print('TTR_emma = {}'.format(ttr_emma))

run_ttr_bryant = running_ttr(words_bryant,1000)					#running
run_ttr_emma = running_ttr(words_emma,1000)
print('Running_TTR_bryant = {}'.format(run_ttr_bryant))
print('Running_TTR_emma = {}'.format(run_ttr_emma))


# Creating rank-frequency table
frequency = {}
for word in words_emma:
	count = frequency.get(word,0)
	frequency[word] = count + 1

frequency = list(reversed(sorted(frequency.items(), key = itemgetter(1))))
# print(frequency)

rank = 1
column_header = ['rank', 'frequency', 'rank_freq']
rank_freq_tbl = pd.DataFrame(columns = column_header)

for word, freq in frequency:
	rank_freq_tbl.loc[word] = [rank, freq, rank*freq]
	rank = rank+1
# print(rank_freq_tbl)
