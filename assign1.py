import nltk
from nltk.corpus import stopwords
import statistics as stats
import pandas as pd
from operator import itemgetter

stop_words = set(stopwords.words('english'))

words_bryant = nltk.Text(nltk.corpus.gutenberg.words('bryant-stories.txt'))
words_bryant = [word.lower() for word in words_bryant if word.isalpha()]
words_bryant = [word for word in words_bryant if word not in stop_words]

# Q1 Write a program to find out wether Mandelbrot's approximation really provides a better
# fit than Zipf's empirical law. Use the same corpus for Zipf and Mandelbrot approximation?
frequency = {}
for word in words_bryant:
	count = frequency.get(word,0)
	frequency[word] = count + 1

frequency = list(reversed(sorted(frequency.items(), key = itemgetter(1))))

rank = 1
column_header = ['rank', 'frequency', 'zipf', 'Mandelbrot']
df = pd.DataFrame(columns = column_header)

for word, freq in frequency:
	df.loc[word] = [rank, freq, rank*freq, int((rank+2.7)*freq)]
	rank = rank + 1

print(df)
print('Std Dev Zipf\'s Estimate = {}'.format(stats.stdev(df.loc[:,'zipf'])))
print('Std Dev Mandelbrot\'s Estimate = {}'.format(stats.stdev(df.loc[:,'Mandelbrot'])))
print("=======================")
# Write  a program for Heap's law and find out the prediction of vocabulary in any corpus. Also
# find out whether it is closer to the actual size of the vocabulary of the same corpus?
n_tokens = len(words_bryant)
n_unq = len(set(words_bryant))
n_pred = int((n_tokens**0.49)*30)
print('Predicted = {}'.format(n_pred))
print('Unique = {}'.format(n_unq))
print('Abs Difference = {}'.format(abs(n_unq - n_pred)))
print('Percentage Difference = {}'.format(round(abs(n_unq - n_pred)/n_unq*100),2))