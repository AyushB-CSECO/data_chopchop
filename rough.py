from operator import itemgetter
import nltk
import pandas as pd

corpus = ['He', 'has', 'three', 'mobile', 'phones',
'There', 'are', 'several', 'mobile', 'devices', 'under', 'two', 'thousand', 'rupees',
'The', 'battery', 'life', 'of', 'mobile', 'phones', 'are', 'better', 'now',
'Can', 'you', 'restrict', 'your', 'mobile', 'phone', 'usage', 'to', 'one', 'hour',
'The', 'average', 'life', 'of', 'most', 'mobile', 'phones', 'is', 'about', 'two', 'years']

print(len(corpus))
print(len(set(corpus)))
print(corpus)
print(set(corpus))