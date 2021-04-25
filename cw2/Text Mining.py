# -*- coding: utf-8 -*-
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

corona = pd.read_csv('data/Corona_NLP_train.csv',encoding='latin-1')
stopwords = np.loadtxt('data/stopwords.txt',dtype=object)
start = time.time()

# 1
# all possible sentiments
sen,sen_fre = np.unique(corona.Sentiment,return_counts=True)
print('There are ',len(sen),' different kinds of sentiments: ',sen)

# 2nd most popular sentiment
dic_s = dict(zip(sen,sen_fre))
dic_s = sorted(dic_s.items(),key=lambda item:item[1],reverse=True)
print('The second most popular sentiment is: ',dic_s[1][0])

# the date with the greatest number of extremely positive tweets
ep = corona[corona.Sentiment == 'Extremely Positive'].TweetAt# dates of all extremely positive tweets
ep = ep.groupby(ep)
temp = dict(ep.count())# get the frequency of extremely positive tweets for every date
temp = sorted(temp.items(),key=lambda item:item[1],reverse=True)
print('The date with the greatest number of extremely positive tweets: ',temp[0][0])

# operations on message
# rules for non-alphabetical characters and additional spaces between words
reg1 = re.compile(r'[^a-z]')
reg2 = re.compile(r'\s{2,}')
message = corona.OriginalTweet.str.lower() # convert to lower case
message = message.str.replace(reg1,' ') 
message = message.str.replace(reg2,' ')
message = message.str.strip() # remove leading and trailing spaces
corona.OriginalTweet = message


# 2
documents = corona.OriginalTweet.str.split(' ')
corpus = np.concatenate(documents.values)
w_list,fre = np.unique(corpus,return_counts=True)
w_fre = pd.Series(data=fre,index=w_list)
w_fre.sort_values(ascending=False,inplace=True)
print('total number of words: ',len(corpus))
print('number of unique words in corpus: ',len(fre))
print('10 most frequent words:\n',w_fre[:10])

# operation on corpus
mylen = np.vectorize(len) #vectorize the len function
# remove stopwords and words with length <= 2
w_fre = w_fre[(~ w_fre.index.isin(stopwords))&(mylen(w_fre.index) > 2)]
w_fre.sort_values(ascending=False,inplace=True)
print('total number of words: ',sum(w_fre))
print('10 most frequent words:')
print(w_fre[:10])


# 3
d = documents.apply(lambda x:list(set(x))) # ensure every words only appear once in each document
corpus1 = np.concatenate(d.values)
w_list,fre = np.unique(corpus1,return_counts=True)
w_fre = pd.Series(data=fre,index=w_list)
# remove stopwords and short words
w_fre = w_fre[(~ w_fre.index.isin(stopwords))&(mylen(w_fre.index) > 2)]
w_fre.sort_values(ascending=True,inplace=True)
data = w_fre.values/len(corona)# compute the fraction
# plot the line chart
plt.plot(data)
plt.title('word frequencies')
plt.xlabel('words')
plt.ylabel('frequencies')
plt.show()


# 4
cv = CountVectorizer()
# a sparse representation of the term-document matrix
cv_fit=cv.fit_transform(corona.OriginalTweet)
# multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(cv_fit,corona.Sentiment)
print('the classifier error rate = ',1 - clf.score(cv_fit,corona.Sentiment))

end = time.time()
print(end-start)
