Text Mining.py: 
read the dataset and stopwords;
Part 1:
1)
compute the frequency of the tweets' sentiments;
output all possible sentiments;
2)
sort its frequency and output the 2nd most popular sentiment;
3)
select the dates of all extremely positive tweets, use group by to get the frequency of extremely 
positive tweets, sort it and output the date with the greatest number of extremely positive tweets;
4)
convert messages to lower case, use regular expressions to replace non-alphabetical characters with 
whitespaces and ensure that the words of a message are separated by a single whitespace.

Part 2:
1)
use split to tokenize the tweets and compute words frequency, output the total number of
all words (including repetitions), the number of all distinct words and the 10 most frequent 
words in the corpus;
2)
use vectorization to remove stopwords and words with length <= 2;
recompute words frequency and output the total number of all words (including repetitions) 
and the 10 most frequent words in the corpus.

Part 3:
1)
use the characteristic of set to ensure every word only appear once in a document, 
and compute the word frequency, remove stopwords and short words just like Part2;
2)
sort the word frequency and use it to plot a line chart

Part 4:
1)
use CountVectorizer to create a sparse representation of the term-document matrix;
2)
produce the multinomial Naive Bayes classifier and output the error rate of the classifier.

********************************************************************************************
Image Processing.py:
Part 1:
read the avenger_imdb.jpg, output its size;
convert it from RGB format to grayscale, then convert it from grayscale 
to binary based on the specified threshold;
display the results.

Part 2:
read the bush_house_wikipedia.jpg;
get bush_noise by adding Gaussian noise to the original image;
filter bush_noise with a Gaussian mask, get bush_gaussian;
filter bush_noise with a uniform smoothing mask, get bush_uniform;
display the results.

Part 3:
read the forestry_commission_gov_uk.jpg;
use k-means segmentation on it and get the segments;
use mark_boundaries to get the segmented original image;
display the results.

Part 4:
read the rolland_garros_tv5monde.jpg;
for better performance, filter with a Gaussian mask;
convert to grayscale format;
do canny edge detection;
apply Hough transform on the results of canny edge;
display the results.
