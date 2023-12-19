# -*- coding: utf-8 -*-
NSDC Data Science Contest - Fake News Detection - Md. Nafiz Imtiaz

**Project Description:**

This project will introduce students to an array of skills as they strive to create a fake news detection model to classify a given news article as real or fake. Fake News Detection leverages both Natural Language Processing and Machine Learning skills - how to represent text in a machine-understandable format so as to classify the text and extract whether a news is fake or real. We will also cover visualizations and how to deploy models in the real world.

**Dataset**

This is the link to the dataset for the contest:
https://raw.githubusercontent.com/raima2001/HerWILL-NSDC-DS-Contest/main/news_dataset_subset%20(1).csv

Task 1: Import all the necessary libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import seaborn as sns

"""Task 2: Now read the dataset using Pandas (Marks: 1)"""

url = 'https://raw.githubusercontent.com/raima2001/HerWILL-NSDC-DS-Contest/main/news_dataset_subset%20(1).csv'

data = pd.read_csv(url)

"""Task 3: Let's see what the data looks like. We can use the `head` function which returns the first 5 rows of the dataframe. (Marks: 1)"""

data.head()

"""
Task 4: Use the `describe()` function which gives us a summary of the data. (Marks: 1)"""

data.describe()

"""Task 5: Use `value_counts()` function to get the count of each unique value in the `word_label` column. (Marks: 1)"""

word_label_counts = data['word_label'].value_counts()

print(word_label_counts)

"""Task 6: Now that we have found the value_counts, create a bar plot to show the distribution. You can refer to this on how to create bar plots:
https://www.analyticsvidhya.com/blog/2021/08/understanding-bar-plots-in-python-beginners-guide-to-data-visualization/
Make sure you are giving a title, x-label and y-label.

Marks: 3
"""

data['word_label'].value_counts().plot(kind='bar')
data['word_label'].value_counts()
plt.xlabel("Word label")
plt.ylabel("Numbers")
plt.title("Distribution showing Real and Fake news")
plt.show()

"""Task7: Use info() function to get all the necessary details of your data. (Marks:1)"""

data.info()

"""Task 8: You must have noticed that there are null values in the dataset. So write a code to see how many total null values we have in the 'text' column. Hint: use the isnull() and sum() functions for this.

Marks: 1
"""

data.isnull().sum()

"""Task 9: Unless we fill in the null values, we will get errors later when we tokenized it. So us fillna() function and fill it with 'No info'.
(Marks: 2)
"""

data['text'].fillna("No info")

"""Task 10: Lets turn the text column to string by using astype(str) function. (Marks: 2)"""

data['text']=data['text'].astype(str)

data.dtypes

"""---

Text preprocessing with Natural Language Processing.

---
---

Task 11: Apply word_tokenize to tokenize the sentences into words. (Marks: 2)
"""

import nltk
nltk.download('punkt')
data['text'] = data['text'].apply(word_tokenize)

data['text'][1]

"""Task 12: Apply isalpha() to remove punctuations and symbols. (Marks: 2)"""

data['text'] = data['text'].apply(lambda x: [item for item in x if item.isalpha()])

"""Task 13: Apply islower() to turn all the sentences to lowercase. (Marks: 2)"""

data['text'] = data['text'].apply(lambda x: [item for item in x if item.islower()])

"""Task 14: Apply stopwords to remove all the filler words.
(Marks: 2)
"""

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

data['text'] = data['text'].apply(lambda x: [item for item in x if item not in stop_words])


Task 15:
Apply PorterStemmer() to your 'text' column
(Marks:2)
"""

from nltk.stem import PorterStemmer
ps = PorterStemmer()

data['text'] = data['text'].apply(lambda x: [ps.stem(item) for item in x])

"""Task 16: Join the 'text' column to get the full propocessed form. For simplicity, I am giving you part of the code:  **.apply(lambda x: " ".join(x))** (Marks:1)"""

data['text'] = data['text'].apply(lambda x: " ".join(x))

"""Task 17: Split the 'text' column data for training. Use the first 5000 rows for training. (Marks:1)

"""

train_text = data.text[:5000]

"""Task 18: Split the 'text' column data for testing. Use the remaining rows for training. Hint: df.text[5000:]
(Marks:1)
"""

test_text = data.text[5000:]

"""Task 19:
Now let us do the same for the 'word_label' column. Split it into training and testing. First 5000 rows for training and remaining for testing. (Marks:2)
"""

train_word_label = data.word_label[:5000]

test_word_label = data.word_label[5000:]

"""Task 20: Initiatize CountVectorizer() (Marks:2)


"""

cv = CountVectorizer(min_df=0, max_df=1, binary = False, ngram_range = (0,3))

"""Task 21: Fit transform the training text. (Marks:1)"""

cv_train_text = cv.fit_transform(train_text)

"""Task 22: Transform the test  text. (Marks:1)"""

cv_test_text = cv.transform(test_text)

"""Task 23: Initialize the LabelBinarizer() (Marks:1)"""

lb = LabelBinarizer()

"""Task 24: Fit transform the training labels. (Marks:1)"""

lb_train_word_label = lb.fit_transform(train_word_label)

"""Task 25: Fit transform the testing labels. (Marks:1)"""

lb_test_word_label = lb.fit_transform(test_word_label)

"""Task 26:
- Initialize the Multinomial Naive Bayes Model
- Fit the data to the model
-  Predict the labels
- find the Accuracy

Marks:4
"""

mnb = MultinomialNB()

mnb_bow = mnb.fit(cv_train_text, lb_train_word_label)

mnb_bow_predict = mnb.predict(cv_test_text)

mnb_bow_score = accuracy_score(lb_test_binary, mnb_bow_predict)
print("Accuracy :", mnb_bow_score)


Task 27: Apply WordCloud to get a visual representation of the most used words from 'fake' news. Marks:2
"""

# Commented out IPython magic to ensure Python compatibility.
# word cloud for positive review words in the entire dataset
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# %matplotlib inline

#join all the positive reviews
real_words = ' '.join(list(df[df['word_label'] == 'fake']['text']))

#word cloud for positive words
wordcloud = WordCloud(width=800, height=800, random_state=21, max_font_size=300).generate(real_words)

plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
