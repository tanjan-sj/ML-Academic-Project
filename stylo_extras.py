df['category_id'] = df['title'].factorize()[0]

category_id_df = df[['title', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'title']].values)

sentiment_counts = df.title.value_counts()



import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer(tweet):

    tokens = nltk.word_tokenize(tweet)
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

#print(normalizer("সময় উপযোগী একটা লিখা"))

pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
df['normalized_data'] = df.data.apply(normalizer)
#print(df[['data','normalized_data']].head())


from nltk import ngrams
def ngrams(input_list):
    #onegrams = input_list
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams+trigrams
df['grams'] = df.normalized_data.apply(ngrams)


import collections
def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt


import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(1,2))

vectorized_data = count_vectorizer.fit_transform(df.data)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))


def sentiment2target(sentiment):
    return {
        'Like (ভাল)': 0,
        'Smiley (স্মাইলি)': 1,
        'HaHa(হা হা)' : 2,
        'Sad (দু: খিত)': 3,
        'Skip ( বোঝতে পারছি না )': 4,
        'Love(ভালবাসা)': 5,
        'WOW(কি দারুন)': 6,
        'Blush(গোলাপী আভা)': 7,
        'Consciousness (চেতনাবাদ)': 8,
        'Rocking (আন্দোলিত হত্তয়া)': 9,
        'Bad (খারাপ)': 10,
        'Angry (রাগান্বিত)': 11,
        'Fail (ব্যর্থ)': 12,
        'Provocative (উস্কানিমুলক)': 13,
        'Shocking (অতিশয় বেদনাদায়ক)': 14,
        'Protestant (প্রতিবাদমূলক)': 15,
        'Evil (জঘন্য)': 16,
        'Skeptical (সন্দেহপ্রবণ)': 17,
    }[sentiment]


#print(sentiment_counts)
# Like (ভাল)                      11099
# Smiley (স্মাইলি)                 3014
# HaHa(হা হা)                      1636
# Sad (দু: খিত)                    1597
# Skip ( বোঝতে পারছি না )          1344
# Love(ভালবাসা)                    1312
# WOW(কি দারুন)                    1094
# Blush(গোলাপী আভা)                810
# Consciousness (চেতনাবাদ)          809
# Rocking (আন্দোলিত হত্তয়া)       760
# Bad (খারাপ)                       690
# Angry (রাগান্বিত)                 655
# Fail (ব্যর্থ)                     639
# Provocative (উস্কানিমুলক)         636
# Shocking (অতিশয় বেদনাদায়ক)      452
# Protestant (প্রতিবাদমূলক)         446
# Evil (জঘন্য)                      390
# Skeptical (সন্দেহপ্রবণ)           348
targets = df.title.apply(sentiment2target)

from sklearn.model_selection import train_test_split
data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.2, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]

print("svm started")
#svm

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
clf_output = clf.fit(data_train, targets_train)


print(clf.score(data_test, targets_test))