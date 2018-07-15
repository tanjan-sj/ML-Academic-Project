import glob
import errno
import codecs
text, label = [], []

# label.append("post")
# labels.append("tags")

import pandas as pd
import numpy as np

df = pd.read_csv('/home/kingbayeed/Downloads/ML lab all dataset/Fake_News_Detection/fake_or_real_news.csv')
print(df.head())

from io import StringIO
col = ['text', 'label']
df = df[col]

df = df[pd.notnull(df['label'])]
df.columns = ['text', 'label']
df['category_id'] = df['text'].factorize()[0]
category_id_df = df[['text', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'text']].values)
print(df.tail())

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.label).toarray()
labels = df.category_id
print(features.shape)

from sklearn.feature_selection import chi2

N = 2
for text, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(text))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree

count_vect = CountVectorizer()
X_data_full_ = count_vect.fit_transform(df['label'])
tfidf_transformer = TfidfTransformer()
X_data_full_tfidf = tfidf_transformer.fit_transform(X_data_full_)
X_train, X_test, y_train, y_test = train_test_split(X_data_full_tfidf, df['text'], test_size=0.2 , random_state = 61)



print("naive bayes accuracy is")
clf = MultinomialNB().fit(X_train, y_train)
print(clf.score(X_test, y_test))


print("Decision tree accuracy is")
#decision tree starts here
clf = tree.DecisionTreeClassifier()
clf_output= clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

print("SVM with linear kernel accuracy is")
#svm starts here
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

#neural net processor starts here
print("Neural Net Accuracy is")
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=61)

clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

#knn starts here
print("KNN accuracy is")

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

print(neigh.score(X_test, y_test))
