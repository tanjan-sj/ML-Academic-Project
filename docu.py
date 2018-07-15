import glob
import errno
import codecs
path1 = '/home/kingbayeed/Downloads/ML lab all dataset/Document_Categorization/category/accident/*.txt'
path2 = '/home/kingbayeed/Downloads/ML lab all dataset/Document_Categorization/category/crime/*.txt'
#path3 = '/home/kingbayeed/Downloads/ML lab all dataset/Document_Categorization/category/education/*.txt'

topics, texts = [], []

files = glob.glob(path1)
# texts.append("post")
# labels.append("tags")
for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            # str = re.sub(' +', ' ', str)
            str = " ".join(str.split())
            topics.append("acc")
            texts.append(str)


    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path2)
for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            topics.append("cri")
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

import pandas as pd
import numpy as np

df = pd.DataFrame({'texts':texts, 'topics':topics})
#print(df.head())

from io import StringIO
col = ['topics', 'texts']
df = df[col]
df = df[pd.notnull(df['texts'])]
df.columns = ['topics', 'texts']
df['category_id'] = df['topics'].factorize()[0]
category_id_df = df[['topics', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'topics']].values)
print(df.tail())

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.texts).toarray()
labels = df.category_id
print(features.shape)

from sklearn.feature_selection import chi2

N = 2
for topics, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(topics))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree

count_vect = CountVectorizer()
X_data_full_ = count_vect.fit_transform(df['texts'])
tfidf_transformer = TfidfTransformer()
X_data_full_tfidf = tfidf_transformer.fit_transform(X_data_full_)
X_train, X_test, y_train, y_test = train_test_split(X_data_full_tfidf, df['topics'], test_size=0.2 , random_state = 61)



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
