import glob
import errno
import codecs
path1 = '/home/kingbayeed/Downloads/ML lab all dataset/Stylogenetics/All Data/Emon Jubayer/*.doc'
path2 = '/home/kingbayeed/Downloads/ML lab all dataset/Stylogenetics/All Data/Hasan Mahbub/*.doc'
path3 = '/home/kingbayeed/Downloads/ML lab all dataset/Stylogenetics/All Data/MZI/*.doc'
path4 = '/home/kingbayeed/Downloads/ML lab all dataset/Stylogenetics/All Data/Nir Shondhani/*.doc'
path5 = '/home/kingbayeed/Downloads/ML lab all dataset/Stylogenetics/All Data/Ronodipom Boshu/*.doc'
path6 = '/home/kingbayeed/Downloads/ML lab all dataset/Stylogenetics/All Data/Tareq Anu/*.doc'
authors, texts = [], []

files = glob.glob(path1)
# texts.append("post")
# labels.append("tags")
for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            # str = re.sub(' +', ' ', str)
            str = " ".join(str.split())
            authors.append("ej")
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

            authors.append("hm")
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path3)
for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            authors.append("mzi")
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
files = glob.glob(path4)
for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            authors.append("ns")
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path5)
for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            authors.append("rb")
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
files = glob.glob(path6)
for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            authors.append("ta")
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
import pandas as pd
import numpy as np


df = pd.DataFrame({'texts':texts, 'authors':authors})
#print(df.head())

from io import StringIO
col = ['authors', 'texts']
df = df[col]
df = df[pd.notnull(df['texts'])]
df.columns = ['authors', 'texts ']
df['category_id'] = df['authors'].factorize()[0]
category_id_df = df[['authors', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'authors']].values)
print(df.tail())

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.texts).toarray()
labels = df.category_id
print(features.shape)

from sklearn.feature_selection import chi2

N = 2
for authors, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(authors))
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
X_train, X_test, y_train, y_test = train_test_split(X_data_full_tfidf, df['authors'], test_size=0.5 , random_state = 52)



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

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=52)

clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

#knn starts here
print("KNN accuracy is")

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

print(neigh.score(X_test, y_test))
