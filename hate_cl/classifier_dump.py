
import arff
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
from pickle import Pickler



data = arff.load(open('./OffComBR3.arff'))
df = DataFrame(data['data'])
df.columns = ['hate', 'sentence']
df['hate'] = df['hate'].apply(lambda x: 1 if x == 'yes' else 0)

X = df['sentence'].tolist()
y = df['hate'].tolist()

cl = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 4))),
               ('clf',
                RandomForestClassifier(n_estimators=100,
                                       max_depth=None,
                                       min_samples_leaf=1,
                                       min_samples_split=2,
                                       min_weight_fraction_leaf=0))])

cl.fit(X, y)


cl_filename = 'randomforest.sav'
df_filename = 'data.sav'

f = open(cl_filename, 'wb')
Pickler(f).dump(cl);
f.close()

f = open(df_filename, 'wb')
Pickler(f).dump(df);
f.close()