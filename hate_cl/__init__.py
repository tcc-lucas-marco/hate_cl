import arff
import nltk
from pickle import Pickler, Unpickler
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

MODEL_LIST = [
    {'model': RandomForestClassifier(),
    'param': {'tfidf__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
                'clf__n_estimators': [10, 20, 100],
                'clf__max_depth': [None, 2, 10],
                'clf__min_samples_split': [2, 10],
                'clf__min_samples_leaf': [1, 10],
                'clf__min_weight_fraction_leaf': [0, 0.1, 0.5]
            }
    },
    {'model': MLPClassifier(),
    'param': {'tfidf__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
              'clf__alpha': [0, 0.2, 0.5, 0.7, 1],
              'clf__activation': ('identity', 'logistic'),
              'clf__solver': ('lbfgs', 'sgd', 'adam')
            }
    },
    {'model': DecisionTreeClassifier(),
    'param': {'tfidf__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
              'clf__criterion': ('gini', 'entropy'),
              'clf__class_weight': ({0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 2, 1: 1},
                                    {0: 2, 1: 1}, {0: 3, 1: 1}, {0: 4, 1: 1}),
              'clf__min_samples_split': [2, 3, 4, 5, 6, 7, 8]
            }
    },
    {'model': MultinomialNB(),
    'param': {'tfidf__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
              'clf__alpha': [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 1],
              'clf__fit_prior': (True, False),
            }
    },
    {'model': SGDClassifier(),
    'param': {'tfidf__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
              'clf__alpha': (1, 1e-2),
              'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
              'clf__penalty': ('none', 'l2', 'l1','elasticnet')
            }
    },
    {'model': SVC(),
    'param': {'tfidf__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
              'clf__C': [1, 2, 3, 4, 5],
              'clf__kernel': ('rbf', 'linear', 'poly', 'sigmoid'),
              'clf__shrinking': (True, False),
              'clf__probability': (True, False),
              'clf__tol': [1e-4, 1e-3, 1e-2, 0.1, 1]
            }
    }
     
]


class HateCl:
    def __init__(self):
        self.classifier = Unpickler(open('./randomforest.sav', 'rb')).load()
        self.df = self.load_df()
        self.X = self.df['sentence'].tolist()
        self.y = self.df['hate'].tolist()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)

    def load_df(self):
        data = arff.load(open('./OffComBR3.arff'))
        df = DataFrame(data['data'])
        df.columns = ['hate', 'sentence']
        df['hate'] = df['hate'].apply(lambda x: 1 if x == 'yes' else 0)
        return df


    def predict(self, text):
        classified_text = self.classifier.predict_proba([text])[:, 1]
        return classified_text[0]

    def get_samples(self):
        df = Unpickler(open('hate_cl/data.sav', 'rb')).load()
        return df

    def refit(self, samples):
        df = Unpickler(open('hate_cl/data.sav', 'rb')).load()
        aux_df = DataFrame(samples, columns=['hate', 'sentence'])
        df = df.append(aux_df, ignore_index=True)
        print(df)
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
        self.classifier = cl
        cl_filename = 'hate_cl/randomforest.sav'
        df_filename = 'hate_cl/data.sav'

        f = open(cl_filename, 'wb')
        Pickler(f).dump(cl)
        f.close()

        f = open(df_filename, 'wb')
        Pickler(f).dump(df)
        f.close()
    
    # method that uses fit on every model with the defined dataset
    def fit_all(self, dataset):
        pass

    def create_best_cl(self, model, params):
        gs_clf = GridSearchCV(model, params, cv=5, iid=False, n_jobs=-1, scoring='roc_auc_score')
        gs_clf.fit(self.X_train, self.y_train)
        print(gs_clf.best_score_)
        print(gs_clf.best_params_)
        return gs_clf

    def grid_search_all(self):
        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('portuguese')

        for model in MODEL_LIST:
            cl = Pipeline([('tfidf', TfidfVectorizer(strip_accents = 'ascii', lowercase=True, 
                                                stop_words = stopwords)),
                    ('clf',model['model'])])
            gs_clf = self.create_best_cl(cl, model['param'])
            pred = gs_clf.best_estimator_.predict(self.X_test)
            print(classification_report(self.y_test, pred))
            
            #saving cl in file
            cl_filename = model['model'].__class__.__name__ + 'sav'
            f = open(cl_filename, 'wb')
            Pickler(f).dump(cl)
            f.close()

    def create_committee(self):
        pass

    def import_commitee(self):
        pass


# Testing
# hatecl = HateCl()
# answer = hatecl.predict("Oi como você vem")
# print(answer)

# hatecl.refit([(0, "Oi tudo bom contigo?"), (1, "Oi tudo mal contigo?"),
#               (0, "Frase fraseante"), (1, "Bobo bobinho")])
