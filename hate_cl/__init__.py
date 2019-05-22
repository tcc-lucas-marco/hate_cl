from pickle import Pickler, Unpickler
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


class HateCl:
    def __init__(self):
        self.classifier = Unpickler(open('hate_cl/randomforest.sav', 'rb')).load()

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


# Testing
# hatecl = HateCl()
# answer = hatecl.predict("Oi como você vem")
# print(answer)

# hatecl.refit([(0, "Oi tudo bom contigo?"), (1, "Oi tudo mal contigo?"),
#               (0, "Frase fraseante"), (1, "Bobo bobinho")])
