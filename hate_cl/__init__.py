import pickle


class HateCl:
    def __init__(self):
        self.classifier = pickle.load(open('hate_cl/randomforest.sav', 'rb'))

    def predict(self, text):
        classified_text = self.classifier.predict_proba([text])[:,1]
        return classified_text[0]

    # def 

# Testing
# hatecl = HateCl()
# answer = hatecl.predict("Oi como você vem")
# print(answer)