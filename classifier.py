import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, linear_model
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

RANDOM_SEED = 67
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# Entrenamenduko datuak irakurri
train_df = pd.read_csv("train.csv")
trainX = train_df["text"]
trainY = train_df["category"]

# dev-eko datuak irakurri

dev_df = pd.read_csv("dev.csv")
devX = dev_df["text"]
devY = dev_df["category"]

# test-eko datuak irakurri

test_df = pd.read_csv("test_clean.csv")
testX = test_df["text"]

STOP_WORDS = list(set(stopwords.words("spanish")) - {"no", "sin", "contra", "ni"}) # Hitz negatibo batzuk garrantzitsuak izan ahal dira
'''
class SimpleTokenizer: 
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {id:token for token, id in vocab.items()}
'''
#    def encode(self, text):
#        preproccessed = re.split(r'[,.;:!?/()"”]|\s', text.lower())
#        preproccessed = [item.strip() for item in preproccessed if item.strip()]
#        preproccessed = [item if item in self.str_to_int else "<|unk|>" for item in preproccessed]
#        ids = [self.str_to_int[s] for s in preproccessed]
#        return ids

#    def decode(self, ids):
#        text = " ".join([self.int_to_str[i] for i in ids])
#        text = re.sub(r'\s+[,.;:!?/()"”]', r'\1', text)
#        return text

#text = " ".join(trainX.tolist()).lower() # Hitz guztiak bektore luze bakar baten batu nahi ditugu gero split bat egin ahal izateko. trainX lista batera bihurtu behar da hori egiteko
#preproccessed = re.split(r'[,.;:!?/()"”]|\s', text)
#preproccessed = [item.strip() for item in preproccessed if item.strip()]
#all_tokens = sorted(set(preproccessed))
#all_tokens.extend(['<|endoftext|>', '<|unk|>'])
#vocab = {token:integer for integer, token in enumerate(all_tokens)}
#print(len(vocab.items()))
#print(list(vocab.items())[:20])
#tokenizer = SimpleTokenizer(vocab)

# Bag of Words eta STOP_WORDS aplikatu gure entrenamendu testura
klaseak = trainY.unique() # Bi klaseak bektore baten gorde
bow_klaseka = {}
X = 5 # Gutxienez 5 aldiz ateratzen diren hitzak gorde, besteak kendu
for k in klaseak:
    klaseko_text = train_df[train_df["category"] == k]["text"]
    text_j = " ".join(klaseko_text.tolist()).lower()
    tokens = re.split(r'[,.;:!?/()"”]|\s', text_j)
    tokens = [item.strip() for item in tokens if item.strip()]
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = Counter(tokens)
    bow_klaseka[k] = Counter({token: freq for token, freq in tokens.items() if freq>=X}).most_common(20)
#print("CRITICAL: " + str(bow_klaseka["CRITICAL"]))
#print("CONSPIRACY: " + str(bow_klaseka["CONSPIRACY"]))


# CountVectorizer bat sortu erregresio logistikoa egin ahal izateko
le = preprocessing.LabelEncoder()
le.fit(trainY)
Y_train_lr = le.transform(trainY)
Y_dev_lr = le.transform(devY)

vectorizer1 = TfidfVectorizer(max_features=15000, lowercase=True, binary=False, ngram_range=(1, 2), stop_words=STOP_WORDS, min_df=3, sublinear_tf=True)
vectorizer2 = CountVectorizer(max_features=7500, lowercase=True, binary=True, ngram_range=(1, 2), stop_words=STOP_WORDS, min_df=3)

# Hiperparametroak doitu, kasu onena bilatu
def saiatu_vectorizer(vectorizer, C):

    X_train_lr = vectorizer.fit_transform(trainX)
    X_dev_lr = vectorizer.transform(devX)

    none_weights = []
    balanced_weights = []
    for c in C:
        logreg = linear_model.LogisticRegression(C=c, solver="liblinear", max_iter=5000, class_weight=None)
        logreg2 = linear_model.LogisticRegression(C=c, solver="liblinear", max_iter=5000, class_weight="balanced")
        logreg.fit(X_train_lr, Y_train_lr)
        logreg2.fit(X_train_lr, Y_train_lr)
        lr_baseline = logreg.score(X_dev_lr, Y_dev_lr)
        lr_baseline2 = f1_score(X_dev_lr, Y_dev_lr)
        print(f"None-ren zehaztasuna (metodoa: liblinear) (c: {c}): " + str(lr_baseline))
        print(f"Balanced-en zehaztasuna (metodoa: liblinear) (c: {c}): " + str(lr_baseline2))
# Kasu onenak: count->liblinear_none[0.12] eta tfid->liblinear_none[4.0]
C1 = [0.5, 1.0, 2.0, 3.0, 3.8, 4.0, 4.2, 4.4, 4.6]
C2 = [0.05, 0.12, 0.2, 0.75, 1.0, 2.0]

#saiatu_vectorizer(vectorizer1, C1)
#saiatu_vectorizer(vectorizer2, C2)


def iragarpen (vectorizer, C):
    X_train_lr = vectorizer.fit_transform(trainX)
    X_dev_lr = vectorizer.transform(devX)

    logreg = linear_model.LogisticRegression(C=C, solver="liblinear", max_iter=5000, class_weight="balanced")
    logreg.fit(X_train_lr, Y_train_lr)
    pred_num = logreg.predict(X_dev_lr)
    pred = le.inverse_transform(pred_num)
    return pred
pred1 = iragarpen(vectorizer1, 4.0) # len = 1000
pred2 = iragarpen(vectorizer2, 0.12) # len = 1000

def iragarpenTest (vectorizer, C):
    X_train_lr = vectorizer.fit_transform(trainX)
    X_test_lr = vectorizer.transform(testX)

    logreg = linear_model.LogisticRegression(C=C, solver="liblinear", max_iter=5000, class_weight=None)
    logreg.fit(X_train_lr, Y_train_lr)
    pred_num = logreg.predict(X_test_lr)
    pred = le.inverse_transform(pred_num)
    return pred

pred1 = iragarpenTest(vectorizer1, 4.0)

pred_df1 = pd.DataFrame({"id": range(len(pred1)), "pred_label": pred1})
pred_df1.to_csv("pred.csv", index=False)
