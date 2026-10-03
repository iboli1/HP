import random
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, linear_model
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

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

vectorizer1 = TfidfVectorizer(max_features=15000, lowercase=True, ngram_range=(1, 2), stop_words=STOP_WORDS, min_df=3,sublinear_tf=True)
vectorizer2 = CountVectorizer(max_features=15000, lowercase=True, ngram_range=(1, 2), stop_words=STOP_WORDS, min_df=3,binary=True)

def saiatu_vectorizer(vectorizer):

    X_train_lr = vectorizer.fit_transform(trainX)
    X_dev_lr = vectorizer.transform(devX)

    C = [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]
    solvers = ["lbfgs", "liblinear", "saga"]
    none_weights = []
    balanced_weights = []

    for c in C:
        for solver in solvers:
            logreg = linear_model.LogisticRegression(C=c, solver=solver, max_iter=5000, class_weight=None)
            logreg2 = linear_model.LogisticRegression(C=c, solver=solver, max_iter=5000, class_weight="balanced")
            logreg.fit(X_train_lr, Y_train_lr)
            logreg2.fit(X_train_lr, Y_train_lr)
            lr_baseline = logreg.score(X_dev_lr, Y_dev_lr)
            lr_baseline2 = logreg2.score(X_dev_lr, Y_dev_lr)
            none_weights.append(lr_baseline)
            balanced_weights.append(lr_baseline2)
            print(f"None-ren zehaztasuna (metodoa: {solver}) (c: {c}): " + str(lr_baseline))
            print(f"Balanced-en zehaztasuna (metodoa: {solver}) (c: {c}): " + str(lr_baseline2))

saiatu_vectorizer(vectorizer2)
# Hiperparametroak doitu, kasu onena bilatu



# Kasu onena: liblinear_none[0.12]