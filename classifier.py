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
import spacy

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
nlp_lema = spacy.load("es_core_news_sm", disable = ["ner", "parser"])

# Lematizazioa gehitu
def lematizatu_corpusa (corpus):
    emaitzak = []
    for dok in nlp_lema.pipe(corpus, batch_size=64):
        lema = [token.lemma_.lower() for token in dok]
        emaitzak.append(" ".join(lema))
    return emaitzak

def kendu_stopwordak (token_zerrenda):
    return [t for t in token_zerrenda if t not in STOP_WORDS]

trainX = lematizatu_corpusa(trainX)
devX = lematizatu_corpusa(devX)
testX = lematizatu_corpusa(testX)

# CountVectorizer bat sortu erregresio logistikoa egin ahal izateko
le = preprocessing.LabelEncoder()
le.fit(trainY)
Y_train_lr = le.transform(trainY)
Y_dev_lr = le.transform(devY)

# Hiperparametroak doitu, kasu onena bilatu
def saiatu_vectorizer(vectorizer, C):

    X_train_lr = vectorizer.fit_transform(trainX)
    X_dev_lr = vectorizer.transform(devX)
    solvers = ["lbfgs", "liblinear", "saga"]
    onena = 0.0
    COnena = None
    solverOnena = None
    weightOnena = None
    for c in C:
        for solver in solvers:
            logreg = linear_model.LogisticRegression(C=c, solver=solver, max_iter=5000, class_weight=None)
            logreg2 = linear_model.LogisticRegression(C=c, solver=solver, max_iter=5000, class_weight="balanced")
            logreg.fit(X_train_lr, Y_train_lr)
            logreg2.fit(X_train_lr, Y_train_lr)
            pred_num1 = logreg.predict(X_dev_lr)
            lr_baseline = f1_score(Y_dev_lr, pred_num1, average="macro")
            #lr_baseline = logreg.score(X_dev_lr, Y_dev_lr)
            pred_num2 = logreg2.predict(X_dev_lr)
            lr_baseline2 = f1_score(Y_dev_lr, pred_num2, average="macro")
            #lr_baseline2 = logreg2.score(X_dev_lr, Y_dev_lr)

            if lr_baseline > lr_baseline2: 
                 unekoa = lr_baseline
                 weightUnekoa = "None"
            else:
                 unekoa = lr_baseline2
                 weightUnekoa = "Balanced"

            if onena < unekoa:
                COnena = c
                solverOnena = solver
                onena = unekoa
                weightOnena = weightUnekoa
    print(f"Max features: {vectorizer.max_features}, min_df: {vectorizer.min_df}, ngrams: {vectorizer.ngram_range},weight: {weightOnena}-ren zehastauna (metodoa: {solverOnena}) (c: {COnena}): " + str(onena))
    return onena, vectorizer.max_features, vectorizer.min_df, vectorizer.ngram_range, weightOnena, solverOnena, COnena
C1 = [0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 50.0]

all_features = [40000, 60000, 80000, 100000]
ngrams = [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10)]
min_dfs = [1, 2, 3, 5]
onenak = []

for features in all_features:
    for ngram in ngrams:
            for mins in min_dfs:
                vectorizer1 = TfidfVectorizer(max_features=features, lowercase=True, binary=False, ngram_range=ngram, stop_words=STOP_WORDS, min_df=mins, sublinear_tf=True)
                sol = saiatu_vectorizer(vectorizer1, C1)
                onenak.append(sol)
                onenak.sort(reverse=True)
                if len(onenak)>5:
                    onenak.pop()
for onena in onenak:
    print(onena)

def iragarpen (vectorizer, C):
    X_train_lr = vectorizer.fit_transform(trainX)
    X_dev_lr = vectorizer.transform(devX)

    logreg = linear_model.LogisticRegression(C=C, solver="lbfgs", max_iter=5000, class_weight="balanced")
    logreg.fit(X_train_lr, Y_train_lr)
    pred_num = logreg.predict(X_dev_lr)
    pred = le.inverse_transform(pred_num)
    return pred
vectorizerIragarpen = TfidfVectorizer(max_features=40000, lowercase=True, binary=False, ngram_range=(1,3), stop_words=STOP_WORDS, min_df=2, sublinear_tf=True)

def iragarpenTest (vectorizer, C):
    X_train_lr = vectorizer.fit_transform(trainX)
    X_test_lr = vectorizer.transform(testX)

    logreg = linear_model.LogisticRegression(C=C, solver="lbfgs", max_iter=5000, class_weight="balanced")
    logreg.fit(X_train_lr, Y_train_lr)
    pred_num = logreg.predict(X_test_lr)
    pred = le.inverse_transform(pred_num)
    return pred

#pred1 = iragarpenTest(vectorizerIragarpen, 10.0)

#pred_df1 = pd.DataFrame({"id": range(len(pred1)), "pred_label": pred1})
#pred_df1.to_csv("pred.csv", index=False)
