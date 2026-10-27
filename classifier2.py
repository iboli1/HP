import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import f1_score

RANDOM_SEED = 67
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# Entrenamenduko datuak irakurri
train_df = pd.read_csv("train.csv")
trainX = train_df["text"]
trainY = train_df["category"]

dev_df = pd.read_csv("dev.csv")
devX = dev_df["text"]
devY = dev_df["category"]

test_df = pd.read_csv("test_clean.csv")
testX = test_df["text"]

le = preprocessing.LabelEncoder()
le.fit(trainY)
Y_train_lr = le.transform(trainY)
Y_dev_lr = le.transform(devY)