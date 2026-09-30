import random
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter

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

def majority_class(trainY, devY):
    counts = Counter(devY)
    klase_nagusia = counts.most_common(1)[0][0]
    zehaztasuna = sum(1 for y in devY if klase_nagusia==y)
    zehaztasuna = zehaztasuna / len(devY)
    print("Klase nagusia: " + str(klase_nagusia))
    print("Zehaztasuna: " + str(zehaztasuna))

majority_class(trainY, devY)    

print(Counter(trainY))
print(Counter(devY))
print("Train: " + str(1586/len(trainY))) # 0.6344
print("Dev: " + str(634/len(devY))) # 0.634
# Train eta dev-eko etiketen %63,4 CRITICAL dira, sailkapena alboratua da. 