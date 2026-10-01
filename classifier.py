import random
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re

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
'''
def majority_class(trainY, devY):
    counts = Counter(devY)
    klase_nagusia = counts.most_common(1)[0][0]
    zehaztasuna = sum(1 for y in devY if klase_nagusia==y)
    zehaztasuna = zehaztasuna / len(devY)
    print("Klase nagusia: " + str(klase_nagusia))
    print("Zehaztasuna: " + str(zehaztasuna))
'''
class SimpleTokenizer: 
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {id:token for token, id in vocab.items()}

    def encode(self, text):
        preproccessed = re.split(r'[,.;:!?/()"”]|\s', text)
        preproccessed = [item.strip() for item in preproccessed if item.strip()]
        preproccessed = [item if item in self.str_to_int else "<|unk|>" for item in preproccessed]
        ids = [self.str_to_int[s] for s in preproccessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+[,.;:!?/()"”]', r'\1', text)
        return text

text = " ".join(trainX.tolist()).lower() # Hitz guztiak bektore luze bakar baten batu nahi ditugu gero split bat egin ahal izateko. trainX lista batera bihurtu behar da hori egiteko
preproccessed = re.split(r'[,.;:!?/()"”]|\s', text)
preproccessed = [item.strip() for item in preproccessed if item.strip()]
all_tokens = sorted(set(preproccessed))
all_tokens.extend(['<|endoftext|>', '<|unk|>'])
vocab = {integer:token for integer, token in enumerate(all_tokens)}

klaseak = trainY.unique() # Bi klaseak bektore baten gorde
bow_klaseka = {}
X = 5 # Gutxienez 5 aldiz ateratzen diren hitzak gorde, besteak kendu
for k in klaseak:
    klaseko_text = train_df[train_df["category"] == k]["text"]
    text_j = " ".join(klaseko_text.tolist()).lower()
    tokens = re.split(r'[,.;:!?/()"”]|\s', text_j)
    tokens = [item.strip() for item in tokens if item.strip()]
    tokens = Counter(tokens)
    bow_klaseka[k] = Counter({token: freq for token, freq in tokens.items() if freq>=X}).most_common(20)


print(bow_klaseka["CRITICAL"])
print(bow_klaseka["CONSPIRACY"])

'''
majority_class(trainY, devY)
print(Counter(trainY))
print(Counter(devY))
print("Train: " + str(1586/len(trainY))) # 0.6344
print("Dev: " + str(634/len(devY))) # 0.634
# Train eta dev-eko etiketen %63,4 CRITICAL dira, sailkapena alboratua da. 
'''
