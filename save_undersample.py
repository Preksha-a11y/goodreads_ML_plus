import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# read/prep data
dat = pd.read_csv("data/tokenized_reviews.csv")
dat = dat.dropna()
dat["quote"] = dat["quote"].astype(int)
if "cleaned_tokens" in dat.columns:
    dat["tokenized_words"] = dat["cleaned_tokens"]
elif "tokenized_words" in dat.columns:
    dat["tokenized_words"] = dat["tokenized_words"].apply(lambda x: str(x).strip("[']").replace("', '"," "))


# 85% train / 15% test
X_train, X_test, y_train, y_test = train_test_split(dat.drop(columns=["popular"]), 
                                                    dat["popular"],
                                                    test_size = 0.15,
                                                    random_state = 229)

# undersample train set to exact 50:50 balance
n_0 = len(y_train[y_train == 0])
n_1 = len(y_train[y_train == 1])
rng = np.random.default_rng(seed=229)

if n_0 > n_1:
    majority_indices = y_train[y_train == 0].index
    drop_indices = rng.choice(majority_indices, n_0 - n_1, replace=False)
    X_train = X_train.drop(drop_indices)
    y_train = y_train.drop(drop_indices)
elif n_1 > n_0:
    majority_indices = y_train[y_train == 1].index
    drop_indices = rng.choice(majority_indices, n_1 - n_0, replace=False)
    X_train = X_train.drop(drop_indices)
    y_train = y_train.drop(drop_indices)



X_train.to_pickle("data/X_train.pkl",protocol=3)
X_test.to_pickle("data/X_test.pkl",protocol=3)
y_train.to_pickle("data/y_train.pkl",protocol=3)
y_test.to_pickle("data/y_test.pkl",protocol=3)
