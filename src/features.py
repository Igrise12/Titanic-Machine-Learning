import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_pickle("../data/interim/Titanic_clean.pkl")


le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
# 1 = Male 0 = Female

# 1 =Q  2 = S 0= C
df["Embarked"] = le.fit_transform(df["Embarked"])

df.to_pickle("../data/interim/Titanic_Ready.pkl")
