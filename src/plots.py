import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

df = pd.read_csv("../data/raw/train.csv")


df.info()


# plot single column
counts = df.groupby("Sex")["Survived"].sum()
plt.bar(counts.index, counts.values)
plt.show()


df["Parch"].unique()