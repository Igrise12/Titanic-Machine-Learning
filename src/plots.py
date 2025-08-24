import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
import seaborn as sns

df = pd.read_pickle("../data/interim/Titanic_clean.pkl")


# aggregate dulu
counts = df.groupby(["Sex","Survived"]).size().reset_index(name="count")
pv = counts.pivot(index="Sex", columns="Survived", values="count").fillna(0)

# plot stacked
ax = pv.plot(kind="bar", stacked=True, figsize=(7,5))
ax.set_xlabel("Sex"); ax.set_ylabel("Count")
ax.set_title("Stacked Bar: Survived by Sex")
ax.legend(title="Survived")
plt.tight_layout(); plt.show()


