import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/interim/Titanic_understanding.csv")
df = df.drop(columns=["Unnamed: 0", "PassengerId"])


def random_sample_imputation(df):
    cols_with_missing_values = df.columns[df.isna().any()].tolist()

    for var in cols_with_missing_values:
        # extract a random sample
        random_sample_df = (
            df[var].dropna().sample(df[var].isnull().sum(), random_state=0)
        )
        # re-index the randomly extracted sample
        random_sample_df.index = df[df[var].isnull()].index

        # replace the NA
        df.loc[df[var].isnull(), var] = random_sample_df

    return df


df_clean = df.copy()
df_clean = random_sample_imputation(df_clean)

df_clean.info()

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

df_clean.reset_index(drop=True)

df_clean_column = ['Pclass', 'Age', 'Parch', 'Fare']

df_clean[df_clean_column + ['Survived']].boxplot(by ="Survived",figsize = (20,10), layout =(1,4))
plt.show()

df_clean = df_clean.to_pickle("../data/interim/Titanic_clean.pkl")