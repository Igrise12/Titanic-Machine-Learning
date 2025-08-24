import pandas as pd 


# read dataset
df = pd.read_csv("../data/raw/train.csv")

# read data info
df.info()

# cek nilai null
df.isnull().sum()

# cek description
df.describe()

# drop column name 
df = df.drop(columns= "Name" )
df = df.drop(columns= "Cabin")
df = df.drop(columns = "Ticket")

df["Age"].describe()
df["Age"].isnull().sum()
df["Age"].value_counts()
df.isna().sum()

df.groupby("Pclass")["Survived"].sum()

df.duplicated().sum()


df = df.to_csv("../data/interim/Titanic_understanding.csv")


