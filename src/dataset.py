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

df["Age"].describe()
df["Age"].isnull().sum()

df.groupby(pd.cut(df["Age"], bins=[0,12,18,40,60,100]))["Survived"].sum()


df["Age"].value_counts()
df.isna().sum()






df_survived  = pd.DataFrame()

# cek survived

df_survived = df[df["Survived"] == 1]

df_survived.groupby("Sex")["Survived"].sum()
df_survived.groupby("Parch")["Survived"].sum()
df_survived.groupby("Embarked")["Survived"].sum()
df_survived.groupby("Pclass")["Survived"].sum()



df_survived.groupby("Age")["Survived"].sum()
df_survived["Age"].hist(bins=15)


