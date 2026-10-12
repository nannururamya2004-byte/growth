
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "year": [2015, 2016, 2017, 2018, 2019,
             2020, 2021, 2022, 2023, 2024],
    "growth": [10, 12, 15, 18, 20, 25, 30, 35, 40, 45],
    "revenue":[None]*10,
    "employees":  [None] * 10,
    "investment": [None] * 10

}
df=pd.read_csv("growth_data.csv")
df=pd.DataFrame(data)
print(df)
print("\nNull values:")
print(df.isnull().sum())



x=df[['year','growth']]

train =KMeans(df)

model = KMeans(n_clusters=7,random_state=42,n_init=10)
model.fit(x)
df['cluster'] = model.labels_
print(df)
plt.scatter(
    df["year"],
    df["growth"],
    c=df["cluster"],
    cmap="tab10",
    s=120
)
plt.xlabel("Year")
plt.ylabel("Growth")
plt.title("K-Means Clustering of Growth Data")
plt.show()
