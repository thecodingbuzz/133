from sklearn.cluster import KMeans
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


df=pd.read_csv('final.csv')

star_masses=df['mass'].to_list()
star_radiuses=df['radius'].to_list()
gravity=df['gravity'].to_list()

X=[]
for ind,star_mass in enumerate(star_masses):
    temp_list=[
        star_radiuses[ind],star_mass
    ]
    X.append(temp_list)

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sb.lineplot(range(1,11),wcss,marker='o',color='red')
plt.title('KMeans Elbow Method')
plt.xlabel("no.of clusters: ")
plt.ylabel("WCSS")
plt.show()

fig = px.scatter(x=star_masses, y=star_radiuses)
fig.show()
