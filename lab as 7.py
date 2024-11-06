import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


data = pd.read_csv("USA_Housing.csv")
print(data.head())

data = data.iloc[:, :-1]

scaler = StandardScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)

#
k=4

kmeans = KMeans(n_clusters=k,random_state=42)
kmeans.fit(scaled_data)
clusters = kmeans.labels_

data['Cluster'] = clusters

plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()







