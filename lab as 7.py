import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


data = pd.read_csv('USA_Housing.csv')
data = data.iloc[:, :-1]  # Dropping the last column
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

#
pca = PCA(n_components=2)
data_scaled = pca.fit_transform(data_scaled)

K_range = range(2, 11)  # Start from 2 to avoid undefined silhouette score at K=1

# Initialize lists to store the scores for each K
inertia_scores = []
silhouette_scores = []
davies_bouldin_scores = []

# Calculate metrics for each K
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)

    # Inertia
    inertia_scores.append(kmeans.inertia_)

    # Silhouette Score
    silhouette_avg = silhouette_score(data_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

    # Davies-Bouldin Index
    davies_bouldin_avg = davies_bouldin_score(data_scaled, kmeans.labels_)
    davies_bouldin_scores.append(davies_bouldin_avg)

# Plot all metrics for each K
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(K_range, inertia_scores, 'bo-', label='Inertia')
ax1.set_xlabel('Number of clusters (K)')
ax1.set_ylabel('Inertia', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(K_range, silhouette_scores, 'go-', label='Silhouette Score')
ax2.plot(K_range, davies_bouldin_scores, 'ro-', label='Davies-Bouldin Index')
ax2.set_ylabel('Score', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adding legends and title
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.85))
plt.title('Clustering Scores for Different Values of K')
plt.savefig('Clustering Scores for Different Values of K.png')

k = 2
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(data_scaled)

data['Cluster'] = kmeans.labels_

# Visualize clustering if data is 2D (adjust for more dimensions as needed)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=data['Cluster'], cmap='viridis')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig('Clusters1.png')

# part 2 k mediods

from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

data = pd.read_csv('USA_Housing.csv')
data = data.iloc[:, :-1]  # Drop the last attribute

print(data.head())
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

#kmedoids = KMedoids(n_clusters=k, random_state=42)
#kmedoids.fit(data_scaled)

#data['Cluster'] = kmedoids.labels_
#print(data.head())
#print("Medoids:")
#print(kmedoids.cluster_centers_)

data_array = np.array(data_scaled)


# K-Medoids function
def k_medoids(data, K, max_iter=100):
    # Randomly select K unique points as initial medoids
    m, n = data.shape
    medoids = data[np.random.choice(m, K, replace=False)]

    for iteration in range(max_iter):
        # Step 1: Assign each point to the nearest medoid
        distances = pairwise_distances(data, medoids, metric='euclidean')
        labels = np.argmin(distances, axis=1)

        # Step 2: Update medoids for each cluster
        new_medoids = np.copy(medoids)
        for k in range(K):
            cluster_points = data[labels == k]
            if len(cluster_points) > 0:
                # Calculate the total distance to other points in the cluster
                intra_cluster_distances = pairwise_distances(cluster_points, cluster_points)
                total_distances = np.sum(intra_cluster_distances, axis=1)
                # Select the point with the minimum total distance as the new medoid
                new_medoids[k] = cluster_points[np.argmin(total_distances)]

        # Check for convergence (if medoids don't change, we're done)
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    return labels, medoids


# Define the number of clusters
optimal_k = 2  # Example, replace with your chosen K

# Run K-Medoids
labels, medoids = k_medoids(data_array, optimal_k)

# Assign labels to the DataFrame for analysis
data['Cluster'] = labels

# Display the resulting clusters
print(data.head())
print("Medoids:\n", medoids)
