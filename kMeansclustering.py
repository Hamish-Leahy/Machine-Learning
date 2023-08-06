def k_means(data, k, max_iterations=100, tolerance=1e-4):
    centroids = data[np.random.choice(len(data), k, replace=False)]
    for _ in range(max_iterations):
        assigned_clusters = np.argmin(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)
        new_centroids = np.array([data[assigned_clusters == i].mean(axis=0) for i in range(k)])
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break
        centroids = new_centroids
    return centroids, assigned_clusters
