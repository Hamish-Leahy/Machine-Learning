def pca(X, n_components):
    covariance_matrix = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_indices = sorted_indices[:n_components]
    top_eigenvectors = eigenvectors[:, top_indices]
    X_pca = np.dot(X, top_eigenvectors)
    return X_pca
