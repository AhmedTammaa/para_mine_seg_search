from sklearn.metrics import silhouette_score


def evaluate_clusters(X, clusters):

    silhouette_avg = silhouette_score(X, clusters)
    return silhouette_avg
