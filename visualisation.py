import numpy as np
import matplotlib.pyplot as plt

def plot_with_query_pca(vectors, query_vector, highlight_indices=None):
    from sklearn.decomposition import PCA

    all_embeddings = np.vstack([vectors, query_vector])

    pca = PCA(n_components=2)
    reduced_all = pca.fit_transform(all_embeddings)

    reduced_embeddings = reduced_all[:-1]
    query_tsne = reduced_all[-1]
    
    plt.figure(figsize=(8, 5))
    # Plot all points
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=10, label="Alle Datenpunkte")
    # Highlight found points if provided
    if highlight_indices is not None and len(highlight_indices) > 0:
        plt.scatter(reduced_embeddings[highlight_indices, 0], reduced_embeddings[highlight_indices, 1], 
                    s=80, c="orange", marker="o", edgecolor="black", linewidths=1.5, label="Gefundene Punkte")
    # Plot query vector
    plt.scatter(query_tsne[0], query_tsne[1], s=200, c="red", marker="*", edgecolor="black", linewidths=1.5, label="Query-Vektor")
    plt.title("Visualisierung der Embeddings mit PCA")
    plt.legend()
    plt.show()

def plot_with_query_umap(vectors, query_vector, highlight_indices=None):
    from umap import UMAP

    all_embeddings = np.vstack([vectors, query_vector])

    umap = UMAP(n_components=2, init="random", random_state=0)
    reduced_all = umap.fit_transform(all_embeddings)

    reduced_embeddings = reduced_all[:-1]
    query_tsne = reduced_all[-1]
    
    plt.figure(figsize=(8, 5))
    # Plot all points
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=10, label="Alle Datenpunkte")
    # Highlight found points if provided
    if highlight_indices is not None and len(highlight_indices) > 0:
        plt.scatter(reduced_embeddings[highlight_indices, 0], reduced_embeddings[highlight_indices, 1], 
                    s=80, c="orange", marker="o", edgecolor="black", linewidths=1.5, label="Gefundene Punkte")
    # Plot query vector
    plt.scatter(query_tsne[0], query_tsne[1], s=200, c="red", marker="*", edgecolor="black", linewidths=1.5, label="Query-Vektor")
    plt.title("Visualisierung der Embeddings mit UMAP")
    plt.legend()
    plt.show()
