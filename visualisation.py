import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List

def plot_with_query_pca(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    highlight_indices: Optional[List[int]] = None,
    hover_texts: Optional[List[str]] = None
) -> None:
    """Visualisiert Embeddings und Query-Vektor mit PCA."""
    from sklearn.decomposition import PCA
    all_embeddings = np.vstack([vectors, query_vector])
    pca = PCA(n_components=2)
    reduced_all = pca.fit_transform(all_embeddings)
    reduced_embeddings = reduced_all[:-1]
    query_2d = reduced_all[-1]
    if hover_texts is None:
        hover_texts = [str(i) for i in range(len(reduced_embeddings))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1],
        mode='markers',
        marker=dict(size=10, color='blue'),
        text=hover_texts,
        name='Alle Datenpunkte',
        hoverinfo='text',
        showlegend=True
    ))
    if highlight_indices is not None and len(highlight_indices) > 0:
        fig.add_trace(go.Scatter(
            x=reduced_embeddings[highlight_indices, 0], y=reduced_embeddings[highlight_indices, 1],
            mode='markers',
            marker=dict(size=16, color='orange', line=dict(width=2, color='black')),
            text=[hover_texts[i] for i in highlight_indices],
            name='Gefundene Punkte',
            hoverinfo='text',
            showlegend=True
        ))
    fig.add_trace(go.Scatter(
        x=[query_2d[0]], y=[query_2d[1]],
        mode='markers',
        marker=dict(size=24, color='red', symbol='star', line=dict(width=2, color='black')),
        text=['Query-Vektor'],
        name='Query-Vektor',
        hoverinfo='text',
        showlegend=True
    ))
    fig.update_layout(title='Visualisierung der Embeddings mit PCA', width=800, height=500)
    fig.show()

def plot_with_query_umap(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    highlight_indices: Optional[List[int]] = None,
    hover_texts: Optional[List[str]] = None
) -> None:
    """Visualisiert Embeddings und Query-Vektor mit UMAP."""
    from umap import UMAP
    all_embeddings = np.vstack([vectors, query_vector])
    umap = UMAP(n_components=2, init="random", random_state=0)
    reduced_all = umap.fit_transform(all_embeddings)
    reduced_embeddings = reduced_all[:-1]
    query_2d = reduced_all[-1]
    if hover_texts is None:
        hover_texts = [str(i) for i in range(len(reduced_embeddings))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1],
        mode='markers',
        marker=dict(size=10, color='blue'),
        text=hover_texts,
        name='Alle Datenpunkte',
        hoverinfo='text',
        showlegend=True
    ))
    if highlight_indices is not None and len(highlight_indices) > 0:
        fig.add_trace(go.Scatter(
            x=reduced_embeddings[highlight_indices, 0], y=reduced_embeddings[highlight_indices, 1],
            mode='markers',
            marker=dict(size=16, color='orange', line=dict(width=2, color='black')),
            text=[hover_texts[i] for i in highlight_indices],
            name='Gefundene Punkte',
            hoverinfo='text',
            showlegend=True
        ))
    fig.add_trace(go.Scatter(
        x=[query_2d[0]], y=[query_2d[1]],
        mode='markers',
        marker=dict(size=24, color='red', symbol='star', line=dict(width=2, color='black')),
        text=['Query-Vektor'],
        name='Query-Vektor',
        hoverinfo='text',
        showlegend=True
    ))
    fig.update_layout(title='Visualisierung der Embeddings mit UMAP', width=800, height=500)
    fig.show()

def plot_with_query(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    highlight_indices: Optional[List[int]] = None,
    hover_texts: Optional[List[str]] = None,
    method: str = 'umap'
) -> None:
    """Visualisiert Embeddings und Query-Vektor mit wählbarer Methode ('umap' oder 'pca')."""
    try:
        if method == 'umap':
            plot_with_query_umap(vectors, query_vector, highlight_indices, hover_texts)
        else:
            plot_with_query_pca(vectors, query_vector, highlight_indices, hover_texts)
    except ImportError:
        plot_with_query_pca(vectors, query_vector, highlight_indices, hover_texts)
