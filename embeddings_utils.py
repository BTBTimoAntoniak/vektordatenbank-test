import os
import numpy as np
from typing import Callable, List

EMBEDDINGS_PATH = "embeddings.npy"

def load_embeddings() -> np.ndarray | None:
    """Lädt die Embeddings aus einer Datei, falls vorhanden."""
    if os.path.exists(EMBEDDINGS_PATH):
        return np.load(EMBEDDINGS_PATH)
    return None

def save_embeddings(embeddings: np.ndarray) -> None:
    """Speichert die Embeddings in einer Datei."""
    np.save(EMBEDDINGS_PATH, embeddings)

def get_or_create_embeddings(workers: List, embedd_func: Callable[[str], list[float]]) -> np.ndarray:
    """
    Lädt Embeddings, falls vorhanden, oder berechnet und speichert sie.
    workers: Liste von MitarbeiterSkills
    embedd_func: Funktion, die ein Text-Embedding erzeugt
    """
    embeddings = load_embeddings()
    if embeddings is not None:
        print("Lade gespeicherte Embeddings...")
        return embeddings
    print("Berechne Embeddings und speichere sie...")
    embedded_data = [embedd_func(worker.to_embedding()) for worker in workers]
    embedded_data = np.array(embedded_data)
    save_embeddings(embedded_data)
    return embedded_data

def get_query_embedding(query: str, embedd_func: Callable[[str], list[float]]) -> np.ndarray:
    """
    Berechnet das Embedding für eine Query, ohne es zu speichern.
    query: Der Suchstring
    embedd_func: Funktion, die ein Text-Embedding erzeugt
    """
    return np.array(embedd_func(query))
