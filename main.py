#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import logging
import sqlite3
import tempfile

from PIL import Image
from tqdm.auto import tqdm
import clip
import numpy as np
import torch

from union_find import UnionFind

# Create database in system temp directory
DB = Path(tempfile.gettempdir()) / "embedding.db"


def setup_db():
    with sqlite3.connect(DB) as conn:
        sql = """
        CREATE TABLE IF NOT EXISTS embedding (
            content TEXT PRIMARY KEY,
            embedding_json BLOB
        )
        """
        conn.execute(sql)


def compute_emeddings(image_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for glob in [Path(image_path).glob(f"*.{ext}") for ext in ["jpg", "webp", "png"]]:
        for p in tqdm(glob):
            try:
                image = preprocess(Image.open(p)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                with sqlite3.connect(DB) as conn:
                    sql = "REPLACE INTO embedding(content, embedding_json) VALUES(?, ?)"
                    conn.execute(
                        sql, (str(p.absolute()), json.dumps(image_features.tolist()))
                    )
            except Exception as ex:
                logging.warning(ex)
                logging.warning(f"Error computing embedding for: {p}")


def read_embeddings():
    """Read all embeddings from the database."""
    embeddings = []
    image_paths = []

    with sqlite3.connect(DB) as conn:
        cursor = conn.execute("SELECT content, embedding_json FROM embedding")
        for row in cursor:
            image_path, embedding_json = row
            embedding = json.loads(embedding_json)
            image_paths.append(image_path)
            embeddings.append(np.array(embedding))

    return image_paths, embeddings


def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    # Flatten embeddings if they are multidimensional
    emb1 = np.array(embedding1).flatten()
    emb2 = np.array(embedding2).flatten()

    # Compute cosine similarity
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def cluster_embeddings(similarity_threshold=0.85):
    """
    Cluster images based on embedding similarity using Union-Find.

    Args:
        similarity_threshold: Cosine similarity threshold for clustering (default: 0.85)

    Returns:
        dict: Mapping of cluster_id to list of image paths
    """
    image_paths, embeddings = read_embeddings()

    if len(embeddings) == 0:
        print("No embeddings found in database")
        return {}

    print(
        f"Clustering {len(embeddings)} images with similarity threshold {similarity_threshold}"
    )

    # Initialize Union-Find structure
    uf = UnionFind(len(embeddings))

    # Compare all pairs and union similar images
    for i in tqdm(range(len(embeddings)), desc="Computing similarities"):
        for j in range(i + 1, len(embeddings)):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            if similarity >= similarity_threshold:
                uf.union(i, j)

    # Group images by cluster
    clusters = {}
    for i in range(len(embeddings)):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(image_paths[i])

    # Print cluster summary
    print(f"\nFound {len(clusters)} clusters:")
    for cluster_id, images in sorted(
        clusters.items(), key=lambda x: len(x[1]), reverse=True
    ):
        print(f"  Cluster {cluster_id}: {len(images)} images")
        if len(images) > 1:
            for img_path in images:
                print(f"    - {img_path}")

    return clusters


def main(image_path=None):
    print("Hello from image-clustering!")

    setup_db()

    compute_emeddings(image_path)

    # Cluster the embeddings
    clusters = cluster_embeddings(similarity_threshold=0.85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image clustering using CLIP embeddings"
    )
    parser.add_argument(
        "--images", type=str, help="Path to image or directory of images to process"
    )
    args = parser.parse_args()

    main(image_path=args.images)
