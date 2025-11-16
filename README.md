# Image Clustering

Automatically cluster similar images using OpenAI's CLIP model and cosine similarity matching.

## Features

- **Semantic Image Clustering**: Groups visually similar images using CLIP ViT-B/32 embeddings
- **Persistent Storage**: Embeddings cached in SQLite database for fast re-clustering
- **GPU Acceleration**: Automatically uses CUDA when available, falls back to CPU
- **Multiple Formats**: Supports jpg, webp, and png image formats
- **Configurable Similarity**: Adjustable cosine similarity threshold (default: 0.85)
- **Detailed Results**: Shows cluster sizes and lists images in multi-image clusters

## Installation

This project requires Python 3.12+ and uses `uv` for package management.

Install dependencies:
```bash
uv pip install git+https://github.com/openai/CLIP.git
```

Note: CLIP must be installed from GitHub as the PyPI version is outdated.

## Usage

Process and cluster images in a directory:
```bash
uv run main.py --images /path/to/images
```

Process images in current directory:
```bash
uv run main.py --images .
```

The tool will:
1. Generate CLIP embeddings for all images
2. Store embeddings in a SQLite database (cached for future runs)
3. Compute pairwise cosine similarity between all images
4. Cluster similar images using Union-Find algorithm
5. Display results showing cluster sizes and grouped image paths

## How It Works

The clustering pipeline:

1. **Embedding Generation**: Each image is processed through CLIP ViT-B/32 to generate a semantic embedding vector
2. **Similarity Computation**: Cosine similarity is calculated between all pairs of embeddings
3. **Clustering**: Images with similarity above the threshold (default 0.85) are merged into the same cluster using a Union-Find data structure
4. **Results**: Clusters are displayed sorted by size, showing only multi-image clusters

## Architecture

**main.py**: Image processing and clustering pipeline
- Database setup with SQLite in system temp directory
- CLIP embedding generation with error handling
- Cosine similarity computation
- Union-Find clustering with configurable threshold
- Results visualization

**union_find.py**: Disjoint Set Union data structure
- Path compression optimization for O(α(n)) find operations
- Union by rank for balanced tree structure
- Efficient clustering with minimal memory overhead

## Configuration

To adjust the similarity threshold, modify the value in [main.py:146](main.py#L146):
```python
clusters = cluster_embeddings(similarity_threshold=0.85)  # Adjust between 0.0 and 1.0
```

Higher values (closer to 1.0) = stricter matching, more clusters
Lower values (closer to 0.0) = looser matching, fewer clusters