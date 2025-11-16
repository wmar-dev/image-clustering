# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Image clustering tool using OpenAI's CLIP model to generate embeddings for images and cluster similar images together. The project uses a Union-Find data structure for efficient clustering operations with cosine similarity matching.

## Development Setup

**Python Version**: 3.12+ (managed via `.python-version`)

**Package Manager**: uv

**Install Dependencies**:
```bash
uv pip install git+https://github.com/openai/CLIP.git
```

Note: CLIP must be installed from GitHub, not PyPI. Additional dependencies (pillow, torch, tqdm) are managed via `pyproject.toml`.

**Run the Application**:
```bash
uv run main.py --images <path_to_images>
```

**Optional Arguments**:
- `--threshold`: Cosine similarity threshold for clustering (default: 0.85, range: 0.0-1.0)

## Architecture

### Core Components

**main.py**: Entry point for image processing and clustering pipeline
- Sets up SQLite database in system temp directory for storing image embeddings
- Processes images using CLIP ViT-B/32 model with GPU acceleration when available
- Supports jpg, webp, and png formats (defined in `IMAGE_EXTENSIONS` global)
- Database schema: `embedding(content TEXT PRIMARY KEY, embedding_json BLOB)`
- Clusters images using cosine similarity (configurable via `--threshold` arg, default: 0.85)
- Prints detailed cluster results showing grouped similar images
- Note: Contains typo in function name `compute_emeddings` (should be `compute_embeddings`)

**union_find.py**: Union-Find (Disjoint Set Union) data structure
- Implements path compression and union by rank optimizations
- Tracks set sizes and can count distinct sets
- Used for efficient clustering of similar image embeddings

### Data Flow

1. Images are discovered from specified directory using glob patterns
2. CLIP model generates embeddings for each image (with error handling for failed images)
3. Embeddings are stored as JSON in SQLite database with absolute image path as key
4. Cosine similarity is computed between all pairs of embeddings
5. Union-Find structure merges images into clusters based on similarity threshold
6. Results are printed showing cluster sizes and image paths for multi-image clusters

### Key Implementation Details

- **Database location**: Uses `tempfile.gettempdir()` for cross-platform compatibility
- **Similarity metric**: Cosine similarity with configurable threshold (default 0.85)
- **Clustering algorithm**: Union-Find with O(n²) pairwise comparison
- **Device support**: Automatically uses CUDA if available, falls back to CPU
- **Error handling**: Failed images are logged as warnings and skipped

### Key Files

- **main.py**: Image processing and clustering entry point
- **union_find.py**: Disjoint set data structure for clustering
- **pyproject.toml**: Project dependencies and metadata
