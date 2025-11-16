# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Image clustering tool using OpenAI's CLIP model to generate embeddings for images and cluster similar images together. The project uses a Union-Find data structure for efficient clustering operations.

## Development Setup

**Python Version**: 3.12 (managed via `.python-version`)

**Package Manager**: uv

**Install Dependencies**:
```bash
uv pip install git+https://github.com/openai/CLIP.git
```

Note: CLIP must be installed from GitHub, not PyPI. Additional dependencies (mediapipe, opencv-python, pillow, torch, tqdm) are managed via `pyproject.toml`.

**Run the Application**:
```bash
uv run main.py --images <path_to_images>
```

## Architecture

### Core Components

**main.py**: Entry point for image processing pipeline
- Sets up SQLite database (`/tmp/embedding.db`) for storing image embeddings
- Processes images using CLIP ViT-B/32 model
- Supports jpg, webp, and png formats
- Database schema: `embedding(content TEXT PRIMARY KEY, embedding_json BLOB)`
- Note: Contains typo in function name `compute_emeddings` (should be `compute_embeddings`)

**union_find.py**: Union-Find (Disjoint Set Union) data structure
- Implements path compression and union by rank optimizations
- Tracks set sizes and can count distinct sets
- Designed for clustering operations (likely to group similar image embeddings)

### Data Flow

1. Images are processed from a specified directory
2. CLIP model generates embeddings for each image
3. Embeddings are stored as JSON in SQLite database with image path as key
4. Union-Find structure can be used to cluster images based on embedding similarity

### Key Files

- Database location: `/tmp/embedding.db` (temporary, will be lost on reboot)
