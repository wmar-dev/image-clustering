# Image Clustering

Image clustering tool using OpenAI's CLIP model to generate embeddings for images.

## Features

- Generate image embeddings using CLIP ViT-B/32 model
- Store embeddings in SQLite database for efficient retrieval
- Support for jpg, webp, and png image formats
- GPU acceleration when available (CUDA)

## Installation

This project requires Python 3.12+ and uses `uv` for package management.

Install dependencies:
```bash
uv pip install git+https://github.com/openai/CLIP.git
```

## Usage

Process images in a directory:
```bash
uv run main.py --images /path/to/images
```

Process images in current directory:
```bash
uv run main.py --images .
```

## Architecture

The project consists of two main components:

- **main.py**: Processes images and generates CLIP embeddings, storing them in SQLite
- **union_find.py**: Union-Find data structure for efficient clustering operations

Embeddings are stored in `/tmp/embedding.db` with the image path as the key.