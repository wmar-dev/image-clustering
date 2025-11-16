#!/usr/bin/env python3
import sqlite3
import json
import argparse
from pathlib import Path

from tqdm.auto import tqdm
import torch
import clip
from PIL import Image

DB = "/tmp/embedding.db"


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


def main(image_path=None):
    print("Hello from image-clustering!")

    setup_db()

    compute_emeddings(image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image clustering using MediaPipe embeddings"
    )
    parser.add_argument(
        "--images", type=str, help="Path to image or directory of images to process"
    )
    args = parser.parse_args()

    main(image_path=args.images)
