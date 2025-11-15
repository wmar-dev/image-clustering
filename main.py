import argparse
import urllib
from pathlib import Path

from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main(image_path=None):
    print("Hello from image-clustering!")

    # Create options for Image Embedder
    base_options = python.BaseOptions(model_asset_path='embedder.tflite')
    l2_normalize = True #@param {type:"boolean"}
    quantize = True #@param {type:"boolean"}
    options = vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=l2_normalize, quantize=quantize)


    # Create Image Embedder
    embedding_results = {}
    with vision.ImageEmbedder.create_from_options(options) as embedder:

        for p in tqdm(Path(image_path).glob('*.webp')):
            # Format images for MediaPipe
            #image = mp.Image.create_from_file(str(p))
            try:
                pil_img = Image.open(p)
                image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))
                result = embedder.embed(image)
                embedding_results[p.name] = result
            except:
                print(p)
    print(embedding_results)
        # # Calculate and print similarity
        # similarity = vision.ImageEmbedder.cosine_similarity(
        #     first_embedding_result.embeddings[0],
        #     second_embedding_result.embeddings[0])
        # print(similarity)        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image clustering using MediaPipe embeddings")
    parser.add_argument("--images", type=str, help="Path to image or directory of images to process")
    args = parser.parse_args()

    main(image_path=args.images)
