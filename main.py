import urllib

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
    print("Hello from image-clustering!")
    
    IMAGE_FILENAMES = ['burger.jpg', 'burger_crop.jpg']

    for name in IMAGE_FILENAMES:
        url = f'https://storage.googleapis.com/mediapipe-assets/{name}'
        urllib.request.urlretrieve(url, name)

    images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
    for name, image in images.items():
        print(name)

    # Create options for Image Embedder
    base_options = python.BaseOptions(model_asset_path='embedder.tflite')
    l2_normalize = True #@param {type:"boolean"}
    quantize = True #@param {type:"boolean"}
    options = vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=l2_normalize, quantize=quantize)


    # Create Image Embedder
    with vision.ImageEmbedder.create_from_options(options) as embedder:

        # Format images for MediaPipe
        first_image = mp.Image.create_from_file(IMAGE_FILENAMES[0])
        second_image = mp.Image.create_from_file(IMAGE_FILENAMES[1])
        first_embedding_result = embedder.embed(first_image)
        second_embedding_result = embedder.embed(second_image)

        # Calculate and print similarity
        similarity = vision.ImageEmbedder.cosine_similarity(
            first_embedding_result.embeddings[0],
            second_embedding_result.embeddings[0])
        print(similarity)        


if __name__ == "__main__":
    main()
