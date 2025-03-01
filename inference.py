import os
import sys
import cv2
import numpy as np
import numpy.typing as npt

from tensorflow.keras.models import load_model

def create_mapping() -> dict[int, int]:
    mapping_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'training files',
        'mapping.txt')
    mapping = {}
    with open(mapping_file, 'r') as file:
        lines = file.read().strip().split('\n')
        for line in lines:
            key, value = map(int, line.split())
            mapping[key] = value
    return mapping

def predict(images: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'model.keras')
    model = load_model(model_path)
    probabilities = model.predict(images, verbose=0)
    predictions: npt.NDArray[np.int_] = np.argmax(probabilities, axis=1)
    return predictions

def load_images(directory_name: str) -> tuple[npt.NDArray[np.float64], list[str]]:
    images = []
    file_paths = []
    for root, _, files in os.walk(directory_name):
        for file in files:
            file_path = os.path.join(root, file)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image.resize((28, 28))
            images.append(image)
            file_paths.append(file_path)
    if len(images) == 0:
        raise FileNotFoundError(f'No images were found in directory {directory_name}')
    images = np.array(images, dtype=np.float64) / 255
    return images, file_paths

def get_directory() -> str:
    directory_name = sys.argv[1]
    if not os.path.exists(directory_name):
        raise FileNotFoundError(f'Directory {directory_name} does not exist')
    if not os.path.isdir(directory_name):
        raise NotADirectoryError(f'{directory_name} is not a directory')
    return directory_name

def main() -> None:
    directory = get_directory()
    images, file_paths = load_images(directory)
    predictions = predict(images)
    mapping = create_mapping()
    for prediction, file_path in zip(predictions, file_paths):
        print(f'{mapping[prediction]}, {file_path}')

if __name__ == '__main__':
    main()
