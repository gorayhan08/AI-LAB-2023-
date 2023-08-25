import torch
from torchvision import transforms
from Model import PCA
import numpy as np
import os
from PIL import Image

def compute_distance_matrix(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            image_paths.append(os.path.join(root, file))

    num_images = len(image_paths)

    # Convert images to tensors
    image_tensors = []
    for path in image_paths:
        image = transforms.ToTensor()(Image.open(path)).flatten().numpy()
        image_tensors.append(image)

    # Perform PCA
    pca = PCA(n_components=min(num_images, image_tensors[0].shape[0]))
    pca.fit(np.vstack(image_tensors))
    transformed_features = pca.transform(np.vstack(image_tensors))
    # Compute the distance matrix
    distance_matrix = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(num_images):
            distance_matrix[i, j] = np.linalg.norm(transformed_features[i] - transformed_features[j])

    return distance_matrix




# Provide the path to the folder containing the images
folder_path = '/home/grayhan/Project/Project JU/Data'

# Compute the distance matrix
distance_matrix = compute_distance_matrix(folder_path)
print(distance_matrix)
np.savetxt('distance_matrix.csv', distance_matrix, delimiter=',')