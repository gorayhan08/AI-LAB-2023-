import torch
from torchvision import transforms
from Model import PCA
import numpy as np
from PIL import Image

def compute_distance_matrix(images):
    num_images = len(images)
    image_size = images[0].size[0] * images[0].size[1]
    # Convert images to tensors
    image_tensors = [transforms.ToTensor()(image).flatten().numpy() for image in images]

    # Perform PCA
    Model = PCA(n_components=min(num_images, image_size))
    Model.fit(np.vstack(image_tensors))
    transformed_features = Model.transform(np.vstack(image_tensors))

    # Compute the distance matrix
    distance_matrix = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(num_images):
            distance_matrix[i, j] = np.linalg.norm(transformed_features[i] - transformed_features[j])

    return distance_matrix

# Assuming you have a list of image paths
image_paths = ['/home/grayhan/Project/Project JU/Data/1.png', '/home/grayhan/Project/Project JU/Data/2.png', '/home/grayhan/Project/Project JU/Data/3.png', '/home/grayhan/Project/Project JU/Data/4.png']

# Load the images
images = [Image.open(path) for path in image_paths]

# Compute the distance matrix
distance_matrix = compute_distance_matrix(images)
print(distance_matrix)
np.savetxt('distance_matrix2.csv', distance_matrix, delimiter=',')