import torch
import torch.nn as nn
from torchvision import transforms
from Model import PCA
import numpy as np
from PIL import Image
from sklearn.random_projection import SparseRandomProjection
image1 = Image.open('/home/grayhan/Project/Project JU/Data/1.png')
image2 = Image.open('/home/grayhan/Project/Project JU/Data/2.png')

def compute_distance_matrix(image1, image2):
    # Convert images to tensors
    image_tensor1 = transforms.ToTensor()(image1).flatten().numpy()
    image_tensor2 = transforms.ToTensor()(image2).flatten().numpy()

    rp = SparseRandomProjection(n_components=2)
    image_tensor1 = rp.fit_transform(image_tensor1.reshape(1, -1))
    image_tensor2 = rp.transform(image_tensor2.reshape(1, -1))

    Model = PCA(n_components=min(image_tensor1.shape[1], image_tensor2.shape[1]))
    Model.fit(np.vstack((image_tensor1, image_tensor2)))
    transformed_features = Model.transform(np.vstack((image_tensor1, image_tensor2)))
    distance_matrix = np.linalg.norm(transformed_features[0] - transformed_features[1])

    return distance_matrix

distance_matrix = compute_distance_matrix(image1, image2)
print(distance_matrix)
