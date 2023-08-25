import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.std = None
        self.eigenvectors = None

    def normalize_data(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        normalized_data = (data - self.mean) / self.std
        return normalized_data

    def compute_covariance_matrix(self, data):
        num_samples = data.shape[0]
        covariance_matrix = np.dot(data.T, data) / (num_samples - 1)
        return covariance_matrix

    def fit(self, data):
        normalized_data = self.normalize_data(data)
        covariance_matrix = self.compute_covariance_matrix(normalized_data)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        top_k_indices = sorted_indices[:self.n_components]
        self.eigenvectors = eigenvectors[:, top_k_indices]

    def transform(self, data):
        normalized_data = self.normalize_data(data)
        transformed_data = np.dot(normalized_data, self.eigenvectors)
        return transformed_data