import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-8 # an arbitrary small value to be used for numerical stability tricks


def min_norm_2(test_value, trained_examples):
    dists = [(test_value - anchor).norm().item() for anchor in trained_examples]
    return np.argmin(dists)


def min_norm_2_squared(test_value, trained_examples):
    dists = [(test_value - anchor).norm().item() ** 2 for anchor in trained_examples]
    return np.argmin(dists)


def cos_sim(a, b):
    """
    Takes 2 vectors a, b and returns the cosine similarity
    """
    dot_product = np.dot(a, b)  # x.y
    norm_a = np.linalg.norm(a)  # |x|
    norm_b = np.linalg.norm(b)  # |y|
    return dot_product / (norm_a * norm_b)


def cosine_similarity(test_value, trained_examples):
    similarities = [cos_sim(test_value, anchor).item() for anchor in trained_examples]
    return np.argmax(similarities)

def euclidean_distance_matrix(x):
  """Efficient computation of Euclidean distance matrix
  Args:
    x: Input tensor of shape (batch_size, embedding_dim)
    
  Returns:
    Distance matrix of shape (batch_size, batch_size)
  """
  # step 1 - compute the dot product

  # shape: (batch_size, batch_size)
  dot_product = torch.mm(x, x.t())

  # step 2 - extract the squared Euclidean norm from the diagonal

  # shape: (batch_size,)
  squared_norm = torch.diag(dot_product)

  # step 3 - compute squared Euclidean distances

  # shape: (batch_size, batch_size)
  distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)

  # get rid of negative distances due to numerical instabilities
  distance_matrix_pos = F.relu(distance_matrix)

  # step 4 - compute the non-squared distances
  
  # handle numerical stability
  # derivative of the square root operation applied to 0 is infinite
  # we need to handle by setting any 0 to eps
  mask = (distance_matrix == 0.0).float()

  # use this mask to set indices with a value of 0 to eps
  distance_matrix_pos += mask * eps

  # now it is safe to get the square root
  distance_matrix_root = torch.sqrt(distance_matrix_pos)

  # undo the trick for numerical stability
  distance_matrix *= (1.0 - mask)

  return distance_matrix
