import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.similarity_functions import euclidean_distance_matrix

def get_triplet_mask(labels):
  """compute a mask for valid triplets
  Args:
    labels: Batch of integer labels. shape: (batch_size,)
  Returns:
    Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
    A triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j`, `k` are different.
  """
  # step 1 - get a mask for distinct indices

  # shape: (batch_size, batch_size)
  indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
  indices_not_equal = torch.logical_not(indices_equal)
  # shape: (batch_size, batch_size, 1)
  i_not_equal_j = indices_not_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_not_equal_k = indices_not_equal.unsqueeze(1)
  # shape: (1, batch_size, batch_size)
  j_not_equal_k = indices_not_equal.unsqueeze(0)
  # Shape: (batch_size, batch_size, batch_size)
  distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

  # step 2 - get a mask for valid anchor-positive-negative triplets

  # shape: (batch_size, batch_size)
  labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
  # shape: (batch_size, batch_size, 1)
  i_equal_j = labels_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_equal_k = labels_equal.unsqueeze(1)
  # shape: (batch_size, batch_size, batch_size)
  valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

  # step 3 - combine two masks
  mask = torch.logical_and(distinct_indices, valid_indices)

  return mask


class BatchAllTtripletLoss(nn.Module):
  """Uses all valid triplets to compute Triplet loss
  Args:
    margin: Margin value in the Triplet Loss equation
  """
  def __init__(self, margin=1.):
    super().__init__()
    self.margin = margin
    
  def forward(self, embeddings, labels):
    """computes loss value.
    Args:
      embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
      labels: Batch of integer labels associated with embeddings. shape: (batch_size,)
    Returns:
      Scalar loss value.
    """
    # step 1 - get distance matrix
    # shape: (batch_size, batch_size)
    distance_matrix = euclidean_distance_matrix(embeddings)

    # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix

    # shape: (batch_size, batch_size, 1)
    anchor_positive_dists = distance_matrix.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    anchor_negative_dists = distance_matrix.unsqueeze(1)
    # get loss values for all possible n^3 triplets
    # shape: (batch_size, batch_size, batch_size)
    triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin

    # step 3 - filter out invalid or easy triplets by setting their loss values to 0

    # shape: (batch_size, batch_size, batch_size)
    mask = get_triplet_mask(labels)
    triplet_loss *= mask
    # easy triplets have negative loss values
    triplet_loss = F.relu(triplet_loss)

    # step 4 - compute scalar loss value by averaging positive losses
    num_positive_losses = (triplet_loss > eps).float().sum()
    triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)

    return triplet_loss