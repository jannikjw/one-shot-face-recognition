import numpy as np


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
