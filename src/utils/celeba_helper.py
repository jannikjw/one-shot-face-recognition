## Create a custom Dataset class
import os
import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from PIL import Image
import numpy as np
import pandas as pd
from src.utils.similarity_functions import (
    cosine_similarity,
    min_norm_2,
    min_norm_2_squared,
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CelebADataset(Dataset):
    def __init__(self, root_dir, mapping_file: str, transform=None):
        """
        Args:
          root_dir (string): Directory with all the images
          mapping_file (string): File path to mapping file from image to person
          transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        image_names = os.listdir(root_dir)

        self.file_label_mapping = pd.read_csv(
            mapping_file, header=None, sep=" ", names=["file_name", "person_id"]
        )
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = natsorted(image_names)

    def __len__(self):
        return len(self.image_names)

    # def __getitem__(self, idx):
    #     # Get the path to the image
    #     img_path = os.path.join(self.root_dir, self.image_names[idx])
    #     # Load image and convert it to RGB
    #     img = Image.open(img_path).convert("RGB")
    #     # Apply transformations to the image
    #     if self.transform:
    #         img = self.transform(img)

    #     return img, self.image_names[idx]
    
    def __getitem__(self, idx):
        # Get the path to the image
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert("RGB")
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        target = self.get_label_from_file_name(self.image_names[idx])

        return img, target

    def get_file_label_mapping(self):
        return self.file_label_mapping

    def get_label_from_file_name(self, file_name):
        label = self.file_label_mapping[
            self.file_label_mapping["file_name"] == file_name
        ]["person_id"].values[0]
        if label is None:
            raise Exception('No Label found.')
        else:
            return label - 1

    def get_labels_from_file_names(self, file_names: list):
        labels = list(map(lambda x: int(x)-1, self.file_label_mapping[
            self.file_label_mapping["file_name"].isin(file_names)
        ]["person_id"].tolist()))
        if labels is None:
            raise Exception('No Labels found.')
        else:
            print(f"Number of people in dataset: {len(np.unique(labels))}")
            return labels


class CelebAClassifier:
    def __init__(self, celeba_dataloader: DataLoader, detection_model, embedding_model):
        """
        Args:
        TODO: Write docstring
        """

        self.data_loader = celeba_dataloader
        self.dataset = celeba_dataloader.dataset
        self.detection_model = detection_model
        self.embedding_model = embedding_model

    def predict(self, test_embeddings: tensor, train_embeddings: tensor, anchor_file_names: list, function: str = "norm_2"): #type: ignore
        """
        Calculate distance for the test dataset and calculate accuracy
        """

        predictions = []
        predictions_files = []
        closest_image_file_name = ""

        for test_embedding in test_embeddings:
            if function == "cosine_similarity":
                closest_image_file_name = anchor_file_names[
                    cosine_similarity(test_embedding, train_embeddings)
                ]
            elif function == "norm_2":
                closest_image_file_name = anchor_file_names[
                    min_norm_2(test_embedding, train_embeddings)
                ]
            elif function == "norm_2_squared":
                closest_image_file_name = anchor_file_names[
                    min_norm_2_squared(test_embedding, train_embeddings)
                ]

            predicted_person_id = self.dataset.get_label_from_file_name(
                closest_image_file_name
            )

            predictions.append(predicted_person_id)
            predictions_files.append(closest_image_file_name)

        return predictions, predictions_files


    def load_data_specific_images(self, files_to_load: list): #TODO: Delete function as you can use SubsetRandomSampler instead
        """
        Load data and create embeddings.
        """
        embeddings = tensor([])
        no_face_detected = []
        count = 0
        face_file_names = []

        for file_name in files_to_load:
            count += 1
            img = Image.open(f"{self.dataset.root_dir}/{file_name}")
            aligned, _ = self.detection_model(img, return_prob=True)

            if aligned is not None:
                aligned = aligned.reshape(
                    [1, aligned.shape[0], aligned.shape[1], aligned.shape[2]]
                )
                face_file_names.append(file_name)

                aligned = aligned.to(device)
                batch_embeddings = self.embedding_model(aligned).detach().cpu()

                if not embeddings.numel():
                    embeddings = batch_embeddings
                else:
                    embeddings = torch.cat([embeddings, batch_embeddings])

                if count % 100 == 0:
                    print(f"Images loaded: {count} / {len(files_to_load)}")
            else:
                no_face_detected.append(file_name)

        print(
            f"No face detected in {len(no_face_detected)} images. The images are the following:\n{no_face_detected}"
        )
        print(f"Size of the embeddings: {embeddings.shape}")

        return embeddings, face_file_names

def save_file_names(file_names: list, destination_path: str):
    with open(destination_path, "w") as fp:
        for item in file_names:
            # write each item on a new line
            fp.write("%s\n" % item)
        print("Done")
        