## Create a custom Dataset class
import os
import torch
from torch import positive, tensor
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from PIL import Image
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from torchvision import transforms
from src.utils.similarity_functions import (
    cosine_similarity,
    min_norm_2,
    min_norm_2_squared,
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CelebADatasetOrig(Dataset):
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

    def __getitem__(self, idx):
        # Get the path to the image
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert("RGB")
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img, self.image_names[idx]

    def get_file_label_mapping(self):
        return self.file_label_mapping

    def get_label_from_file_name(self, file_name):
        return self.file_label_mapping[
            self.file_label_mapping["file_name"] == file_name
        ]["person_id"].values[0]

    def get_labels_from_file_names(self, file_names: list):
        labels = self.file_label_mapping[
            self.file_label_mapping["file_name"].isin(file_names)
        ]["person_id"].values
        print(f"Number of people in dataset: {len(np.unique(labels))}")
        return labels


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

        # Labels Dataframe contains entire image dataset
        file_label_mapping = pd.read_csv(
            mapping_file, header=None, sep=" ", names=["file_name", "person_id"]
        )
        # only keep files in the mapping_file (labels dataframe) that are actually in the img_folder directory
        self.file_label_mapping = file_label_mapping[file_label_mapping['file_name'].isin(image_names)]

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
        #target = self.file_label_mapping.iloc[idx]['person_id']
        
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
            #return label - 1
            return label

    # def get_labels_from_file_names(self, file_names: list):
    #     labels = list(map(lambda x: int(x)-1, self.file_label_mapping[
    #         self.file_label_mapping["file_name"].isin(file_names)
    #     ]["person_id"].tolist()))
    #     if labels is None:
    #         raise Exception('No Labels found.')
    #     else:
    #         print(f"Number of people in dataset: {len(np.unique(labels))}")
    #         return labels


    # def get_file_label_mapping(self):
    #     return self.file_label_mapping

    # def get_label_from_file_name(self, file_name):
    #     return self.file_label_mapping[
    #         self.file_label_mapping["file_name"] == file_name
    #     ]["person_id"].values[0]

    def get_labels_from_file_names(self, file_names: list):
        labels = self.file_label_mapping[
            self.file_label_mapping["file_name"].isin(file_names)
        ]["person_id"].values
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
        print(f'Calculating the {function} metric...')
        n = len(test_embeddings)
        predictions = []
        predictions_files = []
        closest_image_file_name = ""

        for idx, test_embedding in tqdm(enumerate(test_embeddings)):
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


    def load_data_specific_images(self, files_to_load: list):
        """
        Load data and create embeddings.
        """
        embeddings = tensor([])
        no_face_detected = []
        count = 0
        face_file_names = []

        for file_name in tqdm(files_to_load):
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


# Vikram
class CelebADatasetTriplet(CelebADataset):
    def __init__(self, root_dir, mapping_file: str, transform=None, 
                train: bool = True, img_ext: str = 'pgm'):
        """
        Args:
          root_dir (string): Directory with all the images
          mapping_file (string): File path to mapping file from image to person
          transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        image_names = os.listdir(root_dir)
        image_names = [x for x in image_names if x.split(".")[-1]==img_ext]
        print(f'Image names size is: {len(image_names)}')
        self.is_train = train


        self.file_label_mapping = pd.read_csv(
            mapping_file, header=None, sep=" ", names=["file_name", "person_id"]
        )
        self.file_label_mapping = self.file_label_mapping.sort_values(by=["file_name"]).reset_index(drop=True)

        self.root_dir = root_dir
        self.transform = transform
        self.image_names = natsorted(image_names)


    def get_image_label(self, idx):
        # Get the path to the image
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert("RGB")
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        img_name = self.image_names[idx]

        return img, self.file_label_mapping["person_id"][self.file_label_mapping["file_name"]==img_name].iloc[0], img_name

    def __getitem__(self, idx):
        
        anchor, anchor_label, anchor_name = self.get_image_label(idx)

        if self.is_train:

            # loading positive image
            pos_list = self.file_label_mapping["file_name"][(self.file_label_mapping["person_id"]==anchor_label) & (self.file_label_mapping["file_name"]!= anchor_name)]
            pos_name = pos_list.sample(n=1, random_state=42)
            pos_idx = pos_name.index[0]

            positive, pos_label, pos_name = self.get_image_label(pos_idx)

            # loading negative image
            neg_list = self.file_label_mapping["file_name"][self.file_label_mapping["person_id"]!=anchor_label]
            neg_name = neg_list.sample(n=1, random_state=42)
            neg_idx = neg_name.index[0]

            negative, neg_label, neg_name = self.get_image_label(neg_idx)

            return anchor, positive, negative, anchor_label

        else:
            return anchor

