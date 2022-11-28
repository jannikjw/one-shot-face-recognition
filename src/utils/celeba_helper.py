## Create a custom Dataset class
import os
import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from src.utils.similarity_functions import (
    cosine_similarity,
    min_norm_2,
    min_norm_2_squared,
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CelebA_MTCNN_Helper(Dataset):
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
        self.mapping_file_path = mapping_file
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

        target = self.get_label_from_file_name(self.image_names[idx])

        return img, target, self.image_names[idx]
    

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
     
    
    def create_X_or_less_shot_dataset(self, max_img_pp):
        '''
        Returns a list of file names for the training set with at most max_img_pp images per person.
        '''
        flm = pd.DataFrame.copy(self.file_label_mapping)
        flm = flm.sort_values(['person_id', 'file_name'])
        flm['cumcount'] = flm.groupby('person_id').cumcount()
        flm_img_pp = flm[flm['cumcount'] < max_img_pp]
        files = flm_img_pp['file_name'].values
        labels = flm_img_pp['person_id'].values
        return files, labels
    
    
    def create_X_shot_dataset(self, img_pp):
        """
        Select X images per person for train_files. 
        If a person does not have at least X images in total dataset, disregard them.
        Important: Files are always selected in order of [person_id, 'file_name']. 
        """
        flm = pd.DataFrame.copy(self.file_label_mapping)
        flm['count'] = flm.groupby('person_id')['person_id'].transform('size') # Add column with number of occurences of each person
        flm = flm[flm['count'] >= img_pp] # Need to have at least 2 images for a person (2 go into train set)
        flm = flm.sort_values(['person_id', 'file_name']) # Sort
        flm['cumcount'] = flm.groupby('person_id').cumcount() # Add column with cumulative count
        flm = flm[flm['cumcount'] < img_pp] # Filter the first two images of each person for train set
        files = flm['file_name'].values
        labels = flm['person_id'].values

        return files, labels
    
    
    def find_positive_observations(self, X, y, df, sample=False, num_examples=5):
        """Find the positive observations in the supplied dataset for each observation in X 
        and adds features and labels to X and y, respectively.

        Args:
            X (tensor): Features of images. Shape: [batch_size, channels, width, height]
            y (tensor): Labels: Shape: [batch_size]
            df (pd.DataFrame): Dataframe that contains mapping of file IDs and labels ('person_id')

        Returns:
            (tensor, tensor): _description_
        """

        pos_obs_idx = np.array([], dtype=int)
        for anchor in np.unique(y):
            # get file_ids of all positive examples for anchor 
            # positive examples in dataframe
            #pos_examples = df[df['person_id']==int(anchor)]['file_id'].values        
            pos_examples = df[df['person_id']==int(anchor)].index.values

            if sample and len(pos_examples)!=0:
                pos_examples = np.random.choice(pos_examples, size=num_examples)

            pos_obs_idx = np.hstack([pos_obs_idx, pos_examples])

        for pos_obs in pos_obs_idx:
            # get image and label of positive example
            #pos_img, pos_label = self[pos_obs]
            pos_img, pos_label, _ = self[pos_obs]

            # add to batch
            X = torch.cat((X, torch.unsqueeze(pos_img, 0)), dim=0)
            y = torch.cat((y, torch.tensor([pos_label])), dim=0)

        return X, y


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

    def predict(self, test_embeddings: tensor, train_embeddings: tensor, train_labels:list, function: str = "norm_2"): #type: ignore
        """
        Calculate distance for the test dataset and calculate accuracy
        """

        predictions = []
        predictions_files = []
        closest_image_file_name = ""

        for test_embedding in tqdm(test_embeddings):
            if function == "cosine_similarity":
                predicted_person_id = train_labels[
                    cosine_similarity(test_embedding, train_embeddings)
                ]
            elif function == "norm_2":
                predicted_person_id = train_labels[
                    min_norm_2(test_embedding, train_embeddings)
                ]
            elif function == "norm_2_squared":
                predicted_person_id = train_labels[
                    min_norm_2_squared(test_embedding, train_embeddings)
                ]

            predictions.append(predicted_person_id)

        return predictions


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

def get_embeddings(model, dataloader):
    model.eval()
    embeddings = torch.tensor([])
    
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        imgs, batch_labels = batch
        batch_embeddings = model(imgs.to(device)).detach()
        
        if not embeddings.numel():
            embeddings = batch_embeddings
            labels = batch_labels
        else:
            embeddings = torch.cat([embeddings, batch_embeddings])
            labels = torch.cat([labels, batch_labels])
        
        del batch_embeddings, batch_labels
    
    return embeddings.cpu(), labels.cpu()
                           
def get_embeddings_and_file_names(model, data_loader, embeddings_path='', labels_path='', save_tensors=True):
    if not os.path.exists(embeddings_path) or not os.path.exists(labels_path):
        embeddings, labels = get_embeddings(model, data_loader)
        if save_tensors:
            torch.save(embeddings, embeddings_path)
            torch.save(labels, labels_path)
    else:
        embeddings = torch.load(embeddings_path).cpu()
        labels = torch.load(labels_path).cpu()
            
    return embeddings, labels


# Vikram
class CelebADatasetTriplet(CelebADataset):
    def __init__(self, root_dir, mapping_file: str, transform=None, 
                train: bool = True, img_ext: str = 'jpg'):
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
            mapping_file, header=None, sep=" ", names=["file_name", "person_id", "is_train"]
        )
        self.file_label_mapping = self.file_label_mapping.sort_values(by=["file_name"]).reset_index(drop=True)

        self.root_dir = root_dir
        self.transform = transform
        self.image_names = natsorted(image_names)

        self.train_image_names = self.file_label_mapping[self.file_label_mapping["is_train"]==1]["file_name"].reset_index(drop=True)
        self.test_image_names = self.file_label_mapping[self.file_label_mapping["is_train"]==0]["file_name"].reset_index(drop=True)

        # train, test dataframe
        self.train_df = self.file_label_mapping[self.file_label_mapping["is_train"]==1].reset_index(drop=True)
        self.test_df = self.file_label_mapping[self.file_label_mapping["is_train"]==0].reset_index(drop=True)

    def __len__(self):
        if self.is_train:
            return len(self.train_df)
        else:
            return len(self.test_df)


    def get_image_label(self, idx, get_train=True):
        # Get the filename from train or test data
        if get_train:
            assert idx < len(self.train_image_names), "Index is out of range for train dataset"
            filename = self.train_image_names.loc[idx]

        else:
            if idx >= len(self.test_image_names):
                print(idx)
            assert idx < len(self.test_image_names), "Index is out of range for test dataset"
            filename = self.test_image_names.loc[idx]


        img_path = os.path.join(self.root_dir, filename)
        # Load image and convert it to RGB
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise(f"get_train is {get_train}, idx is {idx}, image name is: {filename}")
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        label = self.file_label_mapping["person_id"][self.file_label_mapping["file_name"]==filename].iloc[0]

        return img, label, filename

    def __getitem__(self, idx):

        if self.is_train:
            # loading the train image
            anchor, anchor_label, anchor_name = self.get_image_label(idx, get_train=True)

            # loading positive image
            pos_list = self.test_df["file_name"][(self.test_df["person_id"]==anchor_label)]
            if len(pos_list) == 0:
                positive = anchor
            else:
                pos_name = pos_list.sample(n=1) #random_state=42
                pos_idx = pos_name.index[0]

                positive, pos_label, pos_name = self.get_image_label(pos_idx, get_train=False)

            # loading negative image
            neg_list = self.test_df["file_name"][(self.test_df["person_id"]!=anchor_label)]
            neg_name = neg_list.sample(n=1, random_state=42)
            neg_idx = neg_name.index[0]

            negative, neg_label, neg_name = self.get_image_label(neg_idx, get_train=False)

            return anchor, positive, negative, anchor_label

        else:
            anchor, anchor_label, anchor_name = self.get_image_label(idx, get_train=False)
            return anchor, anchor_label