from src.utils.config import Config
from src.utils.celeba_helper import CelebADataset
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
from tqdm import tqdm

class Experiment:
    """Abstract Experiment class that is inherited to all experiments"""
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)
        self.device = torch.device(self.config.train.device)
    
    def load_data(self):
        """Define the dataset."""

        img_folder = self.config.data.img_folder
        mapping_file = self.config.data.mapping_file
        transform = self._preprocess_data()

        self.dataset = CelebADataset(img_folder, mapping_file, transform)
    
    
    def _preprocess_data(self): 
        """
        Define the transformations and apply augmentations/ MTCNN if desired. 
        This method should be the same for all experiments
        """
        image_size = self.config.data.image_size

        if self.config.train.uses_augmentation:
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.RandomRotation(degrees=(30, 70)),
                np.float32,
                transforms.ToTensor(),
                fixed_image_standardization
                ])
        else:
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                np.float32,
                transforms.ToTensor(),
                fixed_image_standardization
            ])
        
        return transform
    
    def _create_train_test_split(self):
        """
        Unified Train/Test for Experiments
        Train Set: 167,599 images of 10,177 unique ppl
        Test Set: 35,000 images of 10,133 unique ppl

        Returns the indices of the train and test images in the full celeba dataset
        """
        file_label_mapping = self.dataset.file_label_mapping

        # TODO: NEED TO FIX THIS FOR GAN DATASET
        
        # get first file for each person (this will be in train set)
        first_img_file = file_label_mapping.drop_duplicates(subset='person_id', keep='first', inplace=False, ignore_index=False) # 10,177 images
        # define these first file images as the vault images that will be used for inference for any experiment
        vault_inds = first_img_file.index.tolist()
        rest = file_label_mapping.drop(first_img_file.index, axis=0, inplace=False)
        
        # get second file for each person (this will be in test set)
        second_img_file = rest.drop_duplicates(subset='person_id', keep='first', inplace=False, ignore_index=False) # 10,133 images
        rest.drop(second_img_file.index, axis=0, inplace=True)
        
        # sample the remaining images for the test set to reach 35,000 total images
        test_sample = rest.sample(n=35000-len(second_img_file), random_state=42)
        rest.drop(test_sample.index, axis=0, inplace=True)

        self.train_df = pd.concat([first_img_file, rest])
        self.test_df = pd.concat([second_img_file, test_sample])

        train_inds = self.train_df.index.tolist()
        test_inds = self.test_df.index.tolist()

        return train_inds, test_inds, vault_inds

    def _load_image_train(self, train_inds):
        """
        Define and load the train dataset.
        """

        # Create dataloaders
        self.train_loader = DataLoader(
            self.dataset,
            num_workers=self.config.train.num_workers,
            batch_size=self.config.train.batch_size,
            pin_memory=self.config.train.pin_memory,
            sampler=SubsetRandomSampler(train_inds)
        )
    
    def _load_image_test(self, test_inds):
        """
        Define and load the test dataset.
        This method should be the same for all experiments
        """
        self.test_loader = DataLoader(
            self.dataset,
            num_workers=self.config.train.num_workers,
            batch_size=self.config.train.batch_size,
            pin_memory=self.config.train.pin_memory,
            sampler=SubsetRandomSampler(test_inds)
        )

    def _load_image_vault(self, vault_inds):
        """
        Define and Load the Vault Images for the Vault Embeddings
        This vault images stay the same across all the experiments 10,177 images,first file of each unique person
        """
        
        # Create dataloaders
        self.vault_loader = DataLoader(
        self.dataset,
        num_workers=self.config.train.num_workers,
        batch_size=self.config.train.batch_size,
        pin_memory=self.config.train.pin_memory,
        sampler=SubsetRandomSampler(vault_inds)
        )
     
     
    # def build(self):
    #     """Create model."""
    #     self.resnet = InceptionResnetV1(pretrained=self.model.model_weights).to(self.device)

    # def _train_step(self, X, y, train_df):
    #     X, y = self.dataset.find_positive_observations(X, y, train_df) #TODO: Make sure it works in class structure
    #     #resnet = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    #     X_emb = self.resnet(X.to(self.device))
    #     optimizer.zero_grad()

    #     loss = criterion(X_emb, y.to(device))
    #     loss.backward()
    #     optimizer.step()
    #     return loss
    
    
    # def train(self):
    #     """Determine training routine, select which layers should be trained, and fit the model."""
    #     for epoch in tqdm(range(epochs), desc="Epochs", leave=True): #TODO: Make sure this works in class structure
    #         running_loss = []
    #         for step, (X,y) in enumerate(tqdm(train_loader, desc='Current Batch', leave=True)):
    #             loss = self._train_step(X, y, train_df)
    #             running_loss.append(loss.cpu().detach().numpy())

    #         loss_total.append(np.mean(running_loss))
    #         print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))
            
    #     # TODO: Calculate validation accuracy
    #     # TODO: Calculate train accuracy
    #     # TODO: Store all outputs in a folder
    #     pass

    
    # def evaluate(self):
    #     """Predict results for test set and measure accuracy."""
    #     knn = KNeighborsClassifier(n_neighbors=1) #TODO: Make sure this works in class structure and parameterize n_neighbors
    #     knn.fit(train_embeddings, train_labels)
    #     score = knn.score(test_embeddings, test_labels)

    #     pass
    
# class ExperimentAllFiles(Experiment):
#     """Experiment class for using all available files for train set"""
#     def __init__(self, cfg):
#         self.config = Config.from_json(cfg)
    
#     def _load_image_train(self, train_inds):
#         """
#         Define and load the train dataset.
#         """
#         # Create dataloaders
#         train_loader = DataLoader( #TODO: Make sure this works
#             celeba_dataset,
#             num_workers=num_workers,
#             batch_size=batch_size,
#             pin_memory=pin_memory,
#             sampler=SubsetRandomSampler(train_inds)
#         )
    
    
# class Experiment2Shot(Experiment):
#     """
#     Experiment class for 2-shot train set
#     If a person does not have at least 2 images in train dataset, disregard them.
#     """
#     def __init__(self, cfg):
#         self.config = Config.from_json(cfg)
    
#     def _load_image_train(self, train_inds):
#         """
#         Define and load the train dataset.
#         """
#         flm = pd.DataFrame.copy(self.dataset.file_label_mapping)
#         flm = flm.iloc[train_inds]
#         flm['count'] = flm.groupby('person_id')['person_id'].transform('size')
#         flm = flm[flm['count'] >= 2] # Need to have at least 2 images for a person (2 go into train set)
#         self.train_df = flm.groupby('person_id', sort=False).sample(n=2, random_state=42)#.sort_values(by='file_name')

#         new_train_inds = self.train_df.index.tolist()

#         #Create dataloaders
#         self.train_loader = DataLoader(
#             self.dataset,
#             num_workers=self.config.train.num_workers,
#             batch_size=self.config.train.batch_size,
#             pin_memory=self.config.train.pin_memory,
#             sampler=SubsetRandomSampler(new_train_inds)
#         )
        
# class Experiment1Shot(Experiment):
#     """Experiment class for 1-shot train set"""
#     def __init__(self, cfg):
#         self.config = Config.from_json(cfg)
    
#     def _load_image_train(self):
#         """
#         Define and load the train dataset. 
#         To be used with GAN Data Augmentation: this Training Set must contain the first image file_name for each unique person_id in dataset (10,177) images
#         """
#         # Obtain the first file_name for each person_id in the file_label_mapping dataframe
#         flm = pd.DataFrame.copy(self.dataset.file_label_mapping)
#         self.train_df = flm.drop_duplicates(subset='person_id', keep='first', inplace=False, ignore_index=False) # 10,177 images to match if using GAN data aug
#         new_train_inds = self.train_df.index.tolist()
        
#         # Create dataloaders
#         self.train_loader = DataLoader(
#             self.dataset,
#             num_workers=self.config.train.num_workers,
#             batch_size=self.config.train.batch_size,
#             pin_memory=self.config.train.pin_memory,
#             sampler=SubsetRandomSampler(new_train_inds)
#         ) 