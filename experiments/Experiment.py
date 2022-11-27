from src.utils.config import Config
from src.utils.celeba_helper import CelebADataset
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
from tqdm import tqdm
import os, json
from src.utils.triplet_loss import BatchAllTtripletLoss
import torch.optim as optim
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

class Experiment:
    """Experiment class"""
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)

        # Dynamically change the device based on the system
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        parent_dir = '' #'one-shot-face-recognition/' # parent directory for path (github repo name)

        # Initialize the directory structure
        if not os.path.exists(parent_dir+'embeddings'):
            os.mkdir(parent_dir+'embeddings') # This is the folder for the embeddings
        
        if not os.path.exists(parent_dir+'weights'):
            os.mkdir(parent_dir+'weights') # This is the folder for the weights

        if not os.path.exists(parent_dir+'model_data'):
            os.mkdir(parent_dir+'model_data') # This is the folder for all the model related data

        # For the current experiment, we create a separate folder inside mode
        self.folder_name = 'BASELINE'
        self.folder_name += '_FT' if self.config.train.is_finetuning else ''
        self.folder_name += '_FE' if self.config.train.is_feature_extracting else ''
        self.folder_name += '_AUG' if self.config.train.uses_augmentation else ''
        self.folder_name += '_GAN' if self.config.train.uses_GAN else ''
        self.folder_name += '_MTCNN' if self.config.train.uses_MTCNN else ''

        # Create the folder for the experiment inside the weights folder
        self.weights_folder_name = f"{parent_dir}weights/{self.folder_name}"
        if not os.path.exists(self.weights_folder_name):
            os.mkdir(self.weights_folder_name)   

        # Create the folder for the experiment inside the embeddings folder
        self.embeddings_folder_name = f"{parent_dir}embeddings/{self.folder_name}"
        if not os.path.exists(self.embeddings_folder_name):
            os.mkdir(self.embeddings_folder_name)

        # Create the file to store all the model information
        self.model_data_file_name = f"{parent_dir}model_data/{self.folder_name}.json"
        json_data = {"config": cfg}
        with open(self.model_data_file_name, 'w') as model_data:
            json.dump(json_data, model_data)

    def load_data(self):
        """Define the dataset."""

        img_folder = self.config.data.img_folder
        mapping_file = self.config.data.mapping_file
        self.transform = self._preprocess_data()

        self.dataset = CelebADataset(img_folder, mapping_file, self.transform)
    
    
    def _preprocess_data(self): 
        """
        Define the transformations and apply augmentations if desired. 
        This method should be the same for all experiments
        """
        image_size = self.config.data.image_size

        if self.config.train.uses_augmentation: # basic data augmentation
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
        else: # standard transform
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                np.float32,
                transforms.ToTensor(),
                fixed_image_standardization
            ])
        
        return transform
    
    # def _train_test_helper(self, file_label_mapping):
    #     """
    #     Helper Function for Splitting Train & Test
    #     Goal: Test Set should be the same no matter the experiment run
    #     """
    #     # get first file for each person (this will be in train set) 10,177 images
    #     first_img_file = file_label_mapping.drop_duplicates(subset='person_id', keep='first', inplace=False, ignore_index=False)
    #     rest = file_label_mapping.drop(first_img_file.index, axis=0, inplace=False)

    #     # get second file for each person (this will be in test set) 10,133 images
    #     second_img_file = rest.drop_duplicates(subset='person_id', keep='first', inplace=False, ignore_index=False) 
    #     rest.drop(second_img_file.index, axis=0, inplace=True)

    #     # sample the remaining images for the test set to reach 35,000 total images
    #     test_sample = rest.sample(n=35000-len(second_img_file), random_state=42)
    #     rest.drop(test_sample.index, axis=0, inplace=True)

    #     return first_img_file, second_img_file, test_sample, rest
    
    # def _create_train_test_split(self):
    #     """
    #     Unified Train/Test for Experiments
    #     Train Set: if orig data or mtcnn - 167,599 images of 10,177 unique ppl
    #                if GAN Aug - 261,027 images of 10,177 unique ppl (all gan generated images will be in the training set)
    #     Test Set: 35,000 images of 10,133 unique ppl

    #     Returns the indices of the train (can vary btw experiments) & test images (same for all experiments) in the full celeba dataset & the vault inds (same for all experiments)
    #     """
        
    #     file_label_mapping = self.dataset.file_label_mapping

    #     if self.config.train.uses_GAN:
    #         # get the images from the original celeba dataset which all have filenames of length 10 ['000001.jpg' to '202599.jpg']
    #         orig_celeba = file_label_mapping.loc[file_label_mapping['file_name'].str.len() == 10]

    #         first_img_file, second_img_file, test_sample, rest = self._train_test_helper(orig_celeba)

    #         # get the GAN generated images (these will all go in train to keep test set same for all experiments)
    #         gan_imgs = file_label_mapping.drop(orig_celeba.index, axis=0, inplace=False)

    #         # create train df
    #         self.train_df = pd.concat([first_img_file, rest, gan_imgs]) # add GAN generated images to train set
    #     else:
    #         first_img_file, second_img_file, test_sample, rest = self._train_test_helper(file_label_mapping)
            
    #          # create train df
    #         self.train_df = pd.concat([first_img_file, rest])

    #     # create test df (same for all experiments)
    #     self.test_df = pd.concat([second_img_file, test_sample])
        
    #     train_inds = self.train_df.index.tolist()
    #     test_inds = self.test_df.index.tolist()
        
    #     # define the first file images as the vault images that will be used for inference (same for all experiments)
    #     vault_inds = first_img_file.index.tolist()

    #     print(f'Total Dataset size is: {len(self.dataset)}')
    #     print(f'Train Set size is: {len(train_inds)}')
    #     print(f'Test Set size is: {len(test_inds)}')
    #     print(f'Vault Set size is: {len(vault_inds)}')

    #     return train_inds, test_inds, vault_inds

    def _train_test_helper(self, file_label_mapping):
        """
        Helper Function for Splitting Train & Test
        Goal: Test Set should be the same no matter the experiment run (50,000 images)
        """
        # get the first unique image file for each person (10,177 imgs)
        first_img_file = file_label_mapping.drop_duplicates(subset='person_id', keep='first', inplace=False, ignore_index=False)
        
        # we want to make sure that they are some images with labels in the test set that the train set has never seen before
        # split the 10,177 first_file_images which contains an image from each unique person in the dataset
        # sample 1000 unique images from first_file_img. These will go in the test set and any image that has a label from this sample will be removed from training set
        test_fif_unique = first_img_file.sample(n=1000, random_state=42)
        # remaing ~9000 unique images from first_file_img will go into train
        train_fif_unique = first_img_file.drop(test_fif_unique.index, axis=0, inplace=False)
        #get remaining images in dataset after removing the first_file_imgs
        rest = file_label_mapping.drop(first_img_file.index, axis=0, inplace=False)

        # find all images in rest of dataset that has the same label(s) as the 1000 unique images in the test set. We don't want those images in training as well.
        test_fif_unique_match_labels = rest.loc[rest['person_id'].isin(test_fif_unique.person_id.values)]
        # remove the above images with matching labels to the 1000 unique files in the test set from the rest of dataset
        rest.drop(test_fif_unique_match_labels.index, axis=0, inplace=True)

        # get the second unique image file after removing the labels from the 1000 unique images in the test set
        second_img_file = rest.drop_duplicates(subset='person_id', keep='first', inplace=False, ignore_index=False)
        # remove these second_img_files from rest dataset
        rest.drop(second_img_file.index, axis=0, inplace=True)

        # upsample test set to make a test set of 35000 images
        test_len = len(test_fif_unique) + len(test_fif_unique_match_labels) + len(second_img_file)
        test_sample = rest.sample(n=50000-test_len, random_state=42)
        # remaining images go in train
        rest.drop(test_sample.index, axis=0, inplace=True)

        # build train df
        train_df = pd.concat([train_fif_unique, rest])

        # build test df
        test_df = pd.concat([test_fif_unique, test_fif_unique_match_labels, second_img_file, test_sample]) 
        
        return first_img_file, train_df, test_df, test_fif_unique

    def _create_train_test_split(self):
        """"
        Train Set Size: Can change depending on initial dataset (use_GAN boolean)
        Test Set Size: 50,000 images - same for all experiments
        Vault Set Size: 10,177 same images for all experiments (all unique ppl exist in the vault)

        returns train_inds, test_inds, vault_inds
        
        """
        # file label mapping dataframe for all images in dataset
        file_label_mapping = self.dataset.file_label_mapping

        if self.config.train.uses_GAN:
            # get the images from the original celeba dataset which all have filenames of length 10 ['000001.jpg' to '202599.jpg']
            orig_celeba = file_label_mapping.loc[file_label_mapping['file_name'].str.len() == 10]

            first_img_file, train_df, test_df, test_fif_unique = self._train_test_helper(orig_celeba)

            # get the GAN generated images, remove any GAN augs that have same label as the 1000 unique in test, add rest to train for data aug
            gan_imgs = file_label_mapping.drop(orig_celeba.index, axis=0, inplace=False)
            gan_imgs = gan_imgs.loc[~gan_imgs['person_id'].isin(test_fif_unique.person_id.values)]

            # create train df
            self.train_df = pd.concat([train_df, gan_imgs])

        else: 
            first_img_file, train_df, test_df, test_fif_unique = self._train_test_helper(file_label_mapping)

            # create train df
            self.train_df = train_df

        # create test df (same for all experiments)
        self.test_df = test_df  

        train_inds = self.train_df.index.tolist()
        test_inds = self.test_df.index.tolist()
        
        # define the first file images as the vault images that will be used for inference (same for all experiments)
        vault_inds = first_img_file.index.tolist()

        # store labels for data in test that is unseen by train
        self.unseen_labels = test_fif_unique.person_id.values

        print(f'Total Dataset size is: {len(self.dataset)}')
        print(f'Train Set size is: {len(train_inds)}')
        print(f'Number of Unique Ppl in Train is: {self.train_df.person_id.nunique()}')
        print(f'Test Set size is: {len(test_inds)}')
        print(f'Number of Unique Ppl in Test is: {self.test_df.person_id.nunique()}')
        print(f'Vault Set size is: {len(vault_inds)}')
        print(f'Number of Unique Ppl in Vault is: {first_img_file.person_id.nunique()}')
        print(f'Number of Unique ppl in Test Set that DO NOT appear in Train: {len(np.setdiff1d(self.test_df.person_id.values, self.train_df.person_id.values))}')

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
        # Create dataloaders
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
     
    @staticmethod 
    def set_parameter_requires_grad(model):
        for name, param in model.named_parameters():
            if "last" not in name:
                param.requires_grad = False
    
    @staticmethod
    def load_image(path, transform):
        img = Image.open(path).convert("RGB")
        if transform:
                img = transform(img)
        return img
    
    @staticmethod
    def create_embeddings(model, dataloader, device, dataset_size, batch_size):
        embeddings = torch.tensor([])
        
        for idx, batch in tqdm(enumerate(dataloader), total=int(dataset_size/batch_size)):
            imgs, batch_labels, _ = batch
            batch_embeddings = model(imgs.to(device)).detach().cpu()

            if not embeddings.numel():
                embeddings = batch_embeddings
                labels = batch_labels
            else:
                embeddings = torch.cat([embeddings, batch_embeddings])
                labels = torch.cat([labels, batch_labels])
        
        return embeddings, labels

    def build(self):
        """Create model."""
        self.model = InceptionResnetV1(
            pretrained=self.config.model.model_weights).to(self.device)

        # Feature Extracting is freezing all the layers except the last one
        # Fine Tuning is freezing nothing at all.
        if self.config.train.is_feature_extracting:
            Experiment.set_parameter_requires_grad(self.model)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.train.optimizer.lr)
        
        self.criterion = BatchAllTtripletLoss()
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=self.config.train.optimizer.schedule, 
            gamma=self.config.model.gamma
        )

    def _train_step(self, X, y, train_df):
        # find positive observations for images in each batch
        X, y = self.dataset.find_positive_observations(X, y, train_df)

        # Create embeddings
        X_emb = self.model(X.to(self.device))
        
        self.optimizer.zero_grad()

        loss = self.criterion(X_emb, y.to(self.device))
        
        loss.backward()
        
        self.optimizer.step()

        return loss

    def train(self):
        """Determine training routine, select which layers should be trained, and fit the model."""
        # Loading Data
        self.load_data()
        print('Loaded Data')

        # Create Train Test Split & Load
        train_inds, test_inds, vault_inds = self._create_train_test_split()
        self._load_image_train(train_inds)
        self._load_image_test(test_inds)
        self._load_image_vault(vault_inds)
        print('Created Train Test Split & Dataloaders')

        # Build Model
        self.build()
        print('Build Model')
        
        # Model in training mode
        self.model.train()
        print('Start Training Model')
        
        mean_train_loss_per_epoch = []

        for epoch in tqdm(range(self.config.train.epochs), desc="Epochs", leave=True):
            running_loss = []
            # X batch imgs, y batch labels
            for step, (X, y, _) in enumerate(tqdm(self.train_loader, desc='Current Batch', leave=True)): 
                loss = self._train_step(X, y, self.train_df)
                running_loss.append(loss.cpu().detach().numpy())

            # Append the train loss
            mean_train_loss_per_epoch.append(np.mean(running_loss))
            
            # Add the train loss to the json file under model data
            with open(self.model_data_file_name, 'r') as model_data:
                json_data = json.load(model_data)
            
            # make sure there's an array first
            is_training_losses_present = json_data.get("training_losses", False)
            if not is_training_losses_present: 
                json_data['training_losses'] = []

            json_data['training_losses'].append(float(np.mean(running_loss))) # converted to float for json serializable
            
            with open(self.model_data_file_name, 'w') as model_data:
                json.dump(json_data, model_data)

            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, self.config.train.epochs, np.mean(running_loss)))

            # Saving Model Weights after every 10 epochs
            if (epoch + 1) % 10 == 0:
                # Get the path first
                model_state_path = os.path.join(self.weights_folder_name, f'_epoch_{epoch + 1}.pth')
                # Save the weights
                torch.save(self.model.state_dict(), model_state_path)
        
        print('Training Finished')

        # Model Evaluation
        print('Model Evaluation')
        self.model.eval().to(self.device)
       
        print('Saving Embeddings')
        # Computing train embeddings and saving them
        train_embeddings, train_labels = Experiment.create_embeddings(
            model=self.model,
            dataloader=self.train_loader,
            device=self.device,
            dataset_size=len(train_inds),
            batch_size=self.config.train.batch_size
        )

        train_embeddings_file_name = os.path.join(self.embeddings_folder_name, 'train_embeddings.pickle')
        train_labels_file_name = os.path.join(self.embeddings_folder_name, 'train_labels.pickle')
       
        torch.save(train_embeddings, train_embeddings_file_name)
        torch.save(train_labels, train_labels_file_name)
        print('Train Embeddings Saved')

        # Saving the vault embeddings and labels
        vault_embeddings, vault_labels = Experiment.create_embeddings(
            model=self.model,
            dataloader=self.vault_loader,
            device=self.device,
            dataset_size=len(vault_inds),
            batch_size=self.config.train.batch_size
        )

        vault_embeddings_file_name = os.path.join(self.embeddings_folder_name, 'vault_embeddings.pickle')
        vault_labels_file_name = os.path.join(self.embeddings_folder_name, 'vault_labels.pickle')

        torch.save(vault_embeddings, vault_embeddings_file_name)
        torch.save(vault_labels, vault_labels_file_name)
        print('Vault Embeddings Saved')

        # Saving the test embeddings and label
        test_embeddings, test_labels = Experiment.create_embeddings(
            model=self.model,
            dataloader=self.test_loader,
            device=self.device,
            dataset_size=len(test_inds),
            batch_size=self.config.train.batch_size
        )

        test_embeddings_file_name = os.path.join(self.embeddings_folder_name, 'test_embeddings.pickle')
        test_labels_file_name = os.path.join(self.embeddings_folder_name, 'test_labels.pickle')

        torch.save(test_embeddings, test_embeddings_file_name)
        torch.save(test_labels, test_labels_file_name)
        print('Test Embeddings Saved')
        
        # Storing Test Accuracy
        test_accuracy = self.evaluate(vault_embeddings, vault_labels, test_embeddings, test_labels)

        # Unseen Label Test Accuracy only on 1000 unique labels in test
        mask = torch.isin(test_labels, torch.from_numpy(self.unseen_labels))
        unseen_label_test_accuracy = self.evaluate(vault_embeddings, vault_labels, test_embeddings[mask], test_labels[mask])

        with open(self.model_data_file_name, 'r') as model_data:
            json_data = json.load(model_data)
        
        json_data['test_accuracy'] = float(test_accuracy) # convert to float for json serializable
        json_data['unseen_label_test_accuracy'] = float(unseen_label_test_accuracy) # convert to float for json serializable

        with open(self.model_data_file_name, 'w') as model_data:
            json.dump(json_data, model_data)
        print('Saved Test Accuracy')
    
    def evaluate(self, vault_embeddings, vault_labels, test_embeddings, test_labels):
        """Predict results for test set and measure accuracy."""
        
        knn = KNeighborsClassifier(n_neighbors=1)

        knn.fit(vault_embeddings, vault_labels)

        score = knn.score(test_embeddings, test_labels)

        return score
    
    
    # @staticmethod
    # def create_embeddings(model, dataloader):
    #     gt_labels = []
    #     embeddings = torch.tensor([])
        
    #     for i, (X, y, _) in enumerate(dataloader):
    #         img = X # Gives use the images

    #         img_emb = model(img) # img is a batch of images

    #         embeddings = torch.cat([embeddings, img_emb]) 

    #         gt_labels.extend(y) # extend the labels

    #         if i == 1:
    #             break

    #     return embeddings, gt_labels
    
# class ExperimentAllFiles(Experiment):
#     """Experiment class for using all available files for train set"""
#     def __init__(self, cfg):
#         self.config = Config.from_json(cfg)
    
#     def _load_image_train(self, train_inds):
#         """
#         Define and load the train dataset.
#         """
#         # Create dataloaders
#         train_loader = DataLoader( # Make sure this works
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
