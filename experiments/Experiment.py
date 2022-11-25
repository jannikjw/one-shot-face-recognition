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
    """Abstract Experiment class that is inherited to all experiments"""
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)
        # self.device = torch.device(self.config.train.device)
        # Dynamically change the device based on the system
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        p_dir = 'one-shot-face-recognition/' # parent directory (github repo name)

        # Also initialize the directory structure
        if not os.path.exists(p_dir+'data'):
            os.mkdir(p_dir+"data") # This is the folder for training data
        
        if not os.path.exists(p_dir+'embeddings'):
            os.mkdir(p_dir+'embeddings') # This is the folder for the embeddings
        
        if not os.path.exists(p_dir+'weights'):
            os.mkdir(p_dir+'weights') # This is the folder for the embeddings

        if not os.path.exists(p_dir+'model_data'):
            os.mkdir(p_dir+'model_data') # This is the folder for all the model related data

        # For the current experiment, we create a separate folder inside mode
        self.folder_name = 'BASELINE'
        self.folder_name += '_FT' if self.config.train.is_finetuning else ''
        self.folder_name += '_FE' if self.config.train.is_feature_extracting else ''
        self.folder_name += '_AUG' if self.config.train.uses_augmentation else ''
        self.folder_name += '_GAN' if self.config.train.uses_GAN else ''
        self.folder_name += '_MTCNN' if self.config.train.uses_MTCNN else ''

        # Create the folder for the experiment inside the weights folder
        self.weights_folder_name = f"{p_dir}weights/{self.folder_name}"
        if not os.path.exists(self.weights_folder_name):
            os.mkdir(self.weights_folder_name)   

        # Create the folder for the experiment inside the embeddings folder
        self.embeddings_folder_name = f"{p_dir}embeddings/{self.folder_name}"
        if not os.path.exists(self.embeddings_folder_name):
            os.mkdir(self.embeddings_folder_name)

        # Create the file to store all the model information
        self.model_data_file_name = f"{p_dir}model_data/{self.folder_name}.json"
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
    
    def _train_test_helper(self, file_label_mapping):
        """
        Helper Function for Splitting Train & Test
        Goal: Test Set should be the same no matter the experiment run
        """
        # get first file for each person (this will be in train set) 10,177 images
        first_img_file = file_label_mapping.drop_duplicates(subset='person_id', keep='first', inplace=False, ignore_index=False)
        rest = file_label_mapping.drop(first_img_file.index, axis=0, inplace=False)

        # get second file for each person (this will be in test set) 10,133 images
        second_img_file = rest.drop_duplicates(subset='person_id', keep='first', inplace=False, ignore_index=False) 
        rest.drop(second_img_file.index, axis=0, inplace=True)

        # sample the remaining images for the test set to reach 35,000 total images
        test_sample = rest.sample(n=35000-len(second_img_file), random_state=42)
        rest.drop(test_sample.index, axis=0, inplace=True)

        return first_img_file, second_img_file, test_sample, rest
    
    def _create_train_test_split(self):
        """
        Unified Train/Test for Experiments
        Train Set: 167,599 images of 10,177 unique ppl
        Test Set: 35,000 images of 10,133 unique ppl

        Returns the indices of the train and test images (same for all experiments) in the full celeba dataset and the vault inds (same for all experiments)
        """
        
        file_label_mapping = self.dataset.file_label_mapping

        if self.config.train.uses_GAN:
            print('using GAN Aug Images')
            # get the images from the original celeba dataset which all have filenames of length 10 ['000001.jpg' to '202599.jpg']
            orig_celeba = file_label_mapping.loc[file_label_mapping['file_name'].str.len() == 10]

            first_img_file, second_img_file, test_sample, rest = self._train_test_helper(orig_celeba)

            # get the GAN generated images (these will all go in train to keep test set same for all experiments)
            gan_imgs = file_label_mapping.drop(orig_celeba.index, axis=0, inplace=False)

            # create train df
            self.train_df = pd.concat([first_img_file, rest, gan_imgs]) # add GAN generated images to train set
        else:
            print('GAN is False')
            first_img_file, second_img_file, test_sample, rest = self._train_test_helper(file_label_mapping)
            
             # create train df
            self.train_df = pd.concat([first_img_file, rest])

        # create test df (same for all experiments)
        self.test_df = pd.concat([second_img_file, test_sample])

        print(f'Train Set is: {len(self.train_df)} and Test Set is: {len(self.test_df)}')
        
        train_inds = self.train_df.index.tolist()
        test_inds = self.test_df.index.tolist()
        # define the first file images as the vault images that will be used for inference (same for all experiments)
        vault_inds = first_img_file.index.tolist()

        return train_inds, test_inds, vault_inds
    
    def reduced_dataset_exp(self):
        """
        Testing Experiments.py on a smaller dataset to see if it runs properly
        Train: 30,000 images
        TestL 20,000 images
        """
        print('in small celeba dataset')
        file_label_mapping = self.dataset.file_label_mapping
        first_img_file = file_label_mapping.drop_duplicates(subset='person_id', keep='first', inplace=False, ignore_index=False)
        vault_inds = first_img_file.index.tolist()
        rest = file_label_mapping.drop(first_img_file.index, axis=0, inplace=False)
        second_img_file = rest.drop_duplicates(subset='person_id', keep='first', inplace=False, ignore_index=False) # 10,133 images
        rest.drop(second_img_file.index, axis=0, inplace=True)
        test_sample = rest.sample(n=20000-len(second_img_file), random_state=42)
        rest.drop(test_sample.index, axis=0, inplace=True)

        small_train = rest.sample(n=30000-len(first_img_file), random_state=42)

        self.train_df = pd.concat([first_img_file, small_train]) # 30,000 images
        print(len(self.train_df))
        self.test_df = pd.concat([second_img_file, test_sample]) #20,000 images
        print(len(self.test_df))

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
    def create_embeddings(dataloader, model, transform):
        gt_labels = []
        embeddings = torch.tensor([])
        
        for i, (X, y, _) in enumerate(dataloader):
            img = X # Gives use the images

            img_emb = model(img) # img is a batch of images

            embeddings = torch.cat([embeddings, img_emb]) 

            gt_labels.extend(y) # extend the labels

        return embeddings, gt_labels

    def build(self):
        """Create model."""
        print('in build function')
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
        print('in train step')
        X, y = self.dataset.find_positive_observations(X, y, train_df) #TODO: Make sure it works in class structure
        print('passed positive observations')
        # Create embeddings
        X_emb = self.model(X.to(self.device))
        
        self.optimizer.zero_grad()

        loss = self.criterion(X_emb, y.to(self.device))
        
        loss.backward()
        
        self.optimizer.step()
        print('returning loss')
        return loss

    def train(self):
        """Determine training routine, select which layers should be trained, and fit the model."""
        print('inside train function')
        # Loading Data
        self.load_data()
        #train_inds, test_inds, vault_inds = self._create_train_test_split()
        train_inds, test_inds, vault_inds = self.reduced_dataset_exp()
        self._load_image_train(train_inds)
        self._load_image_test(test_inds)
        self._load_image_vault(vault_inds)

        # Build Model
        self.build()
        print('got through build function')
        
        # Model in training mode
        self.model.train()
        print('start training')
        
        mean_train_loss_per_epoch = []

        for epoch in tqdm(range(self.config.train.epochs), desc="Epochs", leave=True): #TODO: Make sure this works in class structure
            running_loss = []
            for step, (X,y,file_names) in enumerate(tqdm(self.train_loader, desc='Current Batch', leave=True)):
                loss = self._train_step(X, y, self.train_df)
                running_loss.append(loss.cpu().detach().numpy())


            # Append the train loss
            mean_train_loss_per_epoch.append(np.mean(running_loss))
            
            # Add the train loss to the json file under model data
            with open(self.model_data_file_name, 'r') as model_data:
                json_data = json.load(model_data)
            
            # TODO: make sure there's an array first
            is_training_losses_present = json_data.get("training_losses", False)
            if not is_training_losses_present: 
                json_data['training_losses'] = []

            json_data['training_losses'].append(np.mean(running_loss))
            
            with open(self.model_data_file_name, 'w') as model_data:
                json.dump(json_data, model_data)

            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, self.config.train.epochs, np.mean(running_loss)))
            
            # Saving Model Weights after every 10 epochs
            if (epoch + 1) % 10 == 0:
                # Get the path first
                model_state_path = self.weights_folder_name + f'_epoch_{epoch + 1}.pth'
                # Save the weights
                torch.save(self.model.state_dict(), model_state_path)
        
        # Computing vault embeddings and saving them
        self.model.eval().to(self.device)

        # TODO: Computing train embeddings and saving them
        train_embeddings, train_labels = Experiment.create_embeddings(
            dataloader=self.train_loader,
            model = self.model,
            transform=self.transform,
        )

        train_embeddings_file_name = os.path.join(self.embeddings_folder_name, 'train_embeddings.pickle')
        train_labels_file_name = os.path.join(self.embeddings_folder_name, 'train_labels.pickle')

        torch.save(train_embeddings, train_embeddings_file_name)
        torch.save(train_labels, train_labels_file_name)

        # TODO: Saving the vault embeddings and labels
        vault_embeddings, vault_labels = Experiment.create_embeddings(
            dataloader=self.vault_loader,
            model = self.model,
            transform=self.transform
        )

        vault_embeddings_file_name = os.path.join(self.embeddings_folder_name, 'vault_embeddings.pickle')
        vault_labels_file_name = os.path.join(self.embeddings_folder_name, 'vault_labels.pickle')

        torch.save(vault_embeddings, vault_embeddings_file_name)
        torch.save(vault_labels, vault_labels_file_name)

        # TODO: Saving the test embeddings and label
        test_embeddings, test_labels = Experiment.create_embeddings(
            dataloader=self.test_loader,
            model = self.model,
            transform=self.transform,
        )

        test_embeddings_file_name = os.path.join(self.embeddings_folder_name, 'test_embeddings.pickle')
        test_labels_file_name = os.path.join(self.embeddings_folder_name, 'test_labels.pickle')

        torch.save(test_embeddings, test_embeddings_file_name)
        torch.save(test_labels, test_labels_file_name)
        
        # TODO: Storing Test Accuracy
        test_accuracy = self.evaluate(self, train_embeddings, train_labels, test_embeddings, test_labels)

        with open(self.model_data_file_name, 'r') as model_data:
            json_data = json.load(model_data)
        json_data['test_accuracy'] = test_accuracy

        with open(self.model_data_file_name, 'w') as model_data:
            json.dump(json_data, model_data)

    
    def evaluate(self, train_embeddings, train_labels, test_embeddings, test_labels):
        """Predict results for test set and measure accuracy."""
        
        knn = KNeighborsClassifier(n_neighbors=1)

        knn.fit(train_embeddings, train_labels)

        score = knn.score(test_embeddings, test_labels)

        return score
    
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