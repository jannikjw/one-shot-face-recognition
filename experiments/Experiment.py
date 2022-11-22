from abc import ABC
from json import Config

class Experiment(ABC):
    """Abstract Experiment class that is inherited to all experiments"""
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)
        
    @abstractmethod
    def load_data(self):
        """Define the dataset."""

        img_folder = self.config['img_folder']
        mapping_file = self.config['mapping_file']
        transform = _preprocess_data()
        
        self.dataset = CelebADataset(img_folder, mapping_file, transform) # TODO: Make sure this works
    
    @abstractmethod
    def _preprocess_data(self): 
        """
        Define the transformations and apply augmentations/ MTCNN if desired. 
        This method should be the same for all experiments
        """
        if self.config['use_augmentation']:
            transform=transforms.Compose([
                transforms.Resize(image_size),
                np.float32,
                transforms.ToTensor(),
                fixed_image_standardization
            ]) #TODO: implement augmentations
        else:
            transform=transforms.Compose([
                transforms.Resize(image_size),
                np.float32,
                transforms.ToTensor(),
                fixed_image_standardization
            ])
        
        if self.config['use_mtcnn']:
            mtcnn = MTCNN(
                image_size=image_size, margin=0, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=False,
                device=device
            )
            
            #TODO: Apply to images
        
        return transform
    
    
    @abstractmethod
    def _load_image_train(self):
        """
        Define and load the train dataset.
        This method 
        """
        # Create dataloaders
        train_loader = DataLoader( #TODO: Make sure this works
            celeba_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            sampler=SubsetRandomSampler(train_inds)
        ) 
        pass

    @abstractmethod
    def _load_image_test(self):
        """
        Define and load the test dataset.
        This method should be the same for all experiments
        """
        test_loader = DataLoader( #TODO: Make sure this works
            celeba_dataset,
            num_workers=num_workers,
            batch_size=test_batch_size,
            pin_memory=pin_memory,
            sampler=SubsetRandomSampler(test_inds)
        )

    @abstractmethod
    def build(self):
        """Create model."""
        pass

    def _train_step(X, y, train_df):
        X, y = celeba_dataset.find_positive_observations(X, y, train_df) #TODO: Make sure it works in class structure

        # Create embeddings
        X_emb = resnet(X.to(device))
        optimizer.zero_grad()

        loss = criterion(X_emb, y.to(device))
        loss.backward()
        optimizer.step()
        return loss
    
    @abstractmethod
    def train(self):
        """Determine training routine, select which layers should be trained, and fit the model."""
        for epoch in tqdm(range(epochs), desc="Epochs", leave=True): #TODO: Make sure this works in class structure
            running_loss = []
            for step, (X,y) in enumerate(tqdm(train_loader, desc='Current Batch', leave=True)):
                loss = self._train_step(X, y, train_df)
                running_loss.append(loss.cpu().detach().numpy())

            loss_total.append(np.mean(running_loss))
            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))
            
        # TODO: Calculate validation accuracy
        # TODO: Calculate train accuracy
        # TODO: Store all outputs in a folder
        pass

    @abstractmethod
    def evaluate(self):
        """Predict results for test set and measure accuracy."""
        knn = KNeighborsClassifier(n_neighbors=1) #TODO: Make sure this works in class structure and parameterize n_neighbors
        knn.fit(train_embeddings, train_labels)
        score = knn.score(test_embeddings, test_labels)

        pass
    
class ExperimentAllFiles(Experiment):
    """Experiment class for using all available files for train set"""
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)
    
    def _load_image_train(self):
        """
        Define and load the train dataset.
        """
        # Create dataloaders
        train_loader = DataLoader( #TODO: Make sure this works
            celeba_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            sampler=SubsetRandomSampler(train_inds)
        )
    
    
class Experiment2Shot(Experiment):
    """Experiment class for 2-shot train set"""
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)
    
    def _load_image_train(self):
        """
        Define and load the train dataset.
        """
        # Create dataloaders
        train_loader = DataLoader( #TODO: Make sure this works
            celeba_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            sampler=SubsetRandomSampler(train_inds)
        )
        
class Experiment1Shot(Experiment):
    """Experiment class for 1-shot train set"""
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)
    
    def _load_image_train(self):
        """
        Define and load the train dataset.
        """
        # Create dataloaders
        train_loader = DataLoader( #TODO: Make sure this works
            celeba_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            sampler=SubsetRandomSampler(train_inds)
        ) 