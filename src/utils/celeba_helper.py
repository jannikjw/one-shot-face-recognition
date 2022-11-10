## Create a custom Dataset class
import os
import torch
from torch import tensor
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from facenet_pytorch import MTCNN, InceptionResnetV1, training, fixed_image_standardization
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
    
def finetune_model(model, train_loader, val_loader, loss_fn=torch.nn.CrossEntropyLoss(), metrics={}, epochs:int=10, lr=0.001):   
    # Make use of multiple GPUs if available
    CUDA = torch.cuda.is_available()
    nGPU = torch.cuda.device_count()

    if CUDA:
        model = model.cuda()
        if nGPU > 1:
            model = nn.DataParallel(model)
        
    logits = model.module.logits.parameters() if nGPU > 1 else model.logits.parameters()
    optimizer = optim.Adam(logits, lr=lr)
    
    scheduler = MultiStepLR(optimizer, [5, 10])
    
    metric_tracker = {}
    
    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        model.train()
        train_loss, train_metrics = training.pass_epoch(
            model, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )
        
        
        model.eval()
        val_loss, val_metrics = training.pass_epoch(
            model, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

    writer.close()
    
    return model

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    predictions = torch.tensor([])
    labels = torch.tensor([])
    
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        idx = 0
        for X, y in tqdm(dataloader, total=num_batches):
            idx += 1
            X, y = X.to(device), y.to(device)
            pred = model(X)
                
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            if not predictions.numel():
                predictions = pred.cpu()
                labels = y.cpu()
            else:
                predictions = torch.cat([predictions, pred.cpu()])
                labels = torch.cat([labels, y.cpu()])
            
            del pred, X, y
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return predictions, labels
        
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