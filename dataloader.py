from matplotlib.streamplot import InvalidIndexError
from torch.utils.data import Dataset
import os, glob
from natsort import natsorted
from PIL import Image

class CelebDataset(Dataset):
    def __init__(self, folder_path, img_ext = None, transform=None):
        if img_ext:
            image_list = [f for f in os.listdir(folder_path) if f.endswith(img_ext)]
        else:
            image_list = os.listdir(folder_path)
        
        self.root_dir = folder_path
        self.transform = transform
        self.image_names = natsorted(image_list)

    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if idx > len(self.image_names) - 1:
            raise InvalidIndexError("Index is out of the range of the images list")
        path = os.path.join(self.root_dir, self.image_names[idx])

        #load image
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)

        return image