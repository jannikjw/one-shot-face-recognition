from dataloader import CelebDataset
from torch.utils.data import DataLoader
import os


class Args():
    def __init__(self):
        self.batch_size = 6
        self.img_folder_path = "../Data/"
        self.img_extention = ".jpeg"


if __name__ == "__main__":
    args = Args()

    dataset = CelebDataset(args.img_folder_path, img_ext = args.img_extention)
    dataloader = DataLoader(dataset, batch_size = args.batch_size,
                            shuffle = True)

    print("hello")
