import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn import cluster, datasets, mixture
import torch

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor

class CrescentDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, n_points = 100, gauss_noise = 0.05):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            n_points: (int) The total number of points generated
            noise: (double) Standard deviation of Gaussian noise added to the data.
        """

        self.xcoords = datasets.make_moons(n_samples=n_points, noise=gauss_noise)[0].astype(np.float32) # Make two interleaving half circles
        self.labels = datasets.make_moons(n_samples=n_points, noise=gauss_noise)[1].astype(np.float32)


    def __len__(self):
        # return size of dataset
        return len(self.xcoords)

    def __getitem__(self, idx):
        """
        Fetch index idx data point and labels from dataset.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            xcoords: (double) The generated samples.
            label: (int) The integer labels (0 or 1) for class membership of each sample.
        """
        return torch.from_numpy(self.xcoords[idx,:]), self.labels[idx]



def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            # path = os.path.join(data_dir, "{}_signs".format(split)) # currently not used

            # initialize the data loader
                # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(CrescentDataset(1000, 0.05), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
                # dl = DataLoader(SIGNSDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                #                         num_workers=params.num_workers,
                #                         pin_memory=params.cuda)
                # torch.utils.data.DataLoader provides an iterator that takes in a Dataset object and performs batching, shuffling and loading of the data.
            else:
                dl = DataLoader(CrescentDataset(100, 0.05), batch_size=params.batch_size, shuffle=True,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)
                # dl = DataLoader(SIGNSDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                #                 num_workers=params.num_workers,
                #                 pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
