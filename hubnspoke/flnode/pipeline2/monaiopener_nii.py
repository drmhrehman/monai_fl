import os

import sys
sys.path.append('.')

import os
import pandas as pd
from flnode.pipeline2.opener import Opener
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch


class MonaiOpenerNii(Opener):
    def __init__(self, data_dir, num_class=0):
        self.data_dir = data_dir
        self.num_class = num_class
        self.class_names = None

    def data_summary(self, folders):
        self.class_names = folders
        self.num_class = len(self.class_names)
        # print("Root Directory (dataset): " + self.data_dir)

        image_files = [
            [
                os.path.join(self.data_dir, self.class_names[i], x)
                for x in os.listdir(os.path.join(self.data_dir, self.class_names[i]))
            ]
            for i in range(self.num_class)
        ]
        num_each = [len(image_files[i]) for i in range(self.num_class)]
        image_files_list = []
        image_class = []
        for i in range(self.num_class):
            image_files_list.extend(image_files[i])
            image_class.extend([i] * num_each[i])
        num_total = len(image_class)

        # get size of each image
        image_sizes = [
            [
                nib.load(f).shape for f in image_files[i]
            ]
            for i in range(self.num_class)
        ]

        print(f"Total image count: {num_total}")
        print("Class details:")
        for i in range(self.num_class):
            print(f"--- Class: {self.class_names[i]}")
            print(f"    Num images: {num_each[i]}")
            mean_size = np.array(image_sizes[i]).mean(axis=0).astype(np.int8)
            print(f"    Mean image size: {mean_size}\n")

        plt.subplots(3, 3, figsize=(8, 8))
        for i, k in enumerate(np.random.randint(num_total, size=9)):
            data = nib.load(image_files_list[k]).get_fdata()
            plt.subplot(3, 3, i + 1)
            plt.xlabel(self.class_names[image_class[k]])
            if data.ndim == 3:
                data_slice = np.s_[:,:,data.shape[2]//2]
            elif data.ndim == 4:
                data_slice = np.s_[:,:,data.shape[2]//2, 0]
            plt.imshow(data[data_slice], cmap="gray")
        plt.tight_layout()
        plt.show()

    def get_x_y(self, folders, frac_val, frac_test):
        train_x = list()
        train_y = list()
        val_x = list()
        val_y = list()
        test_x = list()
        test_y = list()
        
        self.class_names = folders
        self.num_class = len(self.class_names)
        #print("Root Directory (dataset): " + self.data_dir)
        
        image_files = [
            [
                os.path.join(self.data_dir, self.class_names[i], x)
                for x in os.listdir(os.path.join(self.data_dir, self.class_names[i]))
            ]
            for i in range(self.num_class)
        ]
        num_each = [len(image_files[i]) for i in range(self.num_class)]
        image_files_list = []
        image_class = []
        for i in range(self.num_class):
            image_files_list.extend(image_files[i])
            image_class.extend([i] * num_each[i])
        num_total = len(image_class)

        for i in range(num_total):
            rann = np.random.random()
            if rann < frac_val:
                val_x.append(image_files_list[i])
                val_y.append(image_class[i])
            elif rann < (frac_val+frac_test):
                test_x.append(image_files_list[i])
                test_y.append(image_class[i])
            else:
                train_x.append(image_files_list[i])
                train_y.append(image_class[i])
        return (train_x, train_y, val_x, val_y, test_x, test_y)

    def save_predictions(self, y_pred, path):
        with open(path, 'w') as fp:
            y_pred.to_csv(fp, index=False)

    def get_predictions(self, path):
        return pd.read_csv(path)

    def fake_X(self, n_samples):
        return []  # compute random fake data

    def fake_y(self, n_samples):
        return []  # compute random fake data

    def get_X(self, folders):
        return [
            # print(folders)
            folders
        ]

    def get_y(self, folders):
        return [
            folders
            # print(folders)
            # pd.read_csv(folders)#os.path.join(folders, 'y.csv'))
            # for folder in folders
        ]

class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]
