import os
import csv
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class TrainImageNetDataset(Dataset):
    def __init__(self, data_path, data_labels_path, label_mapping_path, transform, data_reduction = 0.0):

        self.data_paths = []
        self.data_labels = {}
        
        for path, subdirs, files in os.walk(data_path):
            # temp_data = []
            for name in files:
                full_path = os.path.join(path, name)
                self.data_paths.append(full_path)

            #     temp_data.append(full_path)
            # temp_data = temp_data[:int(len(temp_data) * (1 - data_reduction))]
            # self.data_paths.extend(temp_data)

        label_mapping = {}

        with open(label_mapping_path) as f:
            reader = csv.reader(f)
            for index, mapping in enumerate(reader):
                mapping = mapping[0]
                id = mapping.split()[0]
                label_mapping[id] = index

        labels = {}

        for path in self.data_paths:
            file_name = path.split("/")[-1].split(".")[0]
            self.data_labels[file_name] = label_mapping[file_name.split("_")[0]]
        
        self.transform=transform
        
    def __len__(self):
        return len(self.data_labels.keys())

    def __getitem__(self, idx):
        image = plt.imread(self.data_paths[idx])
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        elif image.shape[2] == 4:
            image = np.asarray(Image.open(self.data_paths[idx]).convert("RGB"))

        preprocessed_image = self.transform(Image.fromarray(image))
            
        data_file_name = self.data_paths[idx].split("/")[-1].split(".")[0]
        
        label = self.data_labels[data_file_name]
        return preprocessed_image, label
        

class TestImageNetDataset(Dataset):
    def __init__(self, data_path, data_labels_path, label_mapping_path, transform):

        self.data_paths = []
        self.data_labels = {}
        for path, subdirs, files in os.walk(data_path):
            for name in files:
                full_path = os.path.join(path, name)
                self.data_paths.append(full_path)


        label_mapping = {}
        with open(label_mapping_path) as f:
            reader = csv.reader(f)
            for index, mapping in enumerate(reader):
                mapping = mapping[0]
                id = mapping.split()[0]
                label_mapping[id] = index
        
        labels = {}
        with open(data_labels_path) as csvfile:
            reader = csv.reader(csvfile)
            for index, row in enumerate(reader):
                # skip header
                if index == 0:
                    continue
                    
                file_name = row[0]
                label = row[1].split()[0]
                self.data_labels[file_name] = label_mapping[label]
        
        self.transform=transform
        
    def __len__(self):
        return len(self.data_labels.keys())

    def __getitem__(self, idx):
        image = plt.imread(self.data_paths[idx])
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        elif image.shape[2] == 4:
            image = np.asarray(Image.open(self.data_paths[idx]).convert("RGB"))

        preprocessed_image = self.transform(Image.fromarray(image))
            
        data_file_name = self.data_paths[idx].split("/")[-1].split(".")[0]
        
        label = self.data_labels[data_file_name]
        return preprocessed_image, label 