import random
import torch, os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch_lightning import LightningDataModule
import time
from tqdm import tqdm

class CPSDatasetTrain(Dataset):
    def __init__(self, file_path, list_all_files):
        self.file_path = file_path
        self.list_all_files = list_all_files
    
    def __getitem__(self, index):
        file_name = self.list_all_files[index]
        normalized = file_name.replace("\\", "/")
        if os.path.isabs(normalized):
            target_path = os.path.normpath(normalized)
        else:
            target_path = os.path.normpath(os.path.join(self.file_path, normalized))
        arr = np.load(target_path)
        var = arr["var_hist"].astype(np.float32)
        ch0 = var / 255.0
        ch1 = np.log1p(var) / np.log1p(255.0)
        ch2 = (var > 0).astype(np.float32)
        image = np.stack([ch0, ch1, ch2], axis=0)
                
        return torch.from_numpy(image)
    
    def __len__(self):
        return len(self.list_all_files)
                

class CPSDatasetTest(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.list_all_files = []
        self.scan_all_dir(self.file_path)
    
    def __getitem__(self, index):
        file_name = self.list_all_files[index]
        normalized = file_name.replace("\\", "/")
        target_path = os.path.normpath(normalized)
        arr = np.load(target_path)
        var = arr["var_hist"].astype(np.float32)
        ch0 = var / 255.0
        ch1 = np.log1p(var) / np.log1p(255.0)
        ch2 = (var > 0).astype(np.float32)
        image = np.stack([ch0, ch1, ch2], axis=0)
        label = 1 if ("1_august" in file_name or "2_august" in file_name or "5_july" in file_name) else 0
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.list_all_files)

    def scan_all_dir(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                #if "ideal" in str(root + "\\" + file):
                #    if "non_ideal" in str(root + "\\" + file):
                #        continue
                self.list_all_files.append(str(root + "/" + file))


class CPSDatasetModule(LightningDataModule):
    def __init__(self, file_path_training, file_path_test, train_files_list, val_files_list, batch_size):
        super().__init__()
        self.file_path_training = file_path_training
        self.file_path_test = file_path_test
        self.train_files_list = train_files_list
        self.val_files_list = val_files_list
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    def train_dataloader(self):
        train_split = CPSDatasetTrain(file_path = self.file_path_training, list_all_files=self.train_files_list)
        return DataLoader(train_split, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        test_split = CPSDatasetTest(file_path = self.file_path_test)
        return DataLoader(test_split, batch_size=self.batch_size)

    def val_dataloader(self):
        val_split = CPSDatasetTrain(file_path= self.file_path_training, list_all_files=self.val_files_list)
        return DataLoader(val_split, batch_size=self.batch_size)


def scan_all_dir(path, all_files_list):
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files_list.append(os.path.join(root, file))


######### TEST ##########
if __name__ == "__main__":

#     filepath_train = "Dataset Tesi\\Training Set\\"
#     filepath_test = "Dataset Tesi\\Test All\\Test Set ConnDots\\Test all\\"
    filepath_train = "/scratch/user/u.ae271721/dataset/Dataset/Training_Set"
    filepath_test = "/scratch/user/u.ae271721/dataset/Dataset/Test_All/Test Set ConnDots/Test all/"

    train_files_list = np.load("./connected_dots_split/training.npz")["arr_0"]
    val_files_list = np.load("./connected_dots_split/validation.npz")["arr_0"]

    batch_size = 2

    dataset = CPSDatasetTrain(file_path=filepath_train, list_all_files=train_files_list)
    dataLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(dataLoader))
    batch = batch.unsqueeze(1)

    print(dataset[0])
    print(dataset[0].shape)
    print(len(dataset))
    print(batch.shape)

    dataset_val = CPSDatasetTrain(file_path=filepath_train, list_all_files=val_files_list)
    dataLoader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True)
    batch_val = next(iter(dataLoader_val))
    batch_val = batch_val.unsqueeze(1)

    print(dataset_val[0])
    print(dataset_val[0].shape)
    print(len(dataset_val))
    print(batch_val.shape)


    dataset_val = CPSDatasetTest(file_path=filepath_test)
    dataLoader_test = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    batch_val = next(iter(dataLoader_test))
    batch_val, label = batch_val

    print(dataset_val[0])
    #print(dataset_val[0].shape)
    print(len(dataset_val))
    print(batch_val.shape)

    #datamodule = CPSDatasetModule(file_path_training = filepath_train, file_path_test = filepath_test, train_files_list = train_files_list, val_files_list = val_files_list, batch_size = batch_size)
    #dataloader_test = datamodule.train_dataloader()
    #linee_samples = 0
    #points_samples = 0
    #for i, batch in enumerate(tqdm(iter(dataloader_test))):
    #    image = batch
    #    image = torch.sum(image, dim = [1, 2])
    #    #print(image)
    #    points_samples += torch.sum(image <= 255)
    #    linee_samples += torch.sum(image>=256)
    #print(linee_samples)
    #print(points_samples)

    