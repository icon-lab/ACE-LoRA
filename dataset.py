from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import os
import numpy as np

class ChexpertDataset(Dataset):
    def __init__(self, dataframe : pd.DataFrame, transform, data_dir=' '):
        self.df = dataframe
        self.transform = transform
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.data_dir, self.df['Path'][idx])).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = torch.from_numpy(self.df.iloc[idx,6:-1].values.astype(np.int8))

        data = {'image': image, 'label': label}

        return data

class MIMIC_CXR_Dataset(Dataset):
    def __init__(self, dataframe : pd.DataFrame, transform):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.df['full_path'].iloc[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        target = self.df['text'].iloc[idx]

        return image, target, idx

def data_split(df: pd.DataFrame):    
    df_train = df[df['full_path'].str.contains('train')]
    df_val = df[df['full_path'].str.contains('val')]
    df_test = df[df['full_path'].str.contains('test')]

    return df_train, df_val, df_test