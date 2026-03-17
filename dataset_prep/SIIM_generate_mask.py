import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component

SAVE_PATH = './SIIM/segmentation_masks'
df = pd.read_csv('./SIIM/stage_train.csv')
df = df.drop_duplicates('ImageId')

os.makedirs(SAVE_PATH, exist_ok=True)

for idx in tqdm(df.index):
    annotations = df['EncodedPixels'][idx]
    mask = np.zeros((1024, 1024), dtype=np.float32)
    if annotations != '-1': 
        if isinstance(annotations, list): 
            for rle in annotations:
                mask += run_length_decode(rle)  
        else:
            mask = run_length_decode(annotations)
        mask = (mask >= 1).astype('float32')
        filename = str(df['ImageId'][idx]) + '.png'
        plt.imsave(os.path.join(SAVE_PATH, filename), mask, cmap='gray')
        continue
  
