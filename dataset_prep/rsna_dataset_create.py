# code for splitting RSNA dataset for zero-shot classification and detection tasks
import os
from dataset import *
import pandas as pd
from run_utils import *
from sklearn.model_selection import train_test_split
import ast

PNEUMONIA_ORIGINAL_TRAIN_CSV = './RSNA/stage_2_train_labels.csv'
PNEUMONIA_IMG_DIR = './RSNA/full_dataset'
PNEUMONIA_VALID_CSV = './RSNA/val.csv'
PNEUMONIA_TEST_CSV = './RSNA/test.csv'

PNEUMONIA_TRAIN_CSV_DET = './RSNA/detection/train_det.csv'
PNEUMONIA_VALID_CSV_DET = './RSNA/detection/val_det.csv'
PNEUMONIA_TEST_CSV_DET = './RSNA/detection/test_det.csv'

df = pd.read_csv(PNEUMONIA_ORIGINAL_TRAIN_CSV)

detection = True
only_disease = True

def create_bbox(row):
    if row["Target"] == 0:
        return 0
    else:
        x1 = row["x"]
        y1 = row["y"]
        x2 = x1 + row["width"]
        y2 = y1 + row["height"]
        return [x1, y1, x2, y2]

df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)

df = df[["patientId", "bbox"]]
df = df.groupby("patientId").agg(list)
df = df.reset_index()
df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)
# df["bbox"] = df["bbox"].apply(ast.literal_eval)

# create labels
df["Target"] = df["bbox"].apply(lambda x: 0 if x == None else 1)

# no encoded pixels mean healthy
df["Path"] = df["patientId"].apply(lambda x: os.path.join(PNEUMONIA_IMG_DIR, (x + ".dcm")))

df = df[df["Target"] != 0]
    
print(len(df))
exit()
if detection:
    test_fac = 0.3
    # split data
    train_df, val_test_df = train_test_split(df, test_size=test_fac, random_state=0)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=0)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Target"].value_counts())
    print(f"Number of valid samples: {len(val_df)}")
    print(val_df["Target"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Target"].value_counts())

    train_df.to_csv(PNEUMONIA_TRAIN_CSV_DET)
    val_df.to_csv(PNEUMONIA_VALID_CSV_DET)
    test_df.to_csv(PNEUMONIA_TEST_CSV_DET)
else:
    test_fac = 0.15
    # split data
    val_df, test_df = train_test_split(df, test_size=test_fac, random_state=0)

    print(f"Number of valid samples: {len(val_df)}")
    print(val_df["Target"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Target"].value_counts())

    val_df.to_csv(PNEUMONIA_VALID_CSV)
    test_df.to_csv(PNEUMONIA_TEST_CSV)
