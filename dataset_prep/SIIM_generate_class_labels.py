import pandas as pd
from tqdm import tqdm

df = pd.read_csv('./SIIM/stage_2_train.csv')
df = df.drop_duplicates('ImageId')
test_ratio = 0.3
print(len(df))
test_size = int(len(df) * test_ratio)
sampled_df = df.sample(n=test_size, random_state=42)
new_rows = []

for _, row in tqdm(sampled_df.iterrows()):
    new_row = {
        'ImageId': row['ImageId'],
        'Label': 0 if row['EncodedPixels'] == '-1' else 1
    }
    new_rows.append(new_row)

new_df = pd.DataFrame(new_rows)
print(new_df['Label'].sum())
new_df.to_csv('./test_labels.csv')
