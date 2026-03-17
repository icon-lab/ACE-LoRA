import pandas as pd

classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

df = pd.read_csv('./Chest_x_ray_14/data_Data_Entry_2017_v2020.csv')
with open('./Chest_x_ray_14/test_list.txt', 'r') as f:
    test_filenames = f.read().splitlines() 

is_test = df['Image Index'].isin(test_filenames)
test_size = int(len(test_filenames) * 0.2)
print(test_size)
df_test = df[is_test]
df_test_eval = df_test[:test_size]

df_test_eval.to_csv('./Chest_x_ray_14/test_set_biomedclip.csv', index=False)

df = pd.read_csv('./Chest_x_ray_14/test_set_biomedclip.csv')

labels = df['Finding Labels'].tolist()
diseases = [label.split('|') for label in labels]
size = len(classes)

df['Label Indices'] = df['Finding Labels'].apply(lambda x: [1 if disease in x else 0 for disease in classes])
print(type(df['Label Indices'][0]))
print((df['Label Indices'][0]))
df.to_csv('./test_set_biomedclip_hot_encode.csv', index=False) 



