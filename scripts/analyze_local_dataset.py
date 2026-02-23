import pandas as pd
import glob

print('--- BALANCED DATA ---')
for f in glob.glob('g:/Final year project/dataset/multilingual-emotion-dataset/balanced/*.csv'):
    df = pd.read_csv(f)
    print(f'\nFile: {f}')
    print(f'Rows: {len(df)}')
    print(f'Columns: {df.columns.tolist()}')
    if 'emotion' in df.columns:
        print(f'Emotions: {df["emotion"].unique()[:13]}')
    elif 'label' in df.columns:
        print(f'Labels: {df["label"].unique()[:13]}')

print('\n--- IMBALANCED DATA ---')
for f in glob.glob('g:/Final year project/dataset/multilingual-emotion-dataset/imbalanced/*.csv'):
    df = pd.read_csv(f)
    print(f'\nFile: {f}')
    print(f'Rows: {len(df)}')
    print(f'Columns: {df.columns.tolist()}')
    if 'emotion' in df.columns:
        print(f'Emotions: {df["emotion"].unique()[:13]}')
    elif 'label' in df.columns:
        print(f'Labels: {df["label"].unique()[:13]}')
