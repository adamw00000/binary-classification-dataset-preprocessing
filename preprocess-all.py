# %%
import os
import datapackage
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn.manifold import TSNE

# Toggle to save
save = True

data = {}
os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def process_dataset(dataset, preprocessing_fun, save=False):
    data_url = f'https://datahub.io/machine-learning/{dataset}/datapackage.json'

    # to load Data Package into storage
    package = datapackage.Package(data_url)

    # to load only csv data
    resources = package.resources
    for resource in resources:
        if resource.descriptor['datahub']['type'] == 'derived/csv':
        # if resource.tabular:
            df = pd.read_csv(resource.descriptor['path'])
    

    df = preprocessing_fun(df)
    if np.sum(np.isnan(df.to_numpy())) > 0:
        impute_missing_values(df)
    
    print('Data frame info:')
    print(df.info())
    print('Feature description:')
    display(df.describe())
    print('Data:')
    display(df)

    if save:
        df.to_csv(f'data/{dataset.capitalize()}.csv', index=False)
    
    return df

def summary(data):
    stats = []
    for dataset in data:
        df = data[dataset]

        X = df.drop(columns=['Class', 'BinClass'])
        y = df.Class.to_numpy()
        y_bin = df.BinClass.to_numpy()

        stats.append({
            'Dataset': dataset,
            'N': X.shape[0],
            'Dim': X.shape[1],
            'Classes': len(np.unique(y)),
            'Binary class outlier %': 1 - np.mean(y_bin),
        })
    
    stats_df = pd.DataFrame.from_records(stats)
    return stats_df

def class_composition(df):
    print('Class composition:')
    df = data[dataset]
    y = df.Class
    display(y.value_counts())

def cluster_vis(df, dataset, use_binary_class=False, show_labels=False):
    if 'BinClass' in df.columns:
        X = df.drop(columns=['Class', 'BinClass'], )
    else:
        X = df.drop(columns=['Class'])
        
    if not use_binary_class:
        y = df.Class.to_numpy()
    else:
        y = df.BinClass.to_numpy()

    tsne = TSNE(
        n_components=2, init="pca", random_state=0, learning_rate="auto"
    )
    trans_data = tsne.fit_transform(X)
    palette = sns.color_palette("husl", len(np.unique(y)))

    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(x=trans_data[:, 0], y=trans_data[:, 1], hue=y,
        palette=palette,
        s=40 / np.log10(len(df)))
    plt.legend()

    classes_shown = set()
    if show_labels:
        for sample in range(0, X.shape[0]):
            if y[sample] not in classes_shown:
                classes_shown.add(y[sample])
                txt = scatter.text(trans_data[sample, 0] + 0.01, trans_data[sample, 1], 
                        y[sample], horizontalalignment='left', 
                        size='medium', color=palette[y[sample]], weight='semibold')
                txt.set_path_effects([
                    PathEffects.withStroke(linewidth=0.5, foreground='k')
                ])
    
    plt.title('Original classes' if not use_binary_class else 'Binary classes')
    plt.savefig(
        os.path.join('plots', f'{dataset}{"_binary" if use_binary_class else ""}.png'),
        bbox_inches='tight',
        dpi=300
    )
    plt.show()
    plt.close()

is_class_col = lambda cols: (np.array(cols) == 'Class') | (np.array(cols) == 'BinClass')

def impute_missing_values(df):
    col_mask = ~is_class_col(df.columns)

    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    df.loc[:, col_mask] = imputer.fit_transform(df.loc[:, col_mask])
    return df

def normalize_features(df):
    col_mask = ~is_class_col(df.columns)

    scaler = MinMaxScaler()
    df.loc[:, col_mask] = scaler.fit_transform(df.loc[:, col_mask])
    return df

def standardize_features(df):
    col_mask = ~is_class_col(df.columns)

    scaler = StandardScaler()
    df.loc[:, col_mask] = scaler.fit_transform(df.loc[:, col_mask])
    return df

# %%
dataset = 'abalone'
vis = True

def preprocess(df):
    df.Sex = df.Sex.map({'M': 1, 'F': 0, 'I': 0.5})
    df = df.rename(columns={'Class_number_of_rings': 'Class'})
    df.Class = df.Class - 1
    ordered_classes = [x for x in range(np.max(df.Class) + 1) if x in np.unique(df.Class)]
    class_map = {x: i for i, x in enumerate(ordered_classes)}
    df.Class = df.Class.map(class_map)

    df['BinClass'] = np.where(np.isin(df.Class, [0, 1, 2, 3, 4, 5, 6, 7]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'arrhythmia'
vis = True

def preprocess(df):
    df = df.rename(columns={'class': 'Class'})
    df.Class = df.Class - 1
    ordered_classes = [x for x in range(np.max(df.Class) + 1) if x in np.unique(df.Class)]
    class_map = {x: i for i, x in enumerate(ordered_classes)}
    df.Class = df.Class.map(class_map)

    df = standardize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'banknote-authentication'
vis = True

def preprocess(df):
    df.Class = df.Class - 1

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'breast-w'
vis = True

def preprocess(df):
    df.Class = np.where(df.Class == 'malignant', 1, 0)

    df = normalize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
# dataset = 'covertype'
# vis = True

# def preprocess(df):
#     df = df.rename(columns={'class': 'Class'})
#     df.Class = df.Class - 1

#     # df = standardize_features(df)

#     df['BinClass'] = np.where(np.isin(df.Class, [0, 1]), 1, 0)
#     return df

# data[dataset] = process_dataset(dataset, preprocess, save=save)
# class_composition(data[dataset])
# if vis:
#     cluster_vis(data[dataset], dataset, show_labels=True)
#     cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
# dataset = 'creditcard'
# vis = True

# def preprocess(df):
#     df = df.drop(columns=['Time'])
#     df.Class = np.where(df.Class == "'0'", 0, 1)

#     df[['Amount']] = standardize_features(df[['Amount']])
#     # df = standardize_features(df)

#     df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
#     return df

# data[dataset] = process_dataset(dataset, preprocess, save=save)
# class_composition(data[dataset])
# if vis:
#     cluster_vis(data[dataset], dataset, show_labels=True)
#     cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'dermatology'
vis = True

def preprocess(df):
    df = df.rename(columns={'class': 'Class'})
    df.Class = df.Class - 1

    df = normalize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'diabetes'
vis = True

def preprocess(df):
    df = df.rename(columns={'class': 'Class'})
    df.Class = np.where(df.Class == 'tested_positive', 1, 0)

    df = normalize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'fertility'
vis = True

def preprocess(df):
    df = df.rename(columns={'output': 'Class'})
    df.Class = np.where(df.Class == 'O', 1, 0)
    
    df = normalize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'gas-drift'
vis = True

def preprocess(df):
    df.Class = df.Class - 1

    df = standardize_features(df)
    
    df['BinClass'] = np.where(np.isin(df.Class, [0, 1, 2]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'glass'
vis = True

def preprocess(df):
    df = df.rename(columns={'Type': 'Class'})
    df.Class = df.Class.map({x: i for i, x in enumerate(np.unique(df['Class']))})

    df = normalize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0, 1]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'haberman'
vis = True

def preprocess(df):
    df = df.rename(columns={'Survival_status': 'Class'})
    df.Class = df.Class - 1

    df = normalize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'heart-statlog'
vis = True

def preprocess(df):
    df = df.rename(columns={'class': 'Class'})
    df.Class = np.where(df.Class == 'present', 1, 0)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    df = normalize_features(df)

    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'ionosphere'
vis = True

def preprocess(df):
    df = df.rename(columns={'class': 'Class'})
    df.Class = np.where(df.Class == 'g', 1, 0)

    df = normalize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [1]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'isolet'
vis = True

def preprocess(df):
    df = df.rename(columns={'class': 'Class'})
    df.Class = df.Class.map({x: int(x[1:-1]) for i, x in enumerate(np.unique(df['Class']))})
    df.Class = df.Class - 1

    df['BinClass'] = np.where(np.isin(df.Class, [6, 19, 15, 21, 1, 3, 4]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'jm1'
vis = True

def preprocess(df):
    df = df.rename(columns={'defects': 'Class'})
    df.Class = np.where(df.Class == True, 1, 0)

    df = standardize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'kc1'
vis = True

def preprocess(df):
    df = df.rename(columns={'defects': 'Class'})
    df.Class = np.where(df.Class == True, 1, 0)

    df = standardize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'madelon'
vis = True

def preprocess(df):
    df.Class = df.Class - 1

    df = normalize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'musk'
vis = True

def preprocess(df):
    df = df.drop(columns=['molecule_name', 'conformation_name', 'ID'])
    df = df.rename(columns={'class': 'Class'})

    df = normalize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'optdigits'
vis = True

def preprocess(df):
    df = df.rename(columns={'class': 'Class'})

    df = normalize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [4, 9]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'pendigits'
vis = True

def preprocess(df):
    df = df.rename(columns={'class': 'Class'})

    df = normalize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [4, 9]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'satimage'
vis = True

def preprocess(df):
    df = df.rename(columns={'class': 'Class'})
    df.Class = df.Class - 1
    ordered_classes = [x for x in range(np.max(df.Class) + 1) if x in np.unique(df.Class)]
    class_map = {x: i for i, x in enumerate(ordered_classes)}
    df.Class = df.Class.map(class_map)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'segment'
vis = True

def preprocess(df):
    df = df.rename(columns={'class': 'Class'})
    df.drop(columns=['region-pixel-count'])
    df.Class = df.Class.map({x: i for i, x in enumerate(np.unique(df['Class']))})

    df = standardize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [1, 4]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'seismic-bumps'
vis = True

def preprocess(df):
    df.Class = df.Class - 1

    df = normalize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'semeion'
vis = True

def preprocess(df):
    df.Class = df.Class - 1

    df['BinClass'] = np.where(np.isin(df.Class, [2, 8]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'sonar'
vis = True

def preprocess(df):
    df.Class = np.where(df.Class == 'Mine', 1, 0)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'spambase'
vis = True

def preprocess(df):
    df = df.rename(columns={'class': 'Class'})

    df = standardize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'tic-tac-toe'
vis = True

def preprocess(df):
    df.Class = np.where(df.Class == 'positive', 1, 0)
    df.iloc[:, :-1] = np.where(
        df.iloc[:, :-1] == 'x', 
        1, 
        np.where(
            df.iloc[:, :-1] == 'o',
            0,
            0.5
        )
    ).astype(float)

    df['BinClass'] = np.where(np.isin(df.Class, [1]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'vehicle'
vis = True

def preprocess(df):
    df.Class = df.Class.map({x: i for i, x in enumerate(np.unique(df['Class']))})

    df = standardize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'waveform-5000'
vis = True

def preprocess(df):
    df = df.rename(columns={'class': 'Class'})
    
    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'wdbc'
vis = True

def preprocess(df):
    df.Class = df.Class - 1

    df = standardize_features(df)

    df['BinClass'] = np.where(np.isin(df.Class, [0]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
dataset = 'yeast'
vis = True

def preprocess(df):
    df = df.rename(columns={'class_protein_localization': 'Class'})
    df.Class = df.Class.map({x: i for i, x in enumerate(np.unique(df['Class']))})

    df['BinClass'] = np.where(np.isin(df.Class, [6]), 1, 0)
    return df

data[dataset] = process_dataset(dataset, preprocess, save=save)
class_composition(data[dataset])
if vis:
    cluster_vis(data[dataset], dataset, show_labels=True)
    cluster_vis(data[dataset], dataset, use_binary_class=True)

# %%
summary_df = summary(data)
display(summary_df)
summary_df.to_csv('summary.csv', index=False)

# %%
