import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import arff
import h5py
from ucimlrepo import fetch_ucirepo

# load data in various formats

# load text data
def read_data(filePath, dataname, has_labels=True):
    if filePath.endswith('txt'):
        data = []
        labels = []
        with open(filePath, 'r+') as fr:
            for line in fr:
                value = line.split(',')
                value = np.float32(np.array(value))
                if has_labels:
                    data.append(value[:-1])
                    labels.append(value[-1])
                else:
                    data.append(value[0])
        return np.array(data), np.array(labels)
    elif filePath.endswith('xlsx'):
        df = pd.read_excel(
            filePath,
            engine='openpyxl',
        )
        data = df.values[:, :-1]
        labels = df.values[:, -1]

        return data, labels
    elif filePath.endswith('csv'):
        df = pd.read_csv(filePath)
        data = df.values[:, :-1]
        labels = df.values[:, -1]

        return data, labels
    elif filePath.endswith('arff'):
        data = load_arff(filePath)
        if dataname == 'dermatology':
            data = np.delete(data, [33, 34, 35, 36, 262, 263, 264, 265], axis=0)
        if dataname == 'vowel':
            data = np.delete(data, [0, 1, 2], axis=1)
        if dataname == 'wdbc':
            data = np.delete(data, [0], axis=1)
            first_column = data[:, 0].reshape(-1, 1)
            rest_columns = data[:, 1:]
            data = np.concatenate((rest_columns, first_column), axis=1)
        if dataname == 'wine':
            first_column = data[:, 0].reshape(-1, 1)
            rest_columns = data[:, 1:]
            data = np.concatenate((rest_columns, first_column), axis=1)
        if dataname == 'yeast':
            data = np.delete(data, [0], axis=1)

        # assume the arff data have the same structure, the last column is the label in the text form
        encoder = LabelEncoder()
        data[:, -1] = encoder.fit_transform(data[:, -1])
        data = data.astype(float)
        # np.savetxt(f'ground-truth-labels/{dataname}.txt', data[:, -1], fmt='%s')
        return data[:, :-1], data[:, -1]
    elif filePath.endswith('h5'):
        _, _, data, labels = load_h5(filePath)
        return data, labels
    elif filePath.startswith('import'):
        dataset = fetch_ucirepo(id=int(filePath[6:]))
        X = dataset.data.features
        y = dataset.data.targets
        return X, y


def load_arff(file_name):
    # path = 'path/to/your/file.arff'
    with open(file_name, 'r') as file:
        data = arff.load(file)
    data_values = np.array(data['data'])
    # data, meta = arff.loadarff(file_name)
    # reshaped_data = np.array([item for sublist in data for item in sublist]).reshape(len(data), -1)
    return data_values

def load_h5(file_path):
    with h5py.File(file_path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
    return X_tr, y_tr, X_te, y_te
