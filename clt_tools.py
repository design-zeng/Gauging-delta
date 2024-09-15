import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
import os
import pandas as pd

def prepare_test_cases_auto(folder_path):
    test_cases = {}
    try:
        # List all files in the folder
        file_names = os.listdir(folder_path)
    except Exception as e:
        print("Error:", e)

    for file_name in file_names:
        test_cases[os.path.splitext(file_name)[0]] = folder_path + file_name
    return test_cases

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise ValueError('')

def create_distance_matrix(data):
    n = len(data)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            distance_matrix[i, j] = np.linalg.norm(data[i] - data[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix

def read_data(filePath, has_labels=True):
    data = []
    labels = []
    with open(filePath, 'r+') as fr:
        for line in fr:
            value = line.split()
            value = np.float32(np.array(value))
            data.append(value[:-1])
            if has_labels:
                labels.append(value[-1])
    return np.array(data), np.array(labels)


def draw_clusters(data, labels):
    f = plt.figure()
    dimension = len(data[0])
    if dimension == 2:
        ax = f.add_subplot()
        ax.set_aspect('equal', adjustable='box')
    else:
        ax = f.add_subplot(projection='3d')

    label_unique = np.unique(labels)
    for k in label_unique:
        # rows = np.where(data[0][:, 2] == k)
        mask = (labels == k)
        if dimension == 2:
            plt.scatter(data[mask, 0], data[mask, 1], s=6)
        else:
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2], s=6)

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if dimension == 3:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    return f
    # plt.title(name)
    plt.savefig(f'visualization-ground-truth/{name}.png')
    plt.clf()


def load_image(file_name):
    image = plt.imread(file_name)
    image = cv2.resize(image,(50, 50))
    RGB = image.reshape(-1, 3)
    plt.imshow(image)


    X = np.arange(image.shape[0])
    Y = np.arange(image.shape[1])
    x, y = np.meshgrid(X, Y)
    RGBXY = np.column_stack((RGB, x.reshape(-1, 1), y.reshape(-1, 1)))
    RGBXY *= [1, 1, 1, 0.01, 0.01]
    RGBXY = np.float32(RGBXY)
    return RGBXY


def load_clustering_data(file_name):
    # plt.clf('all')
    data = np.loadtxt(file_name)
    # img = np.empty((50, 50, 3))
    # img = np.empty((50, 75, 3))
    clusters = {}
    centroids = {}
    for x in data:
        if int(x[-1]) not in clusters:
            # clusters[int(x[-1])] = [denormalize_imagexy(x[:-1], file_name)]
            clusters[int(x[-1])] = [x[:-1]]
        else:
            # clusters[int(x[-1])].append(denormalize_imagexy(x[:-1], file_name))
            clusters[int(x[-1])].append(x[:-1])

    for k, v in clusters.items():
        v = np.array(v)
        ave_c = np.mean(v[:, :3], axis=0)
        centroids[k] = ave_c
    RGB_recolored = data[:, :3]
    for i in range(RGB_recolored.shape[0]):
        RGB_recolored[i, :] = centroids[data[i, 5]]

    plt.imshow(RGB_recolored.reshape((50,50,3)))
    # plt.show()
    plt.savefig(file_name.split('.')[0] + '.png')


def denormalize_imagexy(imagexy, path):
    # data = np.loadtxt(path.split('.')[0][:-1]+'normalization.out')
    # diffs, mins = data[:5], data[-5:]
    #
    # imagexy /= [50, 50, 50, max(diffs[:3]), max(diffs[:3])]
    # imagexy += mins

    imagexy /= [1, 1, 1, 4, 4]
    return imagexy


def process_mnist():
    mnist = fetch_openml('mnist_784')
    image = mnist.data.to_numpy()
    for i in range(100):

        plt.imshow((image[i].reshape(28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        # plt.show()
        plt.savefig(f'D:\python projects\VisionClustering\plots\evaluation\integration_3\mnist\mnist{i}.png')


def load_mnist(file_name):
    data = np.loadtxt(file_name)
    clusters = {}
    for x in data:
        if int(x[-1]) not in clusters:
            clusters[int(x[-1])] = [x[:-1]]
        else:
            clusters[int(x[-1])].append(x[:-1])

    root = file_name.split('.')[0]
    for k, cs in clusters.items():
        for i, c in enumerate(cs):
            plt.imshow((c.reshape(28, 28) * 255), cmap=plt.cm.gray_r,
                       interpolation='nearest')
            # plt.show()
            plt.savefig(f'D:\python projects\VisionClustering\plots\evaluation\integration_4\\res\{k}_{i}.png')


def load_cifar(file_name):
    data = np.loadtxt(file_name)
    clusters = {}
    for x in data:
        if int(x[-1]) not in clusters:
            clusters[int(x[-1])] = [x[:-1]]
        else:
            clusters[int(x[-1])].append(x[:-1])

    root = file_name.split('.')[0]
    for k, cs in clusters.items():
        for i, c in enumerate(cs):
            plt.imshow((c.reshape(3, 32, 32).transpose(1, 2, 0) / 255))
            # plt.show()
            plt.savefig(f'D:\python projects\VisionClustering\plots\evaluation\integration_5\\cifar_res\{k}_{i}.png')


def get_curve_problem(file_name):
    image = plt.imread(file_name)
    xx = image[:, :, 0]
    blacks = np.where(xx == 0)
    points = np.column_stack((blacks[0], blacks[1]))
    targets = []
    for p in points:
        for t in targets:
            if np.linalg.norm(p-t) < 5:
                break
        else:
            targets.append(p)

    print()
    with open('benchmark/curve/fan.txt', "w+") as fw:
        for i in range(len(targets)):
            fw.write(f"{targets[i][0] / 10} {targets[i][1] / 10} {0}\n")


def box_plot():
    file_name = 'indices_for_boxplot.xlsx'
    sheet_name = 'nmi'

    # First, open the file to get the number of rows, excluding the header
    with pd.ExcelFile(file_name, engine='openpyxl') as xls:
        # Get the number of rows in the specified sheet
        total_rows = xls.book[sheet_name].max_row

    # Calculate the number of rows to read (excluding the last two)
    nrows_to_read = total_rows - 1  # Subtract 1 more because header is not counted

    # Now, read the Excel file with nrows parameter set
    df = pd.read_excel(file_name, sheet_name=sheet_name, header=0, index_col=0, nrows=nrows_to_read, engine='openpyxl')

    # Calculate Q1, Q2 (median), Q3, and 1.5*IQR and store in a DataFrame
    stats = {}
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q2 = df[col].median()
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        stats[col] = [round(Q1, 3), round(Q2, 3), round(Q3, 3), round(1.5*IQR, 3)]

    # Create a DataFrame for the statistics
    stats_df = pd.DataFrame(stats, index=['Q1', 'Median (Q2)', 'Q3', '1.5*IQR'])

    # Export the statistics to an Excel file
    stats_df.to_excel('boxplot_statistics_NMI.xlsx', engine='openpyxl')

    df.boxplot()

    plt.savefig('NMI', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    box_plot()