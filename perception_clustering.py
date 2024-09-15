from perception import Perception
from matplotlib import pyplot as plt
import numpy as np
import random
from load_data import read_data
import os
from clt_tools import create_distance_matrix, mkdir_p
random.seed(999)


def prepare_test_cases():
    test_cases = {
        '3blobs': 'data/3_blobs.txt',
        '3-spiral': 'data/3-spiral.txt',
        'jain': 'data/jain.txt',
        'compound': 'data/compound.txt',
        'flame': 'data/flame.txt',
        'aggregation': 'data/aggregation.txt',
        'pathbased': 'data/pathbased.txt',
        'impossible': 'data/impossible.txt',
        'chainlink': 'data/chainlink.txt',
        'atom': 'data/atom.txt',
    }
    return test_cases


def evaluation(cases_data):
    evaluation_root = 'plots/'
    version = '20240904'
    withline = False

    for k, case in cases_data.items():
        log_path = evaluation_root + version + '/' + k
        mkdir_p(log_path)
        # define the model
        model = Perception(log_path=log_path)

        # fit the model

        X, true_label = read_data(case, k, has_labels=True)
        # distance_matrix = create_distance_matrix(X)
        yhat, xhat = model.fit(X)

        # save the clustering results
        dir_path = f'results/labels/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = os.path.join(dir_path, f'{k}.txt')
        np.savetxt(file_path, yhat, fmt='%d')

        dimension = len(X[0])
        if dimension == 2 or dimension == 3:
            f = plt.figure()
            if dimension == 2:
                ax = f.add_subplot()
                ax.set_aspect('equal', adjustable='box')
            else:
                ax = f.add_subplot(projection='3d')

            for k, cluster in model.initial_clusters.items():
                if dimension == 2:
                    ax.scatter(model.X[cluster['data']][:, 0], model.X[cluster['data']][:, 1])
                    if withline:
                        for track in cluster['traces']:
                            ax.plot(model.X[list(track)][:, 0], model.X[list(track)][:, 1], 'k-')
                else:
                    ax.scatter(model.X[cluster['data']][:, 0], model.X[cluster['data']][:, 1],
                               model.X[cluster['data']][:, 2])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if dimension == 3:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

            plt.savefig(f'{log_path}/final.png')
            plt.clf()





if __name__ == '__main__':
    evaluation(prepare_test_cases())

