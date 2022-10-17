import os

import numpy as np

from deeplearning.utils.plt_utils import heatmap


def replot_conf(datafile, pltfile = None, label2name: dict = None, tick_seq = None):
    matrix = np.loadtxt(datafile, int, delimiter = ",")
    if tick_seq is not None:
        assert label2name is not None, "label2name can not be nan if tick_seq is not nan"
        names = list(label2name.values())
        new_matrix = np.zeros_like(matrix)
        num_classes = len(names)
        assert len(tick_seq) == num_classes
        new_names = []
        for i in range(num_classes):
            for j in range(num_classes):
                new_matrix[i, j] = matrix[tick_seq[i], tick_seq[j]]
            new_names.append(names[tick_seq[i]])
        names = new_names
        matrix = new_matrix
    elif label2name is not None:
        names = list(label2name.values())
    else:
        names = "auto"
    if pltfile is None:
        pltfile = os.path.splitext(datafile)[0] + ".png"
    heatmap(matrix, pltfile, labels = names)


if __name__ == '__main__':
    name2label = {'0': 0, '0-50': 1, '100': 2, '50-75': 3, '75-99': 4}
    keys = list(name2label.values())
    values = list(name2label.keys())
    label2name = dict(zip(keys, values))

    new_name2label = {'0': 0, '0-50': 1, '50-75': 3, '75-99': 4, '100': 2}
    new_keys = list(name2label.values())
    new_values = list(name2label.keys())
    new_label2name = dict(zip(keys, values))
    tick_seq = [0, 1, 3, 4, 2]
    dir = os.path.join("../../results/ffr/Record_2022-10-10-08_27_27")
    for root, dirs, files in os.walk(dir):
        for file in files:
            if "confusion" not in file or not file.endswith(".csv"):
                continue
            replot_conf(os.path.join(root, file), label2name = label2name, tick_seq = tick_seq)
