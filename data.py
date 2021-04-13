
import numpy as np

from scipy.stats import median_abs_deviation

def label_from_hist(data, n, train_indices=None, mode=1):
    """
    data : np.array, each row corresponds to a variable
    """
    labelled_data = []
    for row in data:
        if train_indices is not None:
            row_train = row[train_indices]
        else:
            row_train = row
        mad = median_abs_deviation(row_train[~np.isnan(row_train)])
        if mode == 1:
            div = [-mad * i / n for i in range(n, 0, -1)] + [0] + [mad * i / n for i in range(1, n+1)]
        elif mode == 2:
            div = [-mad * i for i in range(n, 0, -1)] + [0] + [mad * i for i in range(1, n+1)]
        labels = len(div) * np.ones(row.shape)
        for i, x in enumerate(row):
            if np.isnan(x):
                labels[i] = np.nan
            else:
                for j, v in enumerate(div):
                    if x < v:
                        labels[i] = j
                        break
        labelled_data.append(labels)
    return np.array(labelled_data)
