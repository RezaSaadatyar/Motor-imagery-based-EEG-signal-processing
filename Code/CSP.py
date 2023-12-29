import numpy as np
from scipy import linalg


def common_spatial_pattern(data_train1, data_train2, m):
    Rh = 0
    Rf = 0
    for i in range(0, np.shape(data_train1)[0]):
        X1 = data_train1[i, :, :]
        for j in range(0, np.shape(X1)[0]):
            X1[j, :] = X1[j, :] - np.mean(X1[j, :])
            # X1 = X1 - np.mean(X1, axis=0)  # Normalize data1
        rh = X1.dot(X1.T) / np.sum(np.diag(X1.dot(X1.T)))
        Rh = Rh + rh
    # Rh = Rh / np.shape(data_train1)[0]
    rh = 0
    for i in range(0, np.shape(data_train2)[0]):
        X1 = data_train2[i, :, :]
        for j in range(0, np.shape(X1)[0]):
            X1[j, :] = X1[j, :] - np.mean(X1[j, :])
            # X1 = X1 - np.mean(X1, axis=0)  # Normalize data1
        rh = X1.dot(X1.T) / np.sum(np.diag(X1.dot(X1.T)))
        Rf = Rf + rh
    # Rf = Rf / np.shape(data_train2)[0]

    Eig_values, Eig_vectors = linalg.eig(Rh, Rf)   # Generalized eigen value decomposition
    Eig_values = Eig_values.real
    Eig_vectors = Eig_vectors.real
    Ind = np.argsort(-1*Eig_values)  # Sorted descending
    Eig_vectors = Eig_vectors[:, Ind]
    W = np.concatenate((Eig_vectors[:, 0:m], Eig_vectors[:, len(Eig_vectors)-m:len(Eig_vectors)]), axis=1)
    return W



