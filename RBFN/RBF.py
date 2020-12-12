import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import scipy.misc
from sklearn.metrics import accuracy_score

class RBFN(object):
    def __init__(self, hidden_layer, variance=1.0):
        self.hidden_layer = hidden_layer
        self.variance = variance
        self.centers = None
        self.weights = None
        
    def gaussian(self, center, data_point):
        return np.exp(-self.variance*np.linalg.norm(center-data_point)**2)
    
    # basis for train
    def create_kernel_matrix(self, X):
        W = np.zeros((len(X), self.hidden_layer))
        for i, data_point in enumerate(X):
            for j, center in enumerate(self.centers):
                W[i, j] = self.gaussian(center, data_point)
        return W
    
    def find_centers(self, X):
        # for FA Center
#         random_args = np.random.choice(len(X), self.hidden_layer)
#         centers = X[random_args]
        
        km = KMeans(n_clusters=self.hidden_layer, max_iter = 100).fit(X)
        centers = km.cluster_centers_
        #print(k_centers)
        return centers
    
    def fit(self, X, Y):
        self.centers = self.find_centers(X)
        W = self.create_kernel_matrix(X)
        self.weights = np.dot(np.linalg.pinv(W), Y)
        
    def predict(self, X):
        W = self.create_kernel_matrix(X)
        predictions = np.dot(W, self.weights)
        
        return predictions, self.centers


def load_data(dataset):
        f = open(dataset, 'r')
        rows = []
        for line in f:
            row = line.split()
            row[0] = float(row[0])
            row[1] = float(row[1])
            row[2] = int(row[2])
            rows.append(row)
            
        train_x = []
        train_y = []
        for x, y, z in rows:
            train_x.append((x, y))
            train_y.append(z)
            
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        
        return train_x, train_y
    
def load_data_2(dataset):
    f = open(dataset, 'r')
    rows = []

    for line in f:
        row = line.split()
        row[0] = float(row[0])
        row[1] = float(row[1])
        rows.append(row)

    train_x = []
    train_y = []

    for x, y in rows:
        train_x.append(x)
        train_y.append(y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    return train_x, train_y

def mean_squared_error(predict, label):
    #     error = 0
    #     for i in range(len(label)):
    #         error += np.linalg.norm(label-predict)**2
    #     error = error/len(label)
    #     return error  
    return ((label-predict)**2).mean(axis=None)
        
