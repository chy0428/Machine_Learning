import numpy as np
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt
import scipy.misc
import seaborn as sns
import imageio

def load_data(data):
    with open(data, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
   
    return train_set, valid_set, test_set
    
    
if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl')

    train_x, train_y = train_set
    test_x, test_y = test_set
    
    # select data of class 3 or 9
    train_x = train_x[np.where((train_y == 3) | (train_y == 9))]
    train_y = train_y[np.where((train_y == 3) | (train_y == 9))]

    print(train_x.shape)
    print(train_y.shape)
    
 
# To find out the distribution of data

sns.kdeplot(np.mean(train_x[train_y==3], axis=0), color='red', label='3')
sns.kdeplot(np.mean(train_x[train_y==9], axis=0), color='blue', label='9')

x, height = np.unique(train_y.reshape(-1), return_counts=True) 
labels = ["3 (#%i)"%height[0], "9 (#%i)"%height[1]] 

fig = plt.gcf()
fig.savefig('graph.png', dpi=300, format='png', bbox_inches="tight", facecolor="white")

colors = ['red', 'blue']
plt.pie(height, labels=labels, autopct='%1.1f%%', startangle=90, colors = colors)

fig = plt.gcf()
fig.savefig('graph_pie.png', dpi=300, format='png', bbox_inches="tight", facecolor="white")

def kmeans(k, data, no_iter=100, optimized=True):
    total_loss = []
    
    # Initial point random initialization
    cluster_centers = np.random.random([k, data.shape[-1]])
    
    # Calculate the distance of each point to all cluster centers
    dist_vec = np.zeros([k, data.shape[0]])
    
    cur = 0
    while (cur < no_iter):
        for idx, center in enumerate(cluster_centers):
            dist_vec[idx] = np.sum(np.square(np.subtract(np.broadcast_to(center, data.shape), data)), axis=1) 
        # argmin center
        labels = np.argmin(dist_vec, axis=0)
        loss = 0
        for idx in range(k):
            if data[labels == idx].shape[0] == 0:
                cluster_centers = np.random.random([k, data.shape[-1]])
                cur = -1
                break
            # Calculate Loss 
            loss += np.sum(dist_vec[idx][labels == idx])
            # Update cluster centers
            cluster_centers[idx] = np.average(data[labels == idx], axis=0) 
        if cur >= 0:
            total_loss.append(loss)
        if optimized and cur > 1 and (total_loss[-1] == total_loss[-2]):
            break
        cur += 1
        
    print('Iterations: {}'.format(len(total_loss)))
    
    return cluster_centers, labels, total_loss
    
    
    
'''
for igenspaces
'''

def make_eigenvector(data):
    cov_mat = data.T.dot(data)/data.shape[0]
    m_x, m_y = data.T.mean(1).reshape((-1,1)), data.mean(0).reshape((-1,1))
    _cov = cov_mat-m_x.dot(m_y.T)
    eig_val, eig_vec = np.linalg.eig(_cov)
    
    return eig_vec

def To_eigenvector(tar, data, n):
    eig_vec = make_eigenvector(data)
    eigen_vec = np.zeros([tar.shape[0], n])
    for i in range(tar.shape[0]):
        eigen_vec[i] = tar[i].dot(eig_vec.T[:n].T)
        
    return eigen_vec
    
'''
Calculate and Plot raw image loss 
'''

for k in [2, 3, 5, 10]:
    cluster, labels, loss = kmeans(k, train_x, no_iter=50)
    for i in range(k):
        print('Loss: {}'.format(loss), end = '\n')
        file = 'Raw_image/k{}_cluster{}.png'.format(k, i+1)
        imageio.imwrite(file, cluster[i].reshape([28, 28]))
        
    fig = plt.figure()
    axe = fig.add_axes([1, 1, 1, 1])
    
    plt.plot(loss)
    # print('Final Loss: {}'.format(loss), end = '\n')
    plt.savefig('Raw_image/k{}_loss_plot.png'.format(k), bbox_inches='tight')
    
'''
Calculate and Plot raw image loss 
'''

for k in [2, 3, 5, 10]:
    cluster, labels, loss = kmeans(k, train_x, no_iter=50)
    for i in range(k):
        print('Loss: {}'.format(loss), end = '\n')
        file = 'Raw_image/k{}_cluster{}.png'.format(k, i+1)
        imageio.imwrite(file, cluster[i].reshape([28, 28]))
        
    fig = plt.figure()
    axe = fig.add_axes([1, 1, 1, 1])
    
    plt.plot(loss)
    # print('Final Loss: {}'.format(loss), end = '\n')
    plt.savefig('Raw_image/k{}_loss_plot.png'.format(k), bbox_inches='tight')
    
    
'''
Plot raw image
'''
data = To_eigenvector(train_x, train_x, 2)
for k in [2, 3, 5, 10]:
    cluster, labels, loss = kmeans(k, train_x, no_iter=50)
    colors = ['r', 'b', 'g', 'm', 'pink', 'y', 'k', 'olive', 'plum', 'tomato']
    
    for i in range(k):
        c = colors[i]
        plt.plot(data[labels == i][:, 0], data[labels == i][:, 1], '.', c=c)
        #plt.plot(cluster[i, 0], cluster[i, 1], 'ko')

    plt.savefig('Raw_image/k{}_distribution_plot.png'.format(k), bbox_inches='tight')
    
for eig in [2]:
    data = To_eigenvector(train_x, train_x, eig)
    
    for k in [2, 3, 5, 10]:
        cluster, labels, loss = kmeans(k, data, no_iter=50)
        fig = plt.figure  
        #labels = ['+', 'x']
        colors = ['r', 'b', 'g', 'm', 'pink','y', 'k', 'olive', 'plum', 'tomato']

        for i in range(k):
            c = colors[i]
            plt.plot(data[labels == i][:, 0], data[labels == i][:, 1], '.', c=c)
            plt.plot(cluster[i, 0], cluster[i, 1], 'ko')
            
        plt.savefig('Eigen_image/Eig{}_k{}_distribution_plot.png'.format(eig, k), bbox_inches='tight')
        
 for eig in [5, 10]:
    data = To_eigenvector(train_x, train_x, eig)
    pca_data = To_eigenvector(data, data, 2)
    
    for k in [2,3,5,10]:
        cluster, labels, loss = kmeans(k, data, no_iter=50)
        pca_cluster = To_eigenvector(cluster, data, 2)
        fig = plt.figure
        colors = ['r', 'b', 'g', 'm', 'pink', 'y', 'k', 'olive', 'plum', 'tomato']
             
        for i in range(k):
            c = colors[i]
            plt.plot(pca_data[labels == i][:,0], pca_data[labels == i][:,1], '.', c=c)
            plt.plot(pca_cluster[i,0], pca_cluster[i,1], 'ko')
          
        plt.savefig('Eigen_image/eig{}_k{}_distribution_plot.png'.format(eig,k), bbox_inches='tight')
        plt.clf()
