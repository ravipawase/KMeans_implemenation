import numpy as np
import matplotlib.pyplot as plt

# create two distinct clusters
cluster_center_xs = [200, 800]
cluster_center_ys = [200, 800]
sigmas = [100, 50]
no_of_points = [500, 1000]
colors = ['red', 'blue']
filenames = ['file1.txt', 'file2.txt']
#data_root_folder =


plt.plot(cluster_center_xs, cluster_center_ys, 'r.')
for i, _ in enumerate(cluster_center_xs):
    current_cluster_center_xs = np.random.normal(cluster_center_xs[i], sigmas[i], no_of_points[i])
    current_cluster_center_ys = np.random.normal(cluster_center_ys[i], sigmas[i], no_of_points[i])
    #print(len(current_cluster_center_xs), len(current_cluster_center_ys))
    plt.scatter(current_cluster_center_xs, current_cluster_center_ys,  c=colors[i], marker='+', s=2)

    current_cluster_center_xs = np.array(current_cluster_center_xs)
    #print(current_cluster_center_xs.shape)
    current_cluster_center_xs = current_cluster_center_xs[:, np.newaxis]
    #print(current_cluster_center_xs.shape)

    current_cluster_center_ys = np.array(current_cluster_center_ys)
    current_cluster_center_ys = current_cluster_center_ys[:, np.newaxis]
    composite_array = np.hstack((current_cluster_center_xs, current_cluster_center_ys))
    #print(composite_array.shape)
    np.savetxt('/home/ravindra/Desktop/Interrogating_clusters/create_data/2_clusters.txt', composite_array, fmt='%d', delimiter=',')

plt.savefig('two_clusters.png')


