import pandas as pd
import numpy as np
from utilities import calculate_distance
import os


class Kmeans:

    def __init__(self, n_clusters=3, n_init=3, max_iter=1000, tol=0.01, random_state=100):
        """
        Create Kmeans object from given parameters
        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.data = ""
        self.cluster_centers = ""
        self.cluster_distances = ""
        self.cluster_membership = ""

    def get_data(self, data):
        """

        :param data:
        :return:
        """
        # check X is an dataframe
        if not isinstance(data, pd.DataFrame):
            raise ValueError('Expected a  Dataframe, something else received')
        self.data = data.values
        print("Getting the input data")
        print("The input data shape is", self.data.shape)

    def initialize_cluster_centers_random_between(self):
        """
        Initialize cluster centers randomly

        """
        # find minimum and maximum in all dimensions of the data
        no_of_dimensions = self.data.shape[1]
        dim_mins = []
        dim_maxs = []
        for dimension in range(no_of_dimensions):
            column_data = np.array(self.data[:, dimension])
            dim_mins.append(column_data.min())
            dim_maxs.append(column_data.max())

        # select the randomized centers for all dimensions
        dim_spread = zip(dim_mins, dim_maxs)
        # create a 2d numpy array of size rows, cols : no_of_clusters, no_of dimesions/features
        print(self.n_clusters)
        print(no_of_dimensions)

        cluster_centers = np.zeros((int(self.n_clusters), no_of_dimensions))
        print(cluster_centers.shape)

        # iterate over dimensions
        for i, spread in enumerate(dim_spread):
            dim_centers = []
            # iterate over no. of clusters
            for _ in range(self.n_clusters):
                dim_centers.append(np.random.uniform(spread[0], spread[1]))
            print(np.array(dim_centers).shape)
            cluster_centers[:, i] = np.array(dim_centers)

        print(cluster_centers)
        # return cluster_centers
        self.cluster_centers = cluster_centers

    def initialize_cluster_centers_subset(self):
        """

        :return:
        """
        print("Initiating the cluster centers")
        for i in range(1000):
            indices = np.random.randint(0, self.data.shape[0]-1, self.n_clusters)
            if len(indices) == len(np.unique(indices)):
                break
        self.cluster_centers = self.data[(indices), :]
        print("The cluster centers shape is ", self.cluster_centers.shape)
        print("The initiated cluster centers are\n", self.cluster_centers)

    def calculate_distance_from_cluster_centers(self):
        """

        :return:
        """
        print("Calculating the distance of each point from each current cluster centers")
        cluster_distances = np.zeros((self.data.shape[0], self.n_clusters))
        #print(self.data.shape)
        #print("#" * 5)
        #print(cluster_distances.shape)

        for i, datapoint in enumerate(self.data):
            for j, cluster_center in enumerate(self.cluster_centers):
                cluster_distances[i, j] = calculate_distance(datapoint, cluster_center)
        self.cluster_distances = cluster_distances
        print("The shape of distance array is ", cluster_distances.shape)
        print("The distances for each point from cluster centers are\n", cluster_distances)

    def assign_cluster(self):
        """

        :return:
        """
        print("Assigning cluster to each point based on the distances they are from each cluster centers")
        self.cluster_membership = np.array(range(self.data.shape[0]))
        for i, row in enumerate(self.cluster_distances):
            self.cluster_membership[i] = np.argmin(row)
        print("The cluster membership is as follows\n", self.cluster_membership)

    def recalculate_cluster_centers_and_delta(self):
        """

        :return:
        """
        print("Recalculating the cluster centers and checking the deviations from last iterations")
        new_cluster_centers = []
        for cluster_no in range(self.n_clusters):
            current_cluster_candidates_index = np.argwhere(self.cluster_membership == cluster_no)
            current_cluster_candidates = self.data[tuple(current_cluster_candidates_index), :]
            #print("#"*50)
            #print("No of current_cluster_candidates is", len(current_cluster_candidates))
            #print(type(current_cluster_candidates))
            new_cluster_center = np.average(current_cluster_candidates, axis=0)
            new_cluster_centers.append(list(new_cluster_center.ravel()))

        new_cluster_centers = np.array(new_cluster_centers)
        print("The new cluster centers  are\n", new_cluster_centers)
        print("The old cluster centers  were\n", self.cluster_centers)

        # print("The shape of new cluster centers is ", new_cluster_centers.shape)
        if np.isnan(new_cluster_centers).any():
             raise ValueError('At least one cluster has NaN value, contact the developer')

        delta = np.round(np.abs(self.cluster_centers - new_cluster_centers))
        print("And the delta is \n", delta)
        self.cluster_centers = new_cluster_centers
        return delta

    def calculate_average_intra_cluster_spread(self):
        """

        :return:
        """
        print("Calculating the average intra cluster distance")
        cluster_spreads = []
        for cluster_no in range(self.n_clusters):
            current_cluster_candidates_index = np.argwhere(self.cluster_membership == cluster_no)
            current_cluster_candidates = self.data[tuple(current_cluster_candidates_index), :]
            current_cluster_spread = np.std(current_cluster_candidates)
            cluster_spreads.append(current_cluster_spread)
        print("The individual cluster spreads are \n", cluster_spreads)
        average_cluster_spread = np.average(np.array(cluster_spreads))
        return average_cluster_spread

    def fit(self, X, outfile):
        """
        Fit the data to given number of clusters

        X: pandas data frame with data
        :return:
        """
        self.get_data(X)
        no_of_features = self.data.shape[1]
        if no_of_features == 2:
            if os.path.exists(outfile):
                print("The logfile already exists, overwritting it")
                os.remove(outfile)
            outfile = open(outfile, 'ab')

        min_cluster_spread_cluster_centers = ""
        min_cluster_spread_cluster_membership = ""
        print("The requested 'number of clusters to fit'  are ", self.n_clusters)
        print("The requested 'number of initiations'  are ", self.n_init)
        initiation_average_cluster_spreads = []
        for init in range(self.n_init):
            print("Starting the Initialisation no. ", init)
            min_cluster_spread = np.inf
            self.initialize_cluster_centers_subset()
            # dump the data  before iteration starts
            if no_of_features == 2:
                data_length = self.data.shape[0]
                header = "initiation_no,iteration_no,x,y,cluster_association,cluster_center_X,cluster_center_y,delta_x,delta_y"
                fmt = ['%d', '%d', '%1.2f', '%1.2f', '%d', '%1.2f', '%1.2f', '%u', '%u']
                init_array = np.ones(data_length, dtype=np.int) * init
                init_array = init_array[:, np.newaxis].astype(int)
                pre_iteration_array = np.ones(data_length, dtype=np.int) * 0
                pre_iteration_array = pre_iteration_array[:, np.newaxis].astype(int)
                cluster_membership = np.ones(data_length, dtype=np.int) * 0
                cluster_membership = cluster_membership[:, np.newaxis].astype(int)
                cluster_centers = np.ones((data_length, 2)) * -1
                #pre_iteration_delta_x = str(np.ones((data_length, self.n_clusters)) * -1)
                #pre_iteration_delta_y = str(np.ones((data_length, self.n_clusters)) * -1)
                pre_iteration_delta_x = [str('-1 '*self.n_clusters)]
                pre_iteration_delta_x = np.array(pre_iteration_delta_x * data_length)
                pre_iteration_delta_x = pre_iteration_delta_x[:, np.newaxis]
                pre_iteration_delta_y = pre_iteration_delta_x
                if init == 0:
                    np.savetxt(outfile, np.hstack((init_array, pre_iteration_array, self.data, cluster_membership,
                                            cluster_centers, pre_iteration_delta_x, pre_iteration_delta_y)),
                                            fmt='%s', delimiter=',', header=header, comments='')
                else:
                    np.savetxt(outfile, np.hstack((init_array, pre_iteration_array, self.data, cluster_membership,
                                                   cluster_centers, pre_iteration_delta_x, pre_iteration_delta_y)),
                               fmt='%s', delimiter=',', comments='')

            for iteration in range(1, self.max_iter+1):
                print("Performing iteration no.", iteration)
                self.calculate_distance_from_cluster_centers()
                self.assign_cluster()

                #dump data to text file for later visualisation
                if no_of_features == 2:
                    iteration_array = np.ones(data_length, dtype=np.int) * iteration
                    iteration_array = iteration_array[:, np.newaxis].astype(int)
                    cluster_membership = self.cluster_membership[:, np.newaxis]
                    cluster_centers = self.cluster_centers[cluster_membership.ravel()]
                    if iteration == 1:
                        delta = np.ones((self.n_clusters, 2))*-1
                    delta_x = [str(ele) for ele in delta[:, 0].ravel()]
                    delta_x = np.array([" ".join(delta_x)] * data_length)
                    delta_x = delta_x[:, np.newaxis]
                    delta_y = [str(ele) for ele in delta[:, 1].ravel()]
                    delta_y = np.array([" ".join(delta_y)] * data_length)
                    delta_y = delta_y[:, np.newaxis]
                    np.savetxt(outfile, np.hstack((init_array, iteration_array, self.data, cluster_membership,
                                                   cluster_centers, delta_x, delta_y)), fmt='%s', delimiter=',',
                               comments='')
                delta = self.recalculate_cluster_centers_and_delta()
                if delta[delta > self.tol].size == 0:
                    print("Delta, distance between last and current cluster centers, is smaller than tolerance, "
                          "breaking the loop")
                    break
                else:
                    print(f"Delta, distance between last and current cluster centers is greater than tolerance, {self.tol}, "
                          "moving on to next iteration")
            ####
            average_cluster_spread = self.calculate_average_intra_cluster_spread()
            print("average cluster spread for given iteration is", average_cluster_spread)
            if average_cluster_spread < min_cluster_spread:
                min_cluster_spread = average_cluster_spread
                #cluster_centers = ""
                min_cluster_spread_cluster_centers = self.cluster_centers
                min_cluster_spread_cluster_membership = self.cluster_membership

            initiation_average_cluster_spreads.append(min_cluster_spread)
            self.cluster_centers = min_cluster_spread_cluster_centers
            self.cluster_membership = min_cluster_spread_cluster_membership
            print("Minimum cluster spread is", min_cluster_spread)

        print("Average cluster spreads for all initions are \n", initiation_average_cluster_spreads)
        print("Cluster membership is", self.cluster_membership)

        if no_of_features == 2:
            outfile.close()

    def predict(self, testdata):
        """

        :return:
        """
        if not isinstance(testdata, pd.DataFrame):
            raise ValueError('Expected a  Dataframe, something else recieved')
        testdata = testdata.values

        if not testdata.shape[1] == self.data.shape[1]:
            raise ValueError('No. of features in test data do no match the No. of features in training data')

        cluster_distances = np.zeros((testdata.shape[0], self.n_clusters))
        for i, datapoint in enumerate(testdata):
            for j, cluster_center in enumerate(self.cluster_centers):
                cluster_distances[i, j] = calculate_distance(datapoint, cluster_center)

        cluster_membership = np.array(range(testdata.shape[0]))
        for i, row in enumerate(cluster_distances):
            cluster_membership[i] = np.argmin(row)
        return cluster_membership
