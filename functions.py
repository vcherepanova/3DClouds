
import numpy as np
import torch
dtype = torch.cuda.FloatTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def find_neighbors(cloud, k_neighbors, window, n_fixed):
    # function finds neighbors of the points in cloud and returns 
    # dictionary with indexes of neighbor points and dictionary 
    # of distances to them. 
    #
    # INPUT: 
    #
    # cloud:        array of size  n_points*3
    #               array of points coordinates in the form (t,x,y)
    # k_neighbors:  int
    #               number of neighbor points to find 
    # window:       int
    #               size of window where to look for neighbors
    # n_fixed:       int
    #               number of fixed points
    ##################################################
    neighbors_dict = {}
    distance_dict = {}
    for index, point in enumerate(cloud[n_fixed:]):
        # cloud slice where look for neighbors
        index = index+n_fixed
        shift = max(0,index-window)
        cloud_slice = cloud[shift:index]
        #compute distances between points
        dist = np.sum((point - cloud_slice)**2, axis=1)
        dist_sorted = np.argsort(dist)
        neighbors = dist_sorted[1:k_neighbors+1]+shift
        neighbors_dict[index] = list(neighbors)
        distance_dict[index] = dist[neighbors-shift]
    return (neighbors_dict, distance_dict)




def loss(cloud, x, neighbors, distances, n_fixed, lam):
    # function computes the objective function for minimization 
    #
    # INPUT: 
    #
    # cloud:        array of size  n_points*3
    #               array of points coordinates in the form (t,x,y)
    # x:            torch vector variable 
    #               points to optimize
    # neighbors:    dictionary
    #               indexes of neighbor points
    # distances:    dictionary
    #               distances to neighbor points
    # n_fixed:      int
    #               number of fixed points
    # lam:          float
    #               regularization parameter
    #################################################
    f_x = -torch.sum((x[:,0]))
    for index, point in enumerate(cloud[n_fixed:]):
        index_shifted = index+n_fixed
        neighbor_ind = neighbors[index_shifted]
        dist = torch.from_numpy(distances[index_shifted]).to(device)
        # for each neighbor point
        for j in range(len(dist)):
            neighbor = neighbor_ind[j]
            # if the neighbor is a fixed point then constraint is the distance to it
            if neighbor < n_fixed:
                fixed_neighbor = torch.from_numpy(cloud[neighbor]).to(device)
                true_dist = torch.sum((x[index]-fixed_neighbor)**2)
                f_x += lam*(true_dist-dist[j])**2
            # if the neighbor is a moving point then constraint is the variable distance
            else:
                neighbor_shifted = neighbor - n_fixed
                true_dist = torch.sum((x[index]-x[neighbor_shifted])**2)
                f_x += lam*(true_dist-dist[j])**2
            
    return(f_x)





def check_error(cloud, x_val, neighbors, distances, n_fixed):
    # check error for pytorch approach
    error = 0
    for index, point in enumerate(cloud[n_fixed:]):
        index_shifted = index+n_fixed
        neighbor_ind = neighbors[index_shifted]
        dist = distances[index_shifted]
        # for each neighbor point
        for j in range(len(dist)):
            neighbor = neighbor_ind[j]
            if neighbor < n_fixed:
                fixed_neighbor = cloud[neighbor]
                true_dist = np.sum((x_val[index]-fixed_neighbor)**2)
                error += (true_dist-dist[j])**2
            else:
                neighbor_shifted = neighbor - n_fixed
                true_dist = np.sum((x_val[index]-x_val[neighbor_shifted])**2)
                error += (true_dist-dist[j])**2
            
    return(error)



def check_error_2(cloud, x_val, neighbors, distances, n_fixed):
    # check error for non-linear program 
    error = 0
    for index, point in enumerate(cloud[n_fixed:]):
        index_shifted = index+n_fixed
        neighbor_ind = neighbors[index_shifted]
        dist = distances[index_shifted]
        # for each neighbor point
        for j in range(len(dist)):
            neighbor = neighbor_ind[j]
            if neighbor < n_fixed:
                fixed_neighbor = cloud[neighbor]
                true_dist = np.sum((x_val[index]-fixed_neighbor)**2)
                if (true_dist-dist[j])>0:
                    error += 1
            else:
                neighbor_shifted = neighbor - n_fixed
                true_dist = np.sum((x_val[index]-x_val[neighbor_shifted])**2)
                if (true_dist-dist[j])>0:
                    error += 1
            
    return(error)

