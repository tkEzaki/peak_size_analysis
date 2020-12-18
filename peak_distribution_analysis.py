import numpy as np
import random

def fill_surrounding(data):
    # Adds a single row/column to the top,bottom,left, and right of the picture.
    # The added sites have a value equal to the minimum value in the original data.

    minimum = np.min(data)
    data_filled = np.ones((data.shape[0]+2, data.shape[1]+2)) * minimum
    data_filled[1:data.shape[0]+1,1:data.shape[1]+1] = np.copy(data)
    return data_filled


def compute_peak_map(data):
    # Computes the peak map.
    # If the site [i][j] is a local peak, the size of the peak is
    # stored to peak_map[i][j], otherwise peak_map[i][j] = 0.

    minimum = np.min(data)
    peak_map = np.zeros(data.shape)
    for i in range(len(data)-2):
        for j in range(len(data[0])-2):
            if data[i+1][j+1] > minimum:# minimum value represents the background.
                flag = True
                search_i = i+1
                search_j = j+1
                while flag:
                    max_index = np.argmax([data[search_i][search_j],data[search_i-1][search_j],data[search_i][search_j-1],data[search_i+1][search_j],data[search_i][search_j+1]])
                    if max_index == 0:#local max found
                        if (search_i - (i + 1)) ** 2 + (search_j - (j + 1)) ** 2 > 0:
                            peak_map[search_i][search_j] += 1
                        flag = False
                    elif max_index == 1:
                        search_i -= 1
                    elif max_index == 2:
                        search_j -=1
                    elif max_index == 3:
                        search_i += 1
                    elif max_index == 4:
                        search_j +=1
    return peak_map


def peak_distribution(peak_data,peak_max=60):
    # Computes peak_dist[i], the number of peaks having peak size i.

    peak_dist = np.zeros(peak_max)
    for peak_line in peak_data:
        for peak in peak_line:
            if peak >0:
                peak_dist[int(peak)] += 1

    return peak_dist


def expected_value(prob):
    e = 0
    for i in range(len(prob)):
        e += i * prob[i]
    return e


def randomize_data(data):
    minimum = np.min(data)
    values = set(np.ravel(data.tolist()))
    values.remove(minimum)
    value_list = list(values)
    random.shuffle(value_list)

    surrogate_data = np.zeros(data.shape)
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == minimum:
                surrogate_data[i][j] = minimum
            else:
                surrogate_data[i][j] = random.choice(value_list)
    return surrogate_data


def average_list(dist_list):
    case_num = len(dist_list)
    bin_num = len(dist_list[0])
    average_dist = np.zeros(bin_num)
    for i in range(case_num):
        for bin in range(bin_num):
            average_dist[bin] += dist_list[i][bin]/case_num
    return average_dist


def null_peak_distribution(data, n=100):
    dist_list = []
    for i in range(n):
        surrogate_data = randomize_data(data)
        peak_map = compute_peak_map(surrogate_data)
        dist_list.append(peak_distribution(peak_map))
    return average_list(dist_list)


if __name__ == '__main__':
    # Read data and compute the average peak size
    data = fill_surrounding(np.loadtxt('sample_data.txt'))
    peak_map = compute_peak_map(data)
    peak_dist = peak_distribution(peak_map)
    prob_dist = peak_dist / np.sum(peak_dist)
    average_peak_size = expected_value(prob_dist)

    # Generate surrogate data for computing a null distribution
    peak_dist_surrogate = null_peak_distribution(data,n=100)
    prob_dist_surrogate = peak_dist_surrogate / np.sum(peak_dist_surrogate)
    average_peak_size_surrogate = expected_value(prob_dist_surrogate)

    # Normalize the result
    normalized_average_peak_size = average_peak_size / average_peak_size_surrogate
    print(normalized_average_peak_size)