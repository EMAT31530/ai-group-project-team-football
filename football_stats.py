import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xmltodict
from collections2 import OrderedDict


def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    possession = points[84].values
    shots = points[78].values

    return possession, shots

def last_ten_stats(stat):
    recent_stats = []
    number_of_games = len(stat)
    for i in range(-10, 0):
        match_stat = stat[number_of_games + i]
        recent_stats.append(match_stat)

    return recent_stats

def convert_possession(recent_stats):
    dict = []
    for i in range(0, len(recent_stats)):
        dict_stat = xmltodict.parse(recent_stats[i])
        dict_stat = dict_stat['possession']['value']
        length = len(dict_stat)
        dict_stat = dict_stat[length-1]
        dict_stat = int(dict_stat['homepos'])
        dict.append(dict_stat)

    return dict

def convert_shots(recent_stats):
    dict = []
    for i in range(0, len(recent_stats)):
        dict_stat = xmltodict.parse(recent_stats[i])
        dict_stat = dict_stat['shoton']['value']
        shotson = len(dict_stat)
        dict.append(shotson)

    return dict

def regression(dict_stat):
    y = dict_stat
    X = np.arange(len(dict_stat)).reshape(-1, 1)
    X = np.hstack((np.ones(shape=(len(X), 1)), X))
    ls = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    y_hat = X.dot(ls)

    new_x = np.array([[1, 11]])
    new_y_hat = new_x.dot(ls)

    plt.scatter(X[:, 1], y, c='b', label='Data')

    X = np.concatenate((X, new_x))
    y_hat = np.concatenate((y_hat, new_y_hat))

    plt.plot(X[:, 1], y_hat, c='g', label='Model')
    plt.scatter(new_x[:, 1], new_y_hat, c='r', label='Prediction')

    plt.xlabel('Games')
    plt.ylabel('stat')
    plt.legend(loc='best')
    plt.show()

    return float(new_y_hat[0])





filename = r'C:\Users\ollie\OneDrive\Documents\Football Data\chelsea_home.csv'
possession, shots = load_points_from_file(filename)

recent_possession = last_ten_stats(possession)
dict_possession = convert_possession(recent_possession)
possession_prediction=regression(dict_possession)

recent_shots = last_ten_stats(shots)
dict_shots = convert_shots(recent_shots)
shots_prediction = regression(dict_shots)

print(dict_possession)





