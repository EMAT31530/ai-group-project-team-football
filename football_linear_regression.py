from math import *
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    goals_for = points[9].values
    goals_against = points[10].values
    return goals_for, goals_against

def goal_difference(goals_for, goals_against):
    Difference_list = []
    for i in range(0, len(goals_for)):
        Difference = goals_for[i] - goals_against[i]
        Difference_list.append(Difference)

    return Difference_list

def last_ten_games(Difference_list):
    Recent_games = []
    Number_of_games = len(Difference_list)
    for i in range(-10,0):
        Goal_difference = Difference_list[Number_of_games + i ]
        Recent_games.append(Goal_difference)

    return Recent_games


def plt_match_results(Recent_games):
    y = [Recent_games]
    x = [1,2,3,4,5,6,7,8,9,10]
    plt.title('Chelsea goal difference from last ten games')
    plt.scatter(x, y)
    plt.show()

def regression(Recent_games):
    y = np.array([[Recent_games[0]],[Recent_games[1]],[Recent_games[2]],[Recent_games[3]],[Recent_games[4]],\
                  [Recent_games[5]],[Recent_games[6]],[Recent_games[7]],[Recent_games[8]],[Recent_games[9]]])
    X =np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    X = np.hstack((np.ones(shape=(len(X), 1)), X))
    ls = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    y_hat = X.dot(ls)

    mse = np.mean((y.flatten() - y_hat) ** 2)

    new_x = np.array([[1, 11]])
    new_y_hat = new_x.dot(ls)

    plt.scatter(X[:, 1], y, c='b', label='Data')

    X = np.concatenate((X, new_x))
    y_hat = np.concatenate((y_hat, new_y_hat))

    plt.plot(X[:, 1], y_hat, c='g', label='Model')
    plt.scatter(new_x[:, 1], new_y_hat,c='r', label='Prediction')

    plt.xlabel('Games')
    plt.ylabel('Goal Difference')
    plt.legend(loc='best')
    plt.show()


filename = r'C:\Users\hanlo\Documents\database.sqlite\arsenal_home_matches.csv'
goals_for, goals_against = load_points_from_file(filename)
Difference_list = goal_difference(goals_for, goals_against)
print(Difference_list)
Recent_games = last_ten_games(Difference_list)
print(Recent_games)

regression(Recent_games)