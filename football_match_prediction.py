from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    goals_for = points[9].values
    goals_against = points[10].values
    return goals_for, goals_against

def goal_difference(goals_for, goals_against):
    difference_list = []
    for i in range(0, len(goals_for)):
        difference = goals_for[i] - goals_against[i]
        difference_list.append(difference)

    return difference_list

def last_ten_games(difference_list):
    recent_games = []
    number_of_games = len(difference_list)
    for i in range(-10,0):
        goal_difference = difference_list[number_of_games + i ]
        recent_games.append(goal_difference)

    return recent_games


def match_result(Difference_list):
    win_tally = 0
    draw_tally = 0
    loss_tally = 0
    for i in range(0, len(Difference_list)):
        if Difference_list[i] >= 0:
            win_tally += 1
        if Difference_list[i] == 0:
            draw_tally += 1
        if Difference_list[i] <= 0:
            loss_tally += 1
    print(win_tally)
    print(draw_tally)
    print(loss_tally)


    return win_tally, draw_tally, loss_tally

def plt_match_results(win_tally, draw_tally, loss_tally):
    y = [win_tally, draw_tally, loss_tally]
    x = ['win', 'draw', 'loss']
    plt.title('Chelsea home match results from 2008 to 2016')
    plt.bar(x, y)
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
    

    return float(new_y_hat[0])


def prediction(prediction_home,prediction_away):
    if prediction_home - prediction_away >= 1:
        print('home win')
    if prediction_home - prediction_away <= -1:
        print('away win')
    else:
        print('draw')



home = r'C:\Users\hanlo\Documents\database.sqlite\chelsea_matches.csv'
goals_for, goals_against = load_points_from_file(home)
Difference_list = goal_difference(goals_for, goals_against)
#win_tally, draw_tally, loss_tally = match_result(Difference_list)
Recent_games = last_ten_games(Difference_list)
#plt_match_results(win_tally, draw_tally, loss_tally)
prediction_home =regression(Recent_games)


away = r'C:\Users\hanlo\Documents\database.sqlite\arsenal_matches.csv'
goals_for_away, goals_against_away = load_points_from_file(away)
difference_list_away = goal_difference(goals_for_away, goals_against_away)
recent_games_away = last_ten_games(difference_list_away)
prediction_away= regression(recent_games_away)


prediction(prediction_home,prediction_away)