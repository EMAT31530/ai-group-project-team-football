from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from sklearn import metrics
from tqdm import tqdm

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
    return np.random.choice(difference_list,10,replace=False)




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

    '''
    plt.plot(X[:, 1], y_hat, c='g', label='Model')
    plt.scatter(new_x[:, 1], new_y_hat,c='r', label='Prediction')

    plt.xlabel('Games')
    plt.ylabel('Goal Difference')
    plt.legend(loc='best')
    plt.show()
    '''

    return float(new_y_hat[0])


def prediction(prediction_home,prediction_away):
    if prediction_home - prediction_away >= 1:
        return 'home'
    if prediction_home - prediction_away <= -1:
        return 'away'
    else:
        return 'draw'

con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
cur = con.cursor()
cur.execute("SELECT home_team_api_id,away_team_api_id,home_team_goal,away_team_goal,winner FROM Match WHERE league_id = '1729'")
matches = cur.fetchall()

predArr = []
trueArr = []
for match in tqdm(matches):
    home_team_id = match[0]
    away_team_id = match[1]
    home_team_goal = match[2]
    away_team_goal = match[3]
    winner = match[4]

    if winner == home_team_id:
        trueArr.append('home')
    elif winner == away_team_id:
        trueArr.append('away')
    elif winner == 'draw':
        trueArr.append('draw')

    cur.execute(f"SELECT home_team_goal,away_team_goal FROM Match WHERE home_team_api_id = {home_team_id}")
    home_goals = np.array(cur.fetchall())
    cur.execute(f"SELECT away_team_goal,home_team_goal FROM Match WHERE away_team_api_id = {away_team_id}")
    away_goals = np.array(cur.fetchall())

    goals_for_home = home_goals[:,0]
    goals_against_home = home_goals[:,1]
    goals_for_away = away_goals[:,0]
    goals_against_away = away_goals[:,1]

    home_difference = goal_difference(goals_for_home,goals_against_home)
    away_difference = goal_difference(goals_for_away,goals_against_away)

    home_recent_games = last_ten_games(home_difference)
    away_recent_games = last_ten_games(away_difference)

    home_prediction = regression(home_recent_games)
    away_prediction = regression(away_recent_games)

    match_prediction = prediction(home_prediction,away_prediction)
    predArr.append(match_prediction)


acc = metrics.accuracy_score(trueArr,predArr)
print(acc)