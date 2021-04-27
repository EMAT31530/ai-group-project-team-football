import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xmltodict



def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    goals_for = points[9].values
    goals_against = points[10].values
    possession = points[84].values
    crosses = points[82].values
    fouls = points[80].values
    shots = points[78].values

    return goals_for, goals_against,possession, crosses, fouls, shots

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


def convert_crosses(recent_stats):
    dict = []
    for i in range(0, len(recent_stats)):
        dict_stat = xmltodict.parse(recent_stats[i])
        dict_stat = dict_stat['cross']['value']
        crosses = len(dict_stat)
        dict.append(crosses)

    return dict

def convert_fouls(recent_stats):
    dict = []
    for i in range(0, len(recent_stats)):
        dict_stat = xmltodict.parse(recent_stats[i])
        dict_stat = dict_stat['foulcommit']['value']
        crosses = len(dict_stat)
        dict.append(crosses)

    return dict

def convert_shots(recent_stats):
    dict = []
    for i in range(0, len(recent_stats)):
        dict_stat = xmltodict.parse(recent_stats[i])
        dict_stat = dict_stat['shoton']['value']
        shotson = len(dict_stat)
        dict.append(shotson)

    return dict

def average(stat):
    average_stat = sum(stat)/len(stat)
    std_stat = np.std(stat)

    return average_stat, std_stat


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

def attacking_ratings(average_crosses, average_shots, std_crosses, std_shots, cross_prediction, shot_prediction):

    shot_difference = shot_prediction - average_shots
    cross_difference = cross_prediction - average_crosses
    no_std_cross = cross_difference/std_crosses
    no_std_shots = shot_difference/std_shots

    attacking_rating = 1000 + no_std_shots*100 + no_std_cross*100

    return attacking_rating

def defensive_rating(average_possession, average_fouls, std_possession, std_fouls, possession_prediction, fouls_prediction):

    possession_difference = possession_prediction - average_possession
    foul_difference = fouls_prediction - average_fouls
    no_std_possession = possession_difference / std_possession
    no_std_fouls = foul_difference / std_fouls

    defensive_rating = 1000 + no_std_possession * 100 - no_std_fouls * 100

    return defensive_rating






matches = r'C:\Users\ollie\OneDrive\Documents\Football Data\premier_league_matches.csv'
all_goals_for, all_goals_against, all_possession, all_crosses, all_fouls, all_shots = load_points_from_file(matches)
average_goals = average(all_goals_for)
average_conceded = average(all_goals_against)

dict_all_possession = convert_possession(all_possession)
average_possession, std_possession = average(dict_all_possession)


dict_all_crosses = convert_crosses(all_crosses)
average_crosses, std_crosses = average(dict_all_crosses)

dict_all_fouls = convert_fouls(all_fouls)
average_fouls, std_fouls =average(dict_all_fouls)

dict_all_shots = convert_shots(all_shots)
average_shots, std_shots = average(dict_all_shots)



home = r'C:\Users\ollie\OneDrive\Documents\Football Data\chelsea_home.csv'
goals_for, goals_against, possession, crosses, fouls, shots = load_points_from_file(home)

recent_goals_for = last_ten_stats(goals_for)


recent_goals_against = last_ten_stats(goals_against)


recent_possession = last_ten_stats(possession)
dict_possession = convert_possession(recent_possession)
possession_prediction=regression(dict_possession)

recent_shots = last_ten_stats(shots)
dict_shots = convert_shots(recent_shots)
shots_prediction = regression(dict_shots)

recent_fouls = last_ten_stats(fouls)
dict_fouls = convert_fouls(recent_fouls)
fouls_prediction = regression(dict_fouls)

recent_crosses = last_ten_stats(crosses)
dict_crosses = convert_crosses(recent_crosses)
cross_prediction = regression(dict_crosses)


attacking_rating = attacking_ratings(average_crosses, average_shots, std_crosses, std_shots, cross_prediction, shots_prediction)
print(attacking_rating)
defensive_rating = defensive_rating(average_possession, average_fouls, std_possession,std_fouls, possession_prediction, fouls_prediction)
print(defensive_rating)



