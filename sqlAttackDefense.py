import sqlite3
import numpy as np
import xmltodict
import pandas as pd
import collections
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

def load_points_from_db(cur,id):
    cur.execute(f"SELECT possession,cross,foulcommit,shoton FROM Match WHERE {id}")
    fetch = np.array(cur.fetchall())
    possession = fetch[:,0]
    crosses = fetch[:,1]
    fouls = fetch[:,2]
    shots = fetch[:,3]
    return possession, crosses, fouls, shots


def average(stat):
    average_stat = sum(stat)/len(stat)
    std_stat = np.std(stat)
    return average_stat, std_stat


def convert_possession(recent_stats):
    dictArr = []
    for i in range(0, len(recent_stats)):
        dict_stat = xmltodict.parse(recent_stats[i])
        try:
            dict_stat = dict_stat['possession']['value']
            if type(dict_stat) is list:

                dict_stat = int(dict_stat[-1]['homepos'])

            elif type(dict_stat) is collections.OrderedDict:

                dict_stat = int(dict_stat['homepos'])

            dictArr.append(dict_stat)
        except:
            dictArr.append(0)
    return dictArr


def convert_stat(recent_stats,statType):
    dictArr = []
    for i in range(0, len(recent_stats)):
        dict_stat = xmltodict.parse(recent_stats[i])
        try:
            dict_stat = dict_stat[statType]['value']
            shotson = len(dict_stat)
            dictArr.append(shotson)
        except:
            dictArr.append(0)

    return dictArr


def regression(dict_stat):
    y = dict_stat
    X = np.arange(len(dict_stat)).reshape(-1, 1)
    X = np.hstack((np.ones(shape=(len(X), 1)), X))
    ls = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    y_hat = X.dot(ls)

    new_x = np.array([[1, 11]])
    new_y_hat = new_x.dot(ls)

    '''
    plt.scatter(X[:, 1], y, c='b', label='Data')

    X = np.concatenate((X, new_x))
    y_hat = np.concatenate((y_hat, new_y_hat))
    '''
    '''
    plt.plot(X[:, 1], y_hat, c='g', label='Model')
    plt.scatter(new_x[:, 1], new_y_hat, c='r', label='Prediction')

    plt.xlabel('Games')
    plt.ylabel('stat')
    plt.legend(loc='best')
    plt.show()
    '''

    return float(new_y_hat[0])


def get_attacking_ratings(average_crosses, average_shots, std_crosses, std_shots, cross_prediction, shot_prediction):

    shot_difference = shot_prediction - average_shots
    cross_difference = cross_prediction - average_crosses
    no_std_cross = cross_difference/std_crosses
    no_std_shots = shot_difference/std_shots

    attacking_rating = 1000 + no_std_shots*100 + no_std_cross*100

    return attacking_rating


def get_defensive_rating(average_possession, average_fouls, std_possession, std_fouls, possession_prediction, fouls_prediction):

    possession_difference = possession_prediction - average_possession
    foul_difference = fouls_prediction - average_fouls
    no_std_possession = possession_difference / std_possession
    no_std_fouls = foul_difference / std_fouls

    defensive_rating = 1000 + no_std_possession * 100 - no_std_fouls * 100

    return defensive_rating


def last_ten_stats(stat):
    recent_stats = []
    number_of_games = len(stat)
    if number_of_games <= 10:
        return stat
    for i in range(-10, 0):
        match_stat = stat[number_of_games + i]
        recent_stats.append(match_stat)
    return recent_stats


def pad(dict_stat,stat_average):
    if len(dict_stat) == 0:
        dict_stat= np.pad(dict_stat,(0,10),constant_values=stat_average)
    elif len(dict_stat) < 10:
        dict_stat= np.pad(dict_stat,(0,10-len(dict_stat)),mode='mean')
    return dict_stat


def attackDefenceRating(cur,team_api_id,average_possession,std_possession,average_crosses,std_crosses,average_fouls,
                        std_fouls,average_shots,std_shots):

    possession, crosses, fouls, shots = load_points_from_db(cur,f"home_team_api_id = {team_api_id}")

    possession = [pos for pos in possession if pos is not None]
    crosses = [cross for cross in crosses if cross is not None]
    fouls = [foul for foul in fouls if foul is not None]
    shots = [shot for shot in shots if shot is not None]

    recent_possession = last_ten_stats(possession)
    dict_possession = convert_possession(recent_possession)
    dict_possession = pad(dict_possession,average_possession)
    possession_prediction=regression(dict_possession)

    recent_shots = last_ten_stats(shots)
    dict_shots = convert_stat(recent_shots,'shoton')
    dict_shots = pad(dict_shots, average_shots)
    shots_prediction = regression(dict_shots)

    recent_fouls = last_ten_stats(fouls)
    dict_fouls = convert_stat(recent_fouls,'foulcommit')
    dict_fouls = pad(dict_fouls, average_fouls)
    fouls_prediction = regression(dict_fouls)

    recent_crosses = last_ten_stats(crosses)
    dict_crosses = convert_stat(recent_crosses,'cross')
    dict_crosses = pad(dict_crosses, average_crosses)
    cross_prediction = regression(dict_crosses)

    attacking_rating = get_attacking_ratings(average_crosses, average_shots, std_crosses, std_shots, cross_prediction, shots_prediction)

    defensive_rating = get_defensive_rating(average_possession, average_fouls, std_possession,std_fouls, possession_prediction, fouls_prediction)

    return attacking_rating, defensive_rating


def setTeamAttackDefence(cur):
    all_possession, all_crosses, all_fouls, all_shots = load_points_from_db(cur, "league_id = '1729'")

    dict_all_possession = convert_possession(all_possession)
    average_possession, std_possession = average(dict_all_possession)

    dict_all_crosses = convert_stat(all_crosses, 'cross')
    average_crosses, std_crosses = average(dict_all_crosses)

    dict_all_fouls = convert_stat(all_fouls, 'foulcommit')
    average_fouls, std_fouls = average(dict_all_fouls)

    dict_all_shots = convert_stat(all_shots, 'shoton')
    average_shots, std_shots = average(dict_all_shots)

    cur.execute("SELECT home_team_api_id from Match WHERE league_id = '1729'")
    home_teams = cur.fetchall()
    for team in tqdm(home_teams,desc = "Setting Team Attack and Defence Ratings"):
        team_id = team[0]

        attack,defence = attackDefenceRating(cur,team_id,average_possession,std_possession,average_crosses,std_crosses,
                                             average_fouls,std_fouls,average_shots,std_shots)
        print(attack,defence,team_id)
        cur.execute(f"UPDATE Team SET attack = {attack},defence = {defence} WHERE team_api_id = {team_id}")


if __name__ == '__main__':
    con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
    cur = con.cursor()

    #setTeamAttackDefence(cur)
    #con.commit()

    cur.execute("SELECT home_team_api_id from Match WHERE league_id = '1729'")
    team_ids = cur.fetchall()
    team_ids = str(tuple(set([id[0] for id in team_ids])))
    cur.execute(f"SELECT attack,defence,team_api_id FROM Team WHERE team_api_id IN {team_ids}")






