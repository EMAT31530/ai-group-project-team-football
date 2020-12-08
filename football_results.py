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
    Difference_list = []
    for i in range(0, len(goals_for)):
        Difference = goals_for[i] - goals_against[i]
        Difference_list.append(Difference)

    return Difference_list


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


filename = r'C:\Users\hanlo\Documents\database.sqlite\chelsea_matches.csv'
goals_for, goals_against = load_points_from_file(filename)
Difference_list = goal_difference(goals_for, goals_against)
win_tally, draw_tally, loss_tally = match_result(Difference_list)
plt_match_results(win_tally, draw_tally, loss_tally)