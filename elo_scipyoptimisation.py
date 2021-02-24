from elo2 import train
from elo2 import test
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
from scipy.optimize import dual_annealing
from scipy.optimize import differential_evolution



def func(x,*args):

    train(x[0],args[0],1,args[2],x[2],x[3],x[4])
    acc = test(args[1],x[1],args[2])
    return -acc


def callback(x,f,context):
    global count
    count+=1
    print("Iter %d" %count)
    print(-f)



con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
cur = con.cursor()
cur.execute("UPDATE Team SET elo=1000")
cur.execute(
        "SELECT date,home_team_api_id,away_team_api_id,home_team_goal,away_team_goal FROM Match WHERE season != '2015/2016'")
trainMatches = cur.fetchall()
trainMatches.sort(key=lambda x: x[1])
cur.execute(
        "SELECT date,home_team_api_id,away_team_api_id,home_team_goal,away_team_goal FROM Match WHERE id > 25000")
testMatches = cur.fetchall()

args = (trainMatches,testMatches,cur)

count = 0
res = dual_annealing(func,bounds=[(1,300),(0,300),(1,3),(1,4),(1,5)],args=args,callback=callback)
#res = differential_evolution(func,bounds=[(1,300),(0,300),(1,3),(1,4),(1,5)],args=args,callback=callback)

print(res)