import sqlite3
import numpy as np
import pandas as pd
from elo2 import train,test
from scipy.optimize import dual_annealing



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

k = res[0]
mult2 = res[2]
mult3 = res[3]
mult4 = res[4]


cur.execute(
        "SELECT date,home_team_api_id,away_team_api_id,home_team_goal,away_team_goal FROM Match")
allMatches = cur.fetchall()
allMatches.sort(key=lambda x: x[1])
train(k,allMatches,1,cur,mult2,mult3,mult4)



cur.execute("SELECT home_team_api FROM Match")
home_teams = cur.fetchall()

cur.execute("SELECT away_team_api FROM Match")
away_teams = cur.fetchall()

for team in home_teams:
    cur.execute(f"SELECT elo from Team WHERE team_api_id = {team}")
    team_elo = cur.fetchall()
    cur.execute(f"UPDATE Match SET home_team_elo = {team_elo} WHERE home_team_api_id = {team}")

for team in away_teams:
    cur.execute(f"SELECT elo from Team where team_api_id = {team}")
    team_elo = cur.fetchall()
    cur.execute(f"UPDATE Match SET away_team_elo = {team_elo} WHERE away_team_api_id = {team}")





cur.execute(("SELECT season,home_team_api_id,away_team_api_id,home_team_goal,away_team_goal,home_team_elo,away_team_elo,winner FROM Match"))
match_data = cur.fetchall()



