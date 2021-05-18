import sqlite3
import pandas as pd
from elo2 import train
from elo2 import test
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
import sqlite3
from sklearn.model_selection import train_test_split

con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
cur = con.cursor()

cur.execute(
        "SELECT home_team_api_id,away_team_api_id,home_team_goal,away_team_goal,winner FROM Match WHERE league_id = '1729'")
matches = cur.fetchall()
print(len(matches))


'''
cur.execute(("SELECT winner,season,home_team_api_id,away_team_api_id FROM Match"))
match_data = cur.fetchall()
matches_df = pd.DataFrame(match_data,columns=["winner","season","home_team","away_team"])
print(matches_df)
'''
'''
home_elos = [2000,1000,1000]
ids = (1,2,3)
cur.execute("SELECT home_team_api_id FROM Match WHERE id IN ()",ids)
home_ids = [item[0] for item in cur.fetchall()]
print(home_ids)
n = len(home_ids)
s = "?,"*n
s = s[:-1]

cur.execute(f"UPDATE Match SET home_team_elo = ({s}) WHERE home_team_api_id IN ({s})",(home_elos,home_ids))
cur.execute("SELECT home_team_api_ids,home_team_elo FROM Match")
print(cur.fetchall())
'''
'''

cur.execute("UPDATE Team SET elo=1000")
cur.execute(
    "SELECT date,home_team_api_id,away_team_api_id,home_team_goal,away_team_goal FROM Match WHERE season != '2015/2016'")
trainMatches = cur.fetchall()
trainMatches.sort(key=lambda x: x[1])

cur.execute(
    "SELECT date,home_team_api_id,away_team_api_id,winner FROM Match WHERE season = '2015/2016'")
testMatches = cur.fetchall()

train(50,trainMatches,1,cur,1,1,1)

print(test(testMatches,50,cur))
'''
cur.execute("SELECT B365H,B365A,B365D FROM Match")
match_data = cur.fetchall()

imp_mean = SimpleImputer(missing_values=np.nan)
b365d = imp_mean.fit_transform(match_data)
print(np.shape(b365d))


