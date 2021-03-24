import joblib
import sqlite3
import numpy as np
import pandas as pd
from randomforest import predict




def predMatch(model_filepath, home_name, away_name,home_elo,away_elo):
    rf = joblib.load(model_filepath)

    cur.execute(f"SELECT team_api_id FROM Team WHERE team_short_name = '{home_name}'")
    home_id = str(cur.fetchall()[0][0])

    cur.execute(f"SELECT team_api_id FROM Team WHERE team_short_name = '{away_name}'")
    away_id = str(cur.fetchall()[0][0])

    match_df = pd.DataFrame()

    match_df["home_team"] = [home_id]
    match_df["away_team"] = [away_id]
    match_df["home_elo"] = [home_elo]
    match_df["away_elo"] = [away_elo]


    match_np = np.array(match_df)

    pred = predict(rf,match_np)

    return pred



con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
cur = con.cursor()

model_path = r"D:\intro2ai\ai-group-project-team-football\rf.pkl"
res_path = r"D:\intro2ai\ai-group-project-team-football\res.pkl"

pred = predMatch(model_path,"MCI","MUN",1500,700)[0]

if pred != 'draw':
    cur.execute(f"SELECT team_long_name FROM Team WHERE team_api_id = {pred}")
    print(cur.fetchall()[0][0])
else:
    print('draw')