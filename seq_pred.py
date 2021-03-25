import numpy as np
import pandas as pd
import joblib
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def constructInput(cur,scaler,ohe_home,ohe_away,home_name,away_name,home_elo,away_elo,whh,wha,whd):
    cur.execute(f"SELECT team_api_id FROM Team WHERE team_short_name = '{home_name}'")
    home_id = cur.fetchall()[0][0]
    cur.execute(f"SELECT team_api_id FROM Team WHERE team_short_name = '{away_name}'")
    away_id = cur.fetchall()[0][0]
    columns = ["home_team","away_team","home_elo","away_elo","whh","wha","whd"]
    data = np.array([home_id,away_id,home_elo,away_elo,whh,wha,whd])
    match_df = pd.DataFrame([data],columns = columns)

    columns_to_norm = ['home_elo', 'away_elo', 'whh', 'wha', 'whd']
    match_df[columns_to_norm] = scaler.transform(match_df[columns_to_norm])

    match_df["home_team"] = match_df["home_team"].astype("str")
    match_df["away_team"] = match_df["away_team"].astype("str")

    ohe_home_trans = ohe_home.transform(match_df["home_team"])
    ohe_away_trans = ohe_away.transform(match_df["away_team"])
    match_df = match_df.drop(columns=["home_team","away_team"])

    ohe_home_df = pd.DataFrame(ohe_home_trans, columns=ohe_home.classes_)
    ohe_away_df = pd.DataFrame(ohe_away_trans, columns=ohe_away.classes_)

    match_df = match_df.join(ohe_home_df, how="right")
    match_df = match_df.join(ohe_away_df, how="right")

    match_np = np.array(match_df)

    return match_np


con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
cur = con.cursor()

lb = joblib.load(r"D:\intro2ai\ai-group-project-team-football\lb.pkl")

scaler = joblib.load(r"D:\intro2ai\ai-group-project-team-football\scaler.pkl")
ohe_home = joblib.load(r"D:\intro2ai\ai-group-project-team-football\ohe_home.pkl")
ohe_away = joblib.load(r"D:\intro2ai\ai-group-project-team-football\ohe_away.pkl")
label_list = joblib.load(r"D:\intro2ai\ai-group-project-team-football\label_list.pkl")


inp = constructInput(cur,scaler,ohe_home,ohe_away,"MUN","HUL",1500,1000,3.9999,1.7,3.2)

model = load_model(r"D:\intro2ai\ai-group-project-team-football\model")

pred = np.array(model(inp))
print(lb.inverse_transform(pred))
#test_features = joblib.load(r"D:\intro2ai\ai-group-project-team-football\test_features.pkl")

#preds = np.array(model.predict(test_features))

#print(scaler.inverse_transform(np.array([0.3439247 , 0.56786907, 0.11929543 ,0.01742788 ,0.14893617]).reshape(1,-1)))
'''
for pred in preds:
    print(lb.inverse_transform(np.array([pred])))
'''