import sqlite3
import pandas as pd

con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
cur = con.cursor()


cur.execute(("SELECT season,home_team_api_id,away_team_api_id,home_team_goal,away_team_goal, winner FROM Match"))
match_data = cur.fetchall()
print(match_data)
