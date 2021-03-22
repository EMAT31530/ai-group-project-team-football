import sqlite3


con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
cur = con.cursor()

cur.execute("SELECT id,home_team_goal,away_team_goal,home_team_api_id,away_team_api_id FROM Match")
matches = cur.fetchall()
for match in matches:
    id = match[0]
    home_goal = match[1]
    away_goal = match[2]
    home_id = match[3]
    away_id = match[4]

    if home_goal > away_goal:
        cur.execute("UPDATE Match SET winner")
    elif away_goal > home_goal:

    elif home_goal == away_goal: