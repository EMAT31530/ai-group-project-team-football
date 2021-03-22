import sqlite3


con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
cur = con.cursor()

cur.execute("SELECT id,home_team_goal,away_team_goal,home_team_api_id,away_team_api_id FROM Match")
matches = cur.fetchall()
for match in matches:
    match_id = match[0]
    home_goal = match[1]
    away_goal = match[2]
    home_id = match[3]
    away_id = match[4]

    if home_goal > away_goal:
        cur.execute(f"UPDATE Match SET winner = {home_id} WHERE id = {match_id}")
    elif away_goal > home_goal:
        cur.execute(f"UPDATE Match SET winner = {away_id} WHERE id = {match_id}")
    elif home_goal == away_goal:
        cur.execute(f"UPDATE Match SET winner = 0 WHERE id = {match_id}")

cur.execute("SELECT home_team_api_id,away_team_api_id,home_team_goal,away_team_goal,winner FROM Match")
print(cur.fetchall())