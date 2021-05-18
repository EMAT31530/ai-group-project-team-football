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
        cur.execute(f"UPDATE Match SET winner = 'home' WHERE id = {match_id}")
    elif away_goal > home_goal:
        cur.execute(f"UPDATE Match SET winner = 'away' WHERE id = {match_id}")
    elif home_goal == away_goal:
        cur.execute(f"UPDATE Match SET winner = 'draw' WHERE id = {match_id}")

con.commit()

