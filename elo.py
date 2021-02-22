import sqlite3

con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
cur = con.cursor()

cur.execute("UPDATE Team SET elo=1000")
cur.execute(
    "SELECT date,home_team_api_id,away_team_api_id,home_team_goal,away_team_goal FROM Match WHERE season != '2015/2016'")
trainMatches = cur.fetchall()
trainMatches.sort(key=lambda x: x[0])


def eloUpdate(r1, r2, s1, s2, K):  # r1 = team 1 elo, r2 = team 2 elo, s1/2 = 1,0,0.5 depending win/lose/draw
    R1 = 10 ** (r1 / 400)
    R2 = 10 ** (r2 / 400)

    e1 = R1 / (R1 + R2)
    e2 = R2 / (R1 + R2)

    r1_update = r1 + K * (s1 - e1)
    r2_update = r2 + K * (s2 - e2)

    return r1_update, r2_update


for match in trainMatches:
    if match[3] == match[4]:
        s1 = s2 = 0.5
    elif match[3] > match[4]:
        s1 = 1
        s2 = 0
    else:
        s1 = 0
        s2 = 1
    cur.execute("SELECT elo FROM Team where team_api_id = ?", (match[1],))
    homeTeamElo = cur.fetchall()
    cur.execute("SELECT elo FROM Team where team_api_id = ?", (match[2],))
    awayTeamElo = cur.fetchall()

    newElos = eloUpdate(homeTeamElo[0][0], awayTeamElo[0][0], s1, s2, 32)
    cur.execute("UPDATE Team SET elo = ? where team_api_id = ?", (newElos[0], match[1],))
    cur.execute("UPDATE Team SET elo = ? where team_api_id = ?", (newElos[1], match[2],))

cur.execute(
    "SELECT home_team_api_id,away_team_api_id,home_team_goal,away_team_goal FROM Match WHERE season = '2015/2016'")
testMatches = cur.fetchall()

successCount = 0

for match in testMatches:
    cur.execute("SELECT elo FROM Team WHERE team_api_id = ?", (match[0],))
    homeElo = cur.fetchall()
    cur.execute("SELECT elo FROM Team WHERE team_api_id = ?", (match[1],))
    awayElo = cur.fetchall()
    if match[2] > match[3]: # home win
        if homeElo > awayElo:
            successCount+=1

    elif match[2] < match[3]: # away win
        if homeElo<awayElo:
            successCount+=1
    elif match[2] == match[3]: # draw
        if homeElo == awayElo:
            successCount+=1

accuracy = (successCount * 100) / (len(testMatches))
print(accuracy)
