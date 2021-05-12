import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.optimize import dual_annealing


def eloUpdate(r1, r2, s1, s2, k,goal1,goal2,mult2,mult3,mult4):  # r1 = team 1 elo, r2 = team 2 elo, s1/2 = 1,0,0.5 depending win/lose/draw
    if goal1 > goal2:
        if goal1 - goal2 == 2:
            k = mult2*k
        elif goal1 - goal2 == 3:
            k = mult3*k
        elif goal1 - goal2 >= 4:
            k = k * mult4
    elif goal1 < goal2:
        if goal2 - goal1 == 2:
            k = mult2*k
        elif goal2 - goal1 == 3:
            k = mult3*k
        elif goal2 - goal1 >= 4:
            k = k * mult4

    R1 = 10 ** (r1 / 400)
    R2 = 10 ** (r2 / 400)

    e1 = R1 / (R1 + R2)
    e2 = R2 / (R1 + R2)

    r1_update = r1 + k * (s1 - e1)
    r2_update = r2 + k * (s2 - e2)

    return r1_update, r2_update


def percTrainData(i, traindata,k1,k2,k3):
    if i / len(traindata) > 0.3:
        return k2
    if i / len(traindata) > 0.6:
        return k3
    return k1


def train(k, traindata, z,cur,mult2,mult3,mult4,dotqdm = False): #data = [match_1,...,match_n], match = [home_id,away_id,home_goal,away_goal,winner]
    if z == 1:
        cur.execute("UPDATE Team SET elo=1000")
    i = 0
    if dotqdm:
        for match in tqdm(traindata,desc = "Training ELOs", total = len(traindata)):
            if match[4] == 'draw':
                s1 = s2 = 0.5
            elif match[4] == match[0]:
                s1 = 1
                s2 = 0
            elif match[4] == match[1]:
                s1 = 0
                s2 = 1
            cur.execute("SELECT elo FROM Team where team_api_id = ?", (match[0],))
            homeTeamElo = cur.fetchall()[0][0]
            cur.execute("SELECT elo FROM Team where team_api_id = ?", (match[1],))
            awayTeamElo = cur.fetchall()[0][0]

            newElos = eloUpdate(homeTeamElo, awayTeamElo, s1, s2, k, match[2], match[3], mult2, mult3,mult4)

            cur.execute("UPDATE Team SET elo = ? where team_api_id = ?", (newElos[0], match[0],))
            cur.execute("UPDATE Team SET elo = ? where team_api_id = ?", (newElos[1], match[1],))
            i += 1
    else:
        for match in traindata:
            if match[4] == 'draw':
                s1 = s2 = 0.5
            elif match[4] == match[0]:
                s1 = 1
                s2 = 0
            elif match[4] == match[1]:
                s1 = 0
                s2 = 1
            cur.execute("SELECT elo FROM Team where team_api_id = ?", (match[0],))
            homeTeamElo = cur.fetchall()[0][0]
            cur.execute("SELECT elo FROM Team where team_api_id = ?", (match[1],))
            awayTeamElo = cur.fetchall()[0][0]

            newElos = eloUpdate(homeTeamElo, awayTeamElo, s1, s2, k,match[2],match[3],mult2,mult3,mult4)
            cur.execute("UPDATE Team SET elo = ? where team_api_id = ?", (newElos[0], match[0],))
            cur.execute("UPDATE Team SET elo = ? where team_api_id = ?", (newElos[1], match[1],))
            i += 1


def perDiff(val1, val2):
    x = abs(val1 - val2)
    y = (val1 + val2) / 2
    perDiff = x / y
    return perDiff


def test(testdata, drawThreshold,cur):  # testdata = [match_1,...,match_n], match = [home_id,away_id,home_goal,away_goal,winner]

    true = np.empty(shape = len(testdata),dtype=object)
    predictions = np.empty(shape = len(testdata),dtype=object)
    i = 0
    for match in testdata:
        home_id = str(match[0])
        away_id = str(match[1])
        cur.execute("SELECT elo FROM Team WHERE team_api_id = ?", (match[0],))
        homeElo = cur.fetchall()[0][0]

        cur.execute("SELECT elo FROM Team WHERE team_api_id = ?", (match[1],))
        awayElo = cur.fetchall()[0][0]


        if abs(homeElo - awayElo) < drawThreshold:
            predictions[i] = home_id
        elif homeElo > awayElo:
            predictions[i] = home_id
        elif homeElo < awayElo:
            predictions[i] = away_id

        if match[4] == match[0]:  # home win
            true[i] = home_id
        elif match[4] == match[1]:  # away win
            true[i] = away_id
        elif match[4] == 'draw':  # draw
            true[i] = 'draw'
        i+=1
    accuracy = metrics.accuracy_score(true, predictions)
    return accuracy


def randomPred():
    return random.randrange(-1,1)

def main():
    con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
    cur = con.cursor()

    cur.execute("UPDATE Team SET elo=1000")




    cur.execute(
        "SELECT home_team_api_id,away_team_api_id,home_team_goal,away_team_goal,winner FROM Match WHERE league_id = '1729'")
    matches = cur.fetchall()
    trainMatches, testMatches = train_test_split(matches, test_size=0.4)
    trainMatches, trainValMatches = train_test_split(trainMatches, test_size = 0.333)
    testMatches, testValMatches = train_test_split(testMatches,test_size=0.5)

    args = (trainMatches, trainValMatches, cur)

    accArr = []
    def callback(x, f, context):
        # print(f"Acc: {1-f}")
        pbar.set_description(f"Optimising ELO Parameters, Acc.= {1 - f}")
        accArr.append((1-f) * 100)
        pbar.update(1)


    def func(x, *args):
        train(x[0], args[0], 1, args[2], x[2], x[3], x[4])
        acc = test(args[1], x[1], args[2])
        return 1 - acc

    maxiter = 1000
    with tqdm(desc=f"Optimising ELO Parameters, Acc.= NaN") as pbar:
        res = dual_annealing(func, bounds=[(1, 300), (0, 300), (1, 3), (1, 4), (1, 5)], callback=callback,
                             args=args,
                             maxiter=maxiter)
    res = res['x']
    k = res[0]
    d = res[1]
    mult2 = res[2]
    mult3 = res[3]
    mult4 = res[4]

    train(k,testMatches,1,cur,mult2,mult3,mult4)
    testAcc = test(testValMatches,d,cur)
    print(testAcc)

    iters = [i for i in range(1,len(accArr)+1)]
    plt.plot(iters,accArr)
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Percentage Accuracy')
    plt.show()

    '''
    ks = np.linspace(1,500,500)
    ds = np.linspace(0,500,500)
    bestAcc = 0

    for k in tqdm(ks):

        train(k,trainMatches,1,cur,1,1,1,dotqdm=False)
        for d in ds:
            karr.append(k)
            darr.append(d)
            acc = test(trainValMatches,d,cur)
            if acc > bestAcc:
                bestAcc = acc
                bestK = k
                bestD = d
            accarr.append(acc)

    print(bestAcc)
    print(bestK)
    print(bestD)

    train(bestK,testMatches,1,cur,1,1,1)
    testAcc = test(testValMatches,bestD,cur)
    print(testAcc)

    x = np.array(karr)
    y = np.array(darr)
    z = np.array(accarr)

    cols = np.unique(x).shape[0]
    X = x.reshape(-1,cols)
    Y = y.reshape(-1,cols)
    Z = z.reshape(-1,cols)

    plt.contourf(X,Y,Z)
    clb = plt.colorbar(plt.contourf(X,Y,Z))
    clb.set_label('Percentage Accuracy')
    plt.xlabel('K')
    plt.ylabel('D')
    plt.show()
    '''

if __name__ == "__main__":
    main()