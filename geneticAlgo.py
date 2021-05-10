import random
import sqlite3
from elo2 import train
from elo2 import test
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def getProgenitors(num):      #  [k,d,2mult,3mult,4mult]
    ranges = np.asarray([[1, 300], [0,400],[1,3],[1,4],[1,5]])
    starts = ranges[:, 0]
    widths = ranges[:, 1] - ranges[:, 0]
    progenArr = starts + widths * np.random.random(size=(num, widths.shape[0]))
    #progenArr = np.random.randint(valRange[0],valRange[1], size = (num,2))
    return progenArr


def sortedFitness(entities,trainData,testData,cur):

    entityAndAccArr = []

    i = 1

    for entity in entities:
        global genNumber
        '''
        print("Fitness %d/%d:  Gen %d" %(i,len(entities),genNumber))
        '''
        i+=1
        k = entity[0]
        d = entity[1]
        mult2 = entity[2]
        mult3 = entity[3]
        mult4 = entity[4]

        train(k,trainData,1,cur,mult2,mult3,mult4,dotqdm=False)

        '''
        cur.execute("SELECT home_team_api_id from Match WHERE league_id = '1729'")
        team_ids = cur.fetchall()
        team_ids = str(tuple(set([team_id[0] for team_id in team_ids])))

        cur.execute(f"SELECT elo FROM Team WHERE team_api_id IN {team_ids}")
        print(cur.fetchall())
        '''

        acc = test(testData,d,cur)

        entityAndAccArr.append([entity,acc])

    entityAndAccArr = sorted(entityAndAccArr,key=lambda x: x[1])
    entityAndAccArr.reverse()
    return entityAndAccArr


def breeding(parent1,parent2,mutRate,sds):

    crossPos = random.randint(0,4)
    if crossPos == 4:
        child1 = parent1
        child2 = parent2
    else:

        child1 = np.append(parent1[:crossPos], parent2[crossPos:])
        child2 = np.append(parent2[:crossPos], parent1[crossPos:])


    mutChance = random.randrange(0,100)
    if mutChance < mutRate:
        child1 = mutate(child1,sds)
    mutChance = random.randrange(0, 100)
    if mutChance < mutRate:
        child2 = mutate(child2,sds)

    return child1,child2

def mutate(entity,sds):

    mutPos = random.randint(0,4)

    mutGene = entity[mutPos] + random.uniform(-1*sds[mutPos],sds[mutPos])
    if mutGene <= 0:
        mutGene *= -1

    entity[mutPos] = mutGene
    return entity


def newGen(entities,trainData, testData,cur,mutrate,sds):
    trainTest = sortedFitness(entities,trainData,testData,cur)

    sorted = np.array(trainTest,dtype=object)

    bestAcc = np.max(sorted[:,1])
    aveAcc = np.mean(sorted[:,1])

    quarterNum = int(len(sorted)/4)

    topQuarter = sorted[0:quarterNum]

    children = getChildren(topQuarter,3*quarterNum,mutrate,sds)

    noAcc = [ent[0] for ent in topQuarter]

    newGen = noAcc+children
    newGen = np.array(newGen)

    return [newGen,bestAcc,aveAcc]



def getChildren(parents,num,mutrate,sds):

    children = []
    for i in range(int(num/2)):
        parent1 = rouletteSelection(parents)[0]
        parent2 = rouletteSelection(parents)[0]
        child1,child2 = breeding(parent1,parent2,mutrate,sds)
        children.append(child1)
        children.append(child2)
    return children


def rouletteSelection(parents):
    parentsAcc = []
    for ent in parents:
        parentsAcc.append(ent[1])
    parentsProb = []
    prevProb = 0
    for ent in parents:
        prob = (ent[1])/sum(parentsAcc)+prevProb
        prevProb = prob
        parentsProb.append(prob)
    num = random.random()
    for i in range(len(parentsProb)):
        if num < parentsProb[i]:
            selection = i
            break

    return parents[selection]


def algoIters(progens,numIters,trainData,testData,cur,mutrate):
    pop = progens

    aveAccArr = []
    bestAccArr = []

    prevAcc = 0
    repCount = 0
    pbar = tqdm(total=numIters)
    pbar.set_description(f"Ave. Acc. = n/a, Best Acc. = n/a, k = {pop[0][0]}, d = {pop[0][1]}, mult2 = {pop[0][2]}, mult3 = {pop[0][3]}, mult4 = {pop[0][4]}")
    for i in range(numIters):


        k_sd = np.std(pop[:, 0])
        d_sd = np.std(pop[:, 1])
        m2_sd = np.std(pop[:, 2])
        m3_sd = np.std(pop[:, 3])
        m4_sd = np.std(pop[:, 4])
        popSDs = [k_sd, d_sd, m2_sd, m3_sd, m4_sd]

        new = newGen(pop,trainData,testData,cur,mutrate,popSDs)


        pop = new[0]
        bestAcc = new[1]
        aveAcc = new[2]

        pbar.set_description(f"Ave. Acc = {aveAcc}, Best Acc. = {bestAcc}, k = {pop[0][0]}, d = {pop[0][1]}, mult2 = {pop[0][2]}, mult3 = {pop[0][3]}, mult4 = {pop[0][4]}")
        pbar.update(1)
        if abs(prevAcc-aveAcc) <= 0.001:
            repCount+=1
        else:
            repCount = 0
        if repCount == 10:
            print("Same ave acc for 10 iters --> halt")
            print(f"Ave. Acc = {aveAcc}, Best Acc. = {bestAcc}, k = {pop[0][0]}, d = {pop[0][1]}, mult2 = {pop[0][2]}, mult3 = {pop[0][3]}, mult4 = {pop[0][4]}")
            break
        prevAcc = aveAcc

        bestAccArr.append(bestAcc)
        aveAccArr.append(aveAcc)



    return bestAccArr,aveAccArr, [pop[0][0],pop[0][1],pop[0][2],pop[0][3],pop[0][4]]



con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
cur = con.cursor()
cur.execute("UPDATE Team SET elo=1000")
cur.execute(
        "SELECT home_team_api_id,away_team_api_id,home_team_goal,away_team_goal,winner FROM Match WHERE league_id = '1729'")
matches = cur.fetchall()

trainMatches, testMatches = train_test_split(matches, test_size=0.2)
trainMatches, valMatches = train_test_split(trainMatches, test_size=0.2)


progens = getProgenitors(800)

bestAccs,aveAccs, valArr = algoIters(progens,100,trainMatches,valMatches,cur,10)

k = valArr[0]
d = valArr[1]
mult2 = valArr[2]
mult3 = valArr[3]
mult4 = valArr[4]

train(k,testMatches,1,cur,mult2,mult3,mult4)
testAcc = test(testMatches,d,cur)

print(f"Test. Acc for best params. = {testAcc}")

plt.plot(bestAccs,label = 'Best')
plt.plot(aveAccs,label = 'Ave')
plt.legend()
plt.show()






