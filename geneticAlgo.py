import random
import sqlite3
from elo2 import train
from elo2 import test
import matplotlib.pyplot as plt
import numpy as np


def getProgenitors(num):      #  [k,d,2mult,3mult,4n1,4n2,4n3]
    ranges = np.asarray([[1, 300], [0,200],[1,3],[1,3],[0,1],[0,5],[0,10]])
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
        print("Fitness %d/%d:  Gen %d" %(i,len(entities),genNumber))
        i+=1
        k = entity[0]
        d = entity[1]
        mult2 = entity[2]
        mult3 = entity[3]
        mult4_1 = entity[4]
        mult4_2 = entity[5]
        mult4_3 = entity[6]
        train(k,trainData,1,cur,mult2,mult3,mult4_1,mult4_2,mult4_3)
        acc = test(testData,d,cur)

        entityAndAccArr.append([entity,acc])
    sorted(entityAndAccArr, key=lambda x: x[1])
    entityAndAccArr.reverse()
    return entityAndAccArr


def breeding(parent1,parent2,mutRate):

    k1 = parent1[0]
    d1 = parent1[1]
    k2 = parent2[0]
    d2 = parent2[1]

    mult2_1 = parent1[2]
    mult3_1 = parent1[3]
    mult4_1_1 = parent1[4]
    mult4_2_1 = parent1[5]
    mult4_3_1 = parent1[6]

    mult2_2 = parent2[2]
    mult3_2 = parent2[3]
    mult4_1_2 = parent2[4]
    mult4_2_2 = parent2[5]
    mult4_3_2 = parent2[6]

    k_bred = (k1+k2)/2
    d_bred = (d1+d2)/2

    mult2_bred = (mult2_1+mult2_2)/2
    mult3_bred = (mult3_1+mult3_2)/2
    mult4_1_bred = (mult4_1_1+mult4_1_2)/2
    mult4_2_bred =  (mult4_2_1+mult4_2_2)/2
    mult4_3_bred =  (mult4_3_1+mult4_3_2)/2


    child = [k_bred,d_bred,mult2_bred,mult3_bred,mult4_1_bred,mult4_2_bred,mult4_3_bred]

    mutChance = random.randrange(0,100)
    if mutChance < mutRate:
        child = mutate(child)
    return child

def mutate(entity):
    mutAmount = random.uniform(-25,25)
    multMutAmount = random.uniform(-0.5,0.5)
    k_mut = entity[0]+mutAmount
    d_mut = entity[1]+mutAmount

    mult2_mut = entity[2] + multMutAmount
    mult3_mut = entity[3] + multMutAmount
    mult4_1_mut = entity[4] + multMutAmount
    mult4_2_mut = entity[5] + multMutAmount
    mult4_3_mut = entity[6] + multMutAmount

    mutant = [k_mut,d_mut,mult2_mut,mult3_mut,mult4_1_mut,mult4_2_mut,mult4_3_mut]
    return mutant


def newGen(entitys,trainData, testData,cur,mutrate):
    trainTest = sortedFitness(entitys,trainData,testData,cur)

    sorted = trainTest


    bestAcc = sorted[0][1]
    quarterNum = int(len(sorted)/4)



    topQuarter = sorted[0:quarterNum]

    children = getChildren(topQuarter,3*quarterNum,mutrate)

    noAcc = [ent[0] for ent in topQuarter]


    newGen = noAcc+children



    return [newGen,bestAcc]



def getChildren(parents,num,mutrate):
    children = []
    for i in range(num):
        parent1 = rouletteSelection(parents)[0]
        parent2 = rouletteSelection(parents)[0]
        child = breeding(parent1,parent2,mutrate)
        children.append(child)
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

    accArr = []

    for i in range(numIters):
        global genNumber
        genNumber = i + 1
        new = newGen(pop,trainData,testData,cur,mutrate)
        pop = new[0]
        bestAcc = new[1]


        accArr.append(bestAcc)

        print("Generation %d/%d: Best Acc. = %.3f, k = %.3f, d = %.3f, mult2 = %.3f, mult3 = %.3f, mult4_1 = %.3f, mult4_2 = %.3f, mult4_3 = %.3f" %(i+1,numIters,bestAcc,pop[0][0],pop[0][1],pop[0][2],pop[0][3],pop[0][4],pop[0][5],pop[0][6]))

    return accArr



con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
cur = con.cursor()
cur.execute("UPDATE Team SET elo=1000")
cur.execute(
    "SELECT date,home_team_api_id,away_team_api_id,home_team_goal,away_team_goal FROM Match WHERE season != '2015/2016'")
trainMatches = cur.fetchall()
trainMatches.sort(key=lambda x: x[1])
cur.execute(
       "SELECT date,home_team_api_id,away_team_api_id,home_team_goal,away_team_goal FROM Match WHERE id > 25000")
testMatches = cur.fetchall()



progens = getProgenitors(200)


arrs = algoIters(progens,50,trainMatches,testMatches,cur,20)

plt.plot(arrs)

plt.show()






