import random
import sqlite3
from elo2 import train
from elo2 import test
import matplotlib.pyplot as plt
import numpy as np


def getProgenitors(num,valRange=[0,1000]):
    progenArr = np.random.randint(valRange[0],valRange[1], size = (num,2))
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
        train(k,trainData,1,cur)
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

    k_bred = (k1+k2)/2
    d_bred = (d1+d2)/2
    child = [k_bred,d_bred]

    mutChance = random.randrange(0,100)
    if mutChance < mutRate:
        child = mutate(child)
    return child

def mutate(entity):
    mutAmount = random.randrange(-25,25)
    k_mut = entity[0]+mutAmount
    d_mut = entity[1]+mutAmount
    mutant = [k_mut,d_mut]
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

        print("Generation %d/%d: Best Acc. = %d, k = %d, d = %d" %(i+1,numIters,bestAcc,pop[0][0],pop[0][1]))

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



progens = getProgenitors(100, valRange=[1,200])

arrs = algoIters(progens,20,trainMatches,testMatches,cur,20)

plt.plot(arrs)

plt.show()






