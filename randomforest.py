import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm
from elo2 import train, test
from scipy.optimize import dual_annealing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
from multiprocessing import cpu_count
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer



def predict(model, test_features, id=True):
    classes = model.classes_
    probs = model.predict_proba(test_features)
    probPreds = np.empty(shape=np.shape(test_features)[0], dtype=object)
    i = 0
    for feat in test_features:
        home = feat[0]
        away = feat[1]
        home_i = np.where(classes == home)
        away_i = np.where(classes == away)
        draw_i = np.where(classes == 'draw')
        probArr = probs[i]
        home_prob = probArr[home_i]
        away_prob = probArr[away_i]
        draw_prob = probArr[draw_i]
        predChoice = np.array([home, away, 'draw'])
        predArr = np.array([home_prob, away_prob, draw_prob])
        pred = predChoice[np.where(predArr == np.max(predArr))[0][0]]
        probPreds[i] = pred

        i += 1
    return probPreds


def optimiseElo(cur, trainMatches, testMatches, maxiter=1000,save = False,savePath = "path", useSavedRes = False, useSavedResPath = "path"):
    cur.execute("UPDATE Team SET elo=1000")

    args = (trainMatches, testMatches, cur)

    def callback(x, f, context):
        #print(f"Acc: {1-f}")
        pbar.set_description(f"Optimising ELO Parameters, Acc.= {1-f}")
        pbar.update(1)

    def func(x, *args):
        train(x[0], args[0], 1, args[2], x[2], x[3], x[4])
        acc = test(args[1], x[1], args[2])
        return 1 - acc
    if not useSavedRes:
        with tqdm(desc=f"Optimising ELO Parameters, Acc.= NaN") as pbar:
            res = dual_annealing(func, bounds=[(1, 300), (0, 300), (1, 3), (1, 4), (1, 5)], callback=callback, args=args,
                               maxiter=maxiter)
    else:
        res = joblib.load(useSavedResPath)
    if save:
        joblib.dump(res,savePath)
    res = res["x"]
    k = res[0]
    mult2 = res[2]
    mult3 = res[3]
    mult4 = res[4]
    cur.execute(
        "SELECT home_team_api_id,away_team_api_id,home_team_goal,away_team_goal,winner FROM Match")
    allMatches = cur.fetchall()
    allMatches.sort(key=lambda x: x[1])
    train(k, allMatches, 1, cur, mult2, mult3, mult4, dotqdm=True)


def main():
    con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
    cur = con.cursor()

    cur.execute(
        "SELECT home_team_api_id,away_team_api_id,home_team_goal,away_team_goal,winner FROM Match")
    matches = cur.fetchall()
    trainMatches, testMatches = train_test_split(matches, test_size=0.25)

    saveResPath = r"D:\intro2ai\ai-group-project-team-football\res.pkl"
    optimiseElo(cur, trainMatches, testMatches,useSavedRes=True,useSavedResPath=saveResPath)

    cur.execute("SELECT elo,team_api_id FROM TEAM")
    team_elo_and_id = cur.fetchall()

    cur.executemany(f"UPDATE Match SET home_team_elo = ? WHERE home_team_api_id = ?", team_elo_and_id)
    cur.executemany(f"UPDATE Match SET away_team_elo = ? WHERE away_team_api_id = ?", team_elo_and_id)



    cur.execute("SELECT winner,home_team_api_id,away_team_api_id,home_team_elo,away_team_elo FROM Match")
    match_data = cur.fetchall()
    matches_df = pd.DataFrame(match_data, columns=["winner","home_team","away_team", "home_elo", "away_elo"])

    cur.execute("SELECT WHH,WHA,WHD FROM Match")
    gambling_odds = cur.fetchall()
    imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
    gambling_odds = imp_mean.fit_transform(gambling_odds)
    b365_cols = ["whh","wha","whd"]
    matches_df[b365_cols] = gambling_odds

    scaler = MinMaxScaler()
    columns_to_norm = ['home_elo','away_elo','whh','wha','whd']
    x = matches_df[columns_to_norm].values
    x_scaled = scaler.fit_transform(x)
    df_temp = pd.DataFrame(x_scaled,columns = columns_to_norm,index = matches_df.index)
    matches_df[columns_to_norm] = df_temp

    matches_df["winner"] = matches_df["winner"].astype("str")
    matches_df["home_team"] = matches_df["home_team"].astype("str")
    matches_df["away_team"] = matches_df["away_team"].astype("str")

    # matches_df = pd.get_dummies(matches_df)

    labels = matches_df.filter(items=["winner"])

    label_list = labels.columns

    features = matches_df.drop(columns= label_list)
    feature_list = features.columns

    labels = np.array(labels)
    features = np.array(features)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)

    threads = cpu_count() - 1
    rf = RandomForestClassifier(n_estimators=350,verbose=1, n_jobs=threads)


    rf.fit(train_features, train_labels.ravel())


    #joblib.dump(ml, r"D:\intro2ai\ai-group-project-team-football\ml.pkl")


    rf_preds = predict(rf, test_features)



    print("RF Accuracy: ", metrics.accuracy_score(test_labels, rf_preds))



    rf_feature_importances = pd.DataFrame(rf.feature_importances_,
                                       index=feature_list,
                                       columns=['importance']).sort_values('importance', ascending=False)

    print("RF", rf_feature_importances)




if __name__ == '__main__':
    main()
