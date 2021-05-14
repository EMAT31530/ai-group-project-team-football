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
from sklearn.preprocessing import MinMaxScaler,LabelBinarizer
from sklearn.impute import SimpleImputer
from sqlAttackDefense import setTeamAttackDefence


def setAttackDefence(cur,set=False):
    if set:
        setTeamAttackDefence(cur)
    else:
        pass

def baseline_accuracy(test_labels,test_features,feature_list,ohe_home):
    test_features_df = pd.DataFrame(test_features, columns=feature_list)
    home = test_features_df.filter(regex = "(.*)_home")

    true = test_labels
    base_preds = ohe_home.inverse_transform(home.values)
    base_preds = [pred.replace("_home","") for pred in base_preds]

    baseline_accuracy = metrics.accuracy_score(true,base_preds)*100
    return baseline_accuracy


def predict(model, test_features,lb):
    probs = np.array(model.predict_proba(test_features))[:,:,1]
    probs = np.transpose(probs)
    preds = lb.inverse_transform(probs)
    return preds


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

    return res



def main():
    con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
    cur = con.cursor()

    cur.execute(
        "SELECT home_team_api_id,away_team_api_id,home_team_goal,away_team_goal,winner FROM Match WHERE league_id = '1729'")
    matches = cur.fetchall()

    k = 4.849879326548514
    d = 125.81976547554737
    mult2 = 2.1342190777850742
    mult3 = 3.4769118949428233
    mult4 = 1.143940853285419

    train(k, matches, 1, cur, mult2, mult3, mult4, dotqdm=True)



    cur.execute("SELECT home_team_api_id from Match WHERE league_id = '1729'")
    team_ids = cur.fetchall()
    team_ids = str(tuple(set([team_id[0] for team_id in team_ids])))

    with tqdm(desc = "Data Preprocessing Begin",total=6) as pbar:

        pbar.set_description("Setting Home and Away ELOs")

        cur.execute(f"SELECT elo,team_api_id FROM Team WHERE team_api_id IN {team_ids}")
        team_elo_and_id = cur.fetchall()

        cur.executemany(f"UPDATE Match SET home_team_elo = ? WHERE home_team_api_id = ?", team_elo_and_id)
        cur.executemany(f"UPDATE Match SET away_team_elo = ? WHERE away_team_api_id = ?", team_elo_and_id)

        pbar.update(1)
        pbar.set_description("Setting Home and Away Atk. and Def. Ratings")

        setAttackDefence(cur, set=False)

        cur.execute(f"SELECT attack,defence,team_api_id FROM Team WHERE team_api_id IN {team_ids}")
        team_attack_defense_id = cur.fetchall()

        cur.executemany(f"UPDATE MATCH SET home_attack = ?,home_defence = ? WHERE home_team_api_id = ?",
                        team_attack_defense_id)
        cur.executemany(f"UPDATE MATCH SET away_attack = ?,away_defence = ? WHERE away_team_api_id = ?",
                        team_attack_defense_id)

        pbar.update(1)
        pbar.set_description("Generating DataFrame")

        cur.execute(
            "SELECT winner,home_team_api_id,away_team_api_id,home_team_elo,away_team_elo,home_attack,home_defence,"
            "away_attack,away_defence FROM Match WHERE league_id = '1729'")
        match_data = cur.fetchall()
        matches_df = pd.DataFrame(match_data, columns=["winner", "home_team", "away_team", "home_elo", "away_elo",
                                                       "home_attack", "home_defence", "away_attack", "away_defence"])

        '''
        cur.execute("SELECT WHH,WHA,WHD,B365H,B365A,B365D,LBH,LBA,LBD FROM Match WHERE league_id = '1729'")
        gambling_odds = cur.fetchall()
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        gambling_odds = imp_mean.fit_transform(gambling_odds)
        odds_cols = ["whh", "wha", "whd", "b365h", "b365a", "b365d", "lbh", "lba", "lbd"]
        matches_df[odds_cols] = gambling_odds
        '''

        pbar.update(1)
        pbar.set_description("Normalising Numerical Data")

        scaler = MinMaxScaler()
        columns_to_norm = ['home_elo', 'away_elo','home_attack','home_defence','away_attack','away_defence'] #+ odds_cols
        x = matches_df[columns_to_norm].values
        scaler.fit(x)
        x_scaled = scaler.transform(x)
        df_temp = pd.DataFrame(x_scaled, columns=columns_to_norm, index=matches_df.index)
        matches_df[columns_to_norm] = df_temp

        pbar.update(1)
        pbar.set_description("OneHotEncoding Categorical Data")

        matches_df["winner"] = matches_df["winner"].astype("str")
        matches_df["home_team"] = matches_df["home_team"].astype("str") + "_home"
        matches_df["away_team"] = matches_df["away_team"].astype("str") + "_away"

        labels = matches_df.filter(items=["winner"])
        lb = LabelBinarizer()
        lb.fit(labels["winner"])
        lb_trans = lb.transform(labels["winner"])
        labels = pd.DataFrame(lb_trans, columns=lb.classes_)
        label_list = labels.columns

        ohe_home = LabelBinarizer()
        ohe_away = LabelBinarizer()

        features = matches_df

        ohe_home.fit(features["home_team"])
        ohe_away.fit(features["away_team"])

        ohe_home_trans = ohe_home.transform(features["home_team"])
        ohe_away_trans = ohe_away.transform(features["away_team"])

        features = matches_df.drop(columns=["winner", "home_team", "away_team"])

        ohe_home_df = pd.DataFrame(ohe_home_trans, columns=ohe_home.classes_)
        ohe_away_df = pd.DataFrame(ohe_away_trans, columns=ohe_away.classes_)

        features = features.join(ohe_home_df, how="right")
        features = features.join(ohe_away_df, how="right")
        feature_list = features.columns

        labels = np.array(labels).astype('float32')
        features = np.array(features).astype('float32')

        pbar.update(1)
        pbar.set_description("Splitting Data into Train and Test")

        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,random_state=42)

        pbar.set_description("Data Preprocessing Complete")
        pbar.update(1)


    rf = RandomForestClassifier(n_estimators=1000,verbose=1, n_jobs=-1)

    rf.fit(train_features, train_labels)

    #joblib.dump(ml, r"D:\intro2ai\ai-group-project-team-football\ml.pkl")

    rf_preds = predict(rf, test_features,lb)

    test_labels = lb.inverse_transform(test_labels)

    base_acc = baseline_accuracy(test_labels, test_features, feature_list, ohe_home)

    acc = metrics.accuracy_score(test_labels, rf_preds)*100

    print('Baseline Accuracy: %.3f' % base_acc)

    print('RF Accuracy: %.3f' % acc)
    diff = (acc - base_acc)
    print('Model better than baseline by: %.3f pp' % diff)

    rf_feature_importances = pd.DataFrame(rf.feature_importances_,
                                       index=feature_list,
                                       columns=['importance']).sort_values('importance', ascending=False)


    with pd.option_context('display.max_rows', None):
        print("rf",rf_feature_importances)




if __name__ == '__main__':
    main()
