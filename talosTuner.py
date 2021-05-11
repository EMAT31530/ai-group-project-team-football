import numpy as np
import pandas as pd
import joblib
import sqlite3
from sklearn.model_selection import train_test_split
from randomforest import optimiseElo,setAttackDefence
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.impute import SimpleImputer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, AlphaDropout, LayerNormalization
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
from multiprocessing import cpu_count
from tensorflow.keras.optimizers import Adam
from elo2 import train,test
from tqdm import tqdm
import matplotlib.pyplot as plt
import talos as ta
from talos.utils import hidden_layers
from talos.utils.best_model import best_model
from talos import Predict, Analyze








def baseline_accuracy(test_labels, test_features, feature_list, ohe_home, lb):
    test_features_df = pd.DataFrame(test_features, columns=feature_list)
    home = test_features_df.filter(regex="(.*)_home")

    true = lb.inverse_transform(test_labels)
    base_preds = ohe_home.inverse_transform(home.values)
    base_preds = [pred.replace("_home", "") for pred in base_preds]

    baseline_accuracy = metrics.accuracy_score(true, base_preds) * 100
    return baseline_accuracy


def main():
    con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
    cur = con.cursor()

    cur.execute(
        "SELECT home_team_api_id,away_team_api_id,home_team_goal,away_team_goal,winner FROM Match WHERE league_id = '1729'")
    matches = cur.fetchall()
    trainMatches, testMatches = train_test_split(matches, test_size=0.25)

    #resPath = r"D:\intro2ai\ai-group-project-team-football\res.pkl"

    #optimiseElo(cur, trainMatches, testMatches, useSavedRes=True, useSavedResPath=resPath)

    k = 4.849879326548514
    d = 125.81976547554737
    mult2 = 2.1342190777850742
    mult3 = 3.4769118949428233
    mult4 = 1.143940853285419

    train(k, matches, 1, cur, mult2, mult3, mult4, dotqdm=True)

    cur.execute("SELECT home_team_api_id from Match WHERE league_id = '1729'")
    team_ids = cur.fetchall()
    team_ids = str(tuple(set([team_id[0] for team_id in team_ids])))

    cur.execute(f"SELECT elo,team_api_id FROM Team WHERE team_api_id IN {team_ids}")
    team_elo_and_id = cur.fetchall()

    cur.executemany(f"UPDATE Match SET home_team_elo = ? WHERE home_team_api_id = ?", team_elo_and_id)
    cur.executemany(f"UPDATE Match SET away_team_elo = ? WHERE away_team_api_id = ?", team_elo_and_id)

    setAttackDefence(cur,set=False)

    cur.execute(f"SELECT attack,defence,team_api_id FROM Team WHERE team_api_id IN {team_ids}")
    team_attack_defense_id = cur.fetchall()

    cur.executemany(f"UPDATE MATCH SET home_attack = ?,home_defence = ? WHERE home_team_api_id = ?",team_attack_defense_id)
    cur.executemany(f"UPDATE MATCH SET away_attack = ?,away_defence = ? WHERE away_team_api_id = ?",team_attack_defense_id)


    cur.execute("SELECT winner,home_team_api_id,away_team_api_id,home_team_elo,away_team_elo,home_attack,home_defence,"
                "away_attack,away_defence FROM Match WHERE league_id = '1729'")
    match_data = cur.fetchall()
    matches_df = pd.DataFrame(match_data, columns=["winner", "home_team", "away_team", "home_elo", "away_elo",
                                                   "home_attack","home_defence","away_attack","away_defence"])

    cur.execute(
        "SELECT home_team_api_id,away_team_api_id,home_team_goal,away_team_goal,winner FROM Match WHERE league_id = '1729'")
    matches = cur.fetchall()
    trainMatches, testMatches = train_test_split(matches, test_size=0.2)

    k = 4.849879326548514
    d = 125.81976547554737
    mult2 = 2.1342190777850742
    mult3 = 3.4769118949428233
    mult4 = 1.143940853285419

    train(k, trainMatches, 1, cur, mult2, mult3, mult4, dotqdm=False)
    eloAcc = test(testMatches,d,cur)

    '''
    cur.execute("SELECT WHH,WHA,WHD,B365H,B365A,B365D,LBH,LBA,LBD FROM Match WHERE league_id = '1729'")
    gambling_odds = cur.fetchall()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    gambling_odds = imp_mean.fit_transform(gambling_odds)
    odds_cols = ["whh", "wha", "whd", "b365h", "b365a", "b365d", "lbh", "lba", "lbd"]
    matches_df[odds_cols] = gambling_odds
    '''

    scaler = MinMaxScaler()
    columns_to_norm = ['home_elo', 'away_elo','home_attack','home_defence','away_attack','away_defence'] #+ odds_cols


    x = matches_df[columns_to_norm].values
    scaler.fit(x)
    joblib.dump(scaler, r"D:\intro2ai\ai-group-project-team-football\scaler.pkl")
    x_scaled = scaler.transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=columns_to_norm, index=matches_df.index)
    matches_df[columns_to_norm] = df_temp

    matches_df["winner"] = matches_df["winner"].astype("str")
    matches_df["home_team"] = matches_df["home_team"].astype("str") + "_home"
    matches_df["away_team"] = matches_df["away_team"].astype("str") + "_away"

    labels = matches_df.filter(items=["winner"])
    lb = LabelBinarizer()
    lb.fit(labels["winner"])
    lb_trans = lb.transform(labels["winner"])
    labels = pd.DataFrame(lb_trans, columns=lb.classes_)
    label_list = labels.columns
    numLabels = len(label_list)

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

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

    train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels,test_size=0.2)

    n_features = train_features.shape[1]

    p = {'shapes': ['brick', 'triangle', 'funnel'],
         'activation': ['relu'],
         'first_neuron': (50,3000,50),
         'last_neuron' : (50,3000,50),
         'hidden_layers': (0,10,10),
         'dropout': (0.01,0.99,100),
         'batch_size': (300,20000,100),
         'epochs': [500]}


    def football_model(x_train,y_train,x_val,y_val,params):
        model = Sequential()

        es_callback = EarlyStopping(monitor='val_loss', patience=3)

        model.add(Dense(params['first_neuron'], activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
        model.add(Dropout(params['dropout']))

        hidden_layers(model,params,params['last_neuron'])

        model.add(Dense(numLabels, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        threads = cpu_count()
        history = model.fit(x_train,y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0, callbacks=[es_callback],
              validation_data=(x_val,y_val), workers=threads, use_multiprocessing=True)

        return history,model

    t = ta.Scan(x= train_features,y = train_labels,x_val = val_features,y_val = val_labels,
                          model = football_model,params = p,experiment_name = 'football',round_limit=1000)

    p = Predict(t)



    preds = p.predict(test_features,metric='val_accuracy',asc = False).argmax(axis=1)

    acc = metrics.accuracy_score(test_labels.argmax(axis=1), preds) * 100

    base_acc = baseline_accuracy(test_labels, test_features, feature_list, ohe_home, lb)


    print('Baseline Accuracy: %.3f' % base_acc)
    print('Test Accuracy: %.3f' % acc)
    diff = (acc - base_acc)
    print('Model better than baseline by: %.3f pp' % diff)


if __name__ == '__main__':
    main()
