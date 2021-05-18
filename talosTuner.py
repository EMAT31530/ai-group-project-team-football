import numpy as np
import pandas as pd
import joblib
import sqlite3
from sklearn.model_selection import train_test_split
from randomforest import optimiseElo,setAttackDefence,baseline_accuracy
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

    cur.execute(f"SELECT elo,team_api_id FROM Team WHERE team_api_id IN {team_ids}")
    team_elo_and_id = cur.fetchall()


    cur.executemany(f"UPDATE Match SET home_team_elo = ? WHERE home_team_api_id = ?", team_elo_and_id)
    cur.executemany(f"UPDATE Match SET away_team_elo = ? WHERE away_team_api_id = ?", team_elo_and_id)

    setAttackDefence(cur,set=False)

    cur.execute(f"SELECT attack,defence,team_api_id FROM Team WHERE team_api_id IN {team_ids}")
    team_attack_defense_id = cur.fetchall()

    cur.executemany(f"UPDATE MATCH SET home_attack = ?,home_defence = ? WHERE home_team_api_id = ?",team_attack_defense_id)
    cur.executemany(f"UPDATE MATCH SET away_attack = ?,away_defence = ? WHERE away_team_api_id = ?",team_attack_defense_id)


    cur.execute(
        "SELECT winner,home_team_elo,away_team_elo,home_attack,home_defence,"
        "away_attack,away_defence FROM Match WHERE league_id = '1729'")
    match_data = cur.fetchall()
    matches_df = pd.DataFrame(match_data, columns=["winner", "home_elo", "away_elo",
                                                   "home_attack", "home_defence", "away_attack", "away_defence"])
    scaler = MinMaxScaler()
    columns_to_norm = ['home_elo', 'away_elo', 'home_attack', 'home_defence', 'away_attack',
                       'away_defence']  # + odds_cols
    x = matches_df[columns_to_norm].values
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=columns_to_norm, index=matches_df.index)
    matches_df[columns_to_norm] = df_temp

    labels = matches_df.filter(items=["winner"])
    lb = LabelBinarizer()
    lb.fit(labels["winner"])
    lb_trans = lb.transform(labels["winner"])
    labels = pd.DataFrame(lb_trans, columns=lb.classes_)
    label_list = labels.columns
    numLabels = len(label_list)


    features = matches_df.drop(columns=["winner"])

    feature_list = features.columns

    labels = np.array(labels).astype('float32')
    features = np.array(features).astype('float32')

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

    train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels,test_size=0.2)

    n_features = train_features.shape[1]

    p = {'shapes': ['brick', 'triangle', 'funnel'],
         'activation': ['relu'],
         'kernel_initializer': ['he_normal','glorot_normal','uniform'],
         'first_neuron': (50,3000,50),
         'last_neuron': (50,3000,50),
         'hidden_layers': (0,10,10),
         'dropout': (0.01,0.99,100),
         'batch_size': (300,20000,100),
         'epochs': [500]}

    p_best = {'shapes': ['triangle'],
             'activation': ['relu'],
             'kernel_initializer': ['uniform'],
             'first_neuron': [2823],
             'last_neuron' : [50],
             'hidden_layers': [5],
             'dropout': [0.2354],
             'batch_size': [694],
             'epochs': [500]}

    def football_model(x_train,y_train,x_val,y_val,params):
        model = Sequential()

        es_callback = EarlyStopping(monitor='val_loss', patience=3)

        model.add(Dense(params['first_neuron'], activation=params['activation'], kernel_initializer=params['kernel_initializer'], input_shape=(n_features,)))
        model.add(Dropout(params['dropout']))

        hidden_layers(model,params,params['last_neuron'])

        model.add(Dense(numLabels, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        threads = cpu_count()
        history = model.fit(x_train,y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0, callbacks=[es_callback],
              validation_data=(x_val,y_val), workers=threads, use_multiprocessing=True)

        return history,model

    t = ta.Scan(x= train_features,y = train_labels,x_val = val_features,y_val = val_labels,
                          model = football_model,params = p_best,experiment_name = 'football',round_limit=1)

    p = Predict(t)

    preds = p.predict(test_features,metric='val_accuracy',asc = False).argmax(axis=1)

    acc = metrics.accuracy_score(test_labels.argmax(axis=1), preds) * 100

    base_acc = baseline_accuracy(test_labels)

    print('Baseline Accuracy: %.3f' % base_acc)
    print('Test Accuracy: %.3f' % acc)
    diff = (acc - base_acc)
    print('Model better than baseline by: %.3f pp' % diff)


if __name__ == '__main__':
    main()
