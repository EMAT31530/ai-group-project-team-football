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
from talos.utils.recover_best_model import recover_best_model
from talos import Predict, Analyze, Evaluate




def main(csv_path, n):
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

    train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels,test_size=0.33)

    n_features = train_features.shape[1]


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



    df = pd.read_csv(csv_path)

    n = n

    topn = df.nlargest(n,'val_accuracy')[['val_accuracy','batch_size','activation','kernel_initializer','dropout','first_neuron','hidden_layers','last_neuron','shapes']]
    best = df.nlargest(1,'val_accuracy')[['val_accuracy','batch_size','activation','kernel_initializer','dropout','first_neuron','hidden_layers','last_neuron','shapes']]
    print(topn['val_accuracy'])
    inp = input("gen or test")

    if inp == 'test':
        bestAcc = 0
        lossStop_callback = EarlyStopping(monitor='loss', patience=3)
        for row in tqdm(topn.itertuples(index = True),total=n):
            row_index = row[0]
            row_val_accuracy = row[1]
            row_batch_size = row[2]
            row_activation = row[3]
            row_kernel_initializer = row[4]
            row_dropout = row[5]
            row_first_neuron = row[6]
            row_hidden_layers = row[7]
            row_last_neuron = row[8]
            row_shapes = row[9]

            p_row = {'shapes': row_shapes,
                      'activation': row_activation,
                      'kernel_initializer': row_kernel_initializer,
                      'first_neuron': row_first_neuron,
                      'last_neuron': row_last_neuron,
                      'hidden_layers': row_hidden_layers,
                      'dropout': row_dropout,
                      'batch_size': row_batch_size,
                      'epochs': 500}

            history, model = football_model(train_features,train_labels,val_features,val_labels,p_row)
            loss, acc = model.evaluate(test_features, test_labels, batch_size=row_batch_size,verbose=0,callbacks=lossStop_callback)
            if acc>bestAcc:
                bestAcc = acc
                bestLoss = loss
                bestHistory = history


        print(bestAcc)

        plt.plot(bestHistory.history['loss'])
        plt.plot(bestHistory.history['val_loss'])
        plt.axhline(y = bestLoss,c = 'k', ls = '--')

        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation',f'Test = {bestLoss}'], loc='upper left')
        plt.grid()
        plt.show()
        plt.plot(bestHistory.history['accuracy'])
        plt.plot(bestHistory.history['val_accuracy'])
        plt.axhline(y=bestAcc, c='k', ls='--')

        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation',f'Test = {bestAcc}'], loc='upper left')
        plt.grid()
        plt.show()

        bestAcc *= 100
        base_acc = baseline_accuracy(lb.inverse_transform(test_labels))

        print('Baseline Accuracy: %.3f' % base_acc)
        print('Test Accuracy: %.3f' % bestAcc)
        diff = (bestAcc - base_acc)
        print('Model better than baseline by: %.3f pp' % diff)







    elif inp == 'gen':
        batch_size_topn = np.unique(np.array(topn['batch_size'])).tolist()
        activation_topn = np.unique(np.array(topn['activation'])).tolist()
        kernel_initializer_topn = np.unique(np.array(topn['kernel_initializer'])).tolist()
        dropout_topn = np.unique(np.array(topn['dropout'])).tolist()
        first_neuron_topn = np.unique(np.array(topn['first_neuron'])).tolist()
        hidden_layers_topn = np.unique(np.array(topn['hidden_layers'])).tolist()
        last_neuron_topn = np.unique(np.array(topn['last_neuron'])).tolist()
        shapes_topn = np.unique(np.array(topn['shapes'])).tolist()

        p_topn = {'shapes': shapes_topn,
                     'activation': activation_topn,
                     'kernel_initializer': kernel_initializer_topn,
                     'first_neuron': first_neuron_topn,
                     'last_neuron' : last_neuron_topn,
                     'hidden_layers': hidden_layers_topn,
                     'dropout': dropout_topn,
                     'batch_size': batch_size_topn,
                     'epochs': [500]}

        t = ta.Scan(x= train_features,y = train_labels,x_val = val_features,y_val = val_labels,
                              model = football_model,params = p_topn,experiment_name = 'football_best',round_limit=700)

    else:
        print("Please input either 'gen' or 'test' !!")
        main(csv_path,n)


if __name__ == '__main__':
    csv = r'D:\intro2ai\ai-group-project-team-football\football_best\051721202208.csv'
    main(csv,20)





