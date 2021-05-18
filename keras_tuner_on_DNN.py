import numpy as np
import pandas as pd
import joblib
import sqlite3
from sklearn.model_selection import train_test_split
from randomforest import optimiseElo,baseline_accuracy
from sklearn.preprocessing import MinMaxScaler,LabelBinarizer
from sklearn.impute import SimpleImputer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import save_model,load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
from multiprocessing import cpu_count
from tensorflow.keras.optimizers import Adam
import kerastuner as kt
from randomforest import setAttackDefence
from elo2 import train


def baseline_accuracy(test_labels,test_features,feature_list,ohe_home,lb):
    test_features_df = pd.DataFrame(test_features, columns=feature_list)
    home = test_features_df.filter(regex = "(.*)_home")

    true = lb.inverse_transform(test_labels)
    base_preds = ohe_home.inverse_transform(home.values)
    base_preds = [pred.replace("_home","") for pred in base_preds]

    baseline_accuracy = metrics.accuracy_score(true,base_preds)*100
    return baseline_accuracy


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

    def model_builder(hp):
        model = Sequential()


        hp_dense1 = hp.Int('dense1', min_value=1, max_value=100000, step=10)
        hp_dense2 = hp.Int('dense2', min_value=1, max_value=100000, step=10)
        hp_dense3 = hp.Int('dense3', min_value=1, max_value=100000, step=10)

        hp_dropout1 = hp.Float('dropout1', min_value=0.01, max_value=0.99, step=0.01)
        hp_dropout2 = hp.Float('dropout2', min_value=0.01, max_value=0.99, step=0.01)
        hp_dropout3 = hp.Float('dropout3', min_value=0.01, max_value=0.99, step=0.01)

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1,1e-2, 1e-3, 1e-4,1e-5,1e-6])

        model.add(Dense(units=hp_dense1, activation='elu', kernel_initializer='he_normal', input_shape=(n_features,)))
        model.add(Dropout(hp_dropout1))
        model.add(Dense(units=hp_dense2, activation='elu', kernel_initializer='he_normal'))
        model.add(Dropout(hp_dropout2))
        model.add(Dense(units=hp_dense2, activation='elu', kernel_initializer='he_normal'))
        model.add(Dropout(hp_dropout3))
        model.add(Dense(units=hp_dense3, activation='elu', kernel_initializer='he_normal'))
        model.add(Dropout(hp_dropout3))

        model.add(Dense(numLabels, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=500,
                         factor=2,
                         hyperband_iterations = 1000
                         )


    es_callback = EarlyStopping(monitor='val_loss', patience=3)
    threads = cpu_count()

    batchSize = 10000

    tuner.search(train_features, train_labels, epochs=300, batch_size=batchSize, verbose=1, callbacks=[es_callback],
              validation_data=(val_features, val_labels), workers=threads)

    best_hps = tuner.get_best_hyperparameters(num_trials=100)[0]

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_features, train_labels, epochs=300, batch_size=batchSize, verbose=1, callbacks=[es_callback],
              validation_data=(val_features, val_labels), workers=threads,use_multiprocessing=True)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

    model.fit(train_features, train_labels, epochs=best_epoch, batch_size=batchSize, verbose=1, callbacks=[es_callback],
              validation_data=(val_features, val_labels), workers=threads,use_multiprocessing=True)

    loss, acc = model.evaluate(test_features, test_labels, verbose=1)
    acc*=100

    print(f"""
        dense1 = {best_hps.get('dense1')}\n
        dense2 = {best_hps.get('dense2')}\n
        dense3 = {best_hps.get('dense3')}\n

        drop1 = {best_hps.get('dropout1')}\n
        drop2 = {best_hps.get('dropout2')}\n
        drop3 = {best_hps.get('dropout3')}\n
 
        
        
        learning_rate = {best_hps.get('learning_rate')}
        """)

    print('Best epoch: %d' % (best_epoch,))

    print('Test Accuracy: %.3f' % acc)


if __name__ == '__main__':
    main()





