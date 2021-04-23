import numpy as np
import pandas as pd
import joblib
import sqlite3
from sklearn.model_selection import train_test_split
from randomforest import optimiseElo
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
            "SELECT home_team_api_id,away_team_api_id,home_team_goal,away_team_goal,winner FROM Match")
    matches = cur.fetchall()
    trainMatches, testMatches = train_test_split(matches, test_size=0.25)

    resPath = r"D:\intro2ai\ai-group-project-team-football\res.pkl"

    optimiseElo(cur, trainMatches, testMatches,useSavedRes=True,useSavedResPath=resPath)

    cur.execute("SELECT elo,team_api_id FROM TEAM")
    team_elo_and_id = cur.fetchall()

    cur.executemany(f"UPDATE Match SET home_team_elo = ? WHERE home_team_api_id = ?", team_elo_and_id)
    cur.executemany(f"UPDATE Match SET away_team_elo = ? WHERE away_team_api_id = ?", team_elo_and_id)

    cur.execute("SELECT winner,home_team_api_id,away_team_api_id,home_team_elo,away_team_elo FROM Match")
    match_data = cur.fetchall()
    matches_df = pd.DataFrame(match_data, columns=["winner","home_team","away_team", "home_elo", "away_elo"])

    cur.execute("SELECT WHH,WHA,WHD,B365H,B365A,B365D,LBH,LBA,LBD FROM Match")
    gambling_odds = cur.fetchall()
    imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
    gambling_odds = imp_mean.fit_transform(gambling_odds)
    odds_cols = ["whh","wha","whd","b365h","b365a","b365d","lbh","lba","lbd"]
    matches_df[odds_cols] = gambling_odds

    scaler = MinMaxScaler()
    columns_to_norm = ['home_elo','away_elo','whh','wha','whd',"b365h","b365a","b365d","lbh","lba","lbd"]
    x = matches_df[columns_to_norm].values
    scaler.fit(x)
    joblib.dump(scaler,r"D:\intro2ai\ai-group-project-team-football\scaler.pkl")
    x_scaled = scaler.transform(x)
    df_temp = pd.DataFrame(x_scaled,columns = columns_to_norm,index = matches_df.index)
    matches_df[columns_to_norm] = df_temp

    matches_df["winner"] = matches_df["winner"].astype("str")
    matches_df["home_team"] = matches_df["home_team"].astype("str")+"_home"
    matches_df["away_team"] = matches_df["away_team"].astype("str")+"_away"

    labels = matches_df.filter(items=["winner"])
    lb = LabelBinarizer()
    lb.fit(labels["winner"])
    lb_trans = lb.transform(labels["winner"])
    labels = pd.DataFrame(lb_trans,columns = lb.classes_)
    label_list = labels.columns

    ohe_home = LabelBinarizer()
    ohe_away = LabelBinarizer()

    features = matches_df

    ohe_home.fit(features["home_team"])
    ohe_away.fit(features["away_team"])

    ohe_home_trans = ohe_home.transform(features["home_team"])
    ohe_away_trans = ohe_away.transform(features["away_team"])

    features = matches_df.drop(columns = ["winner","home_team","away_team"])

    ohe_home_df = pd.DataFrame(ohe_home_trans,columns=ohe_home.classes_)
    ohe_away_df = pd.DataFrame(ohe_away_trans,columns = ohe_away.classes_)

    features = features.join(ohe_home_df,how = "right")
    features = features.join(ohe_away_df,how = "right")
    feature_list = features.columns

    labels = np.array(labels).astype('float32')
    features = np.array(features).astype('float32')

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

    train_features, val_features, train_labels, val_labels = train_test_split(train_features,train_labels,test_size=0.2)

    n_features = train_features.shape[1]

    def model_builder(hp):
        model = Sequential()

        hp_dense1 = hp.Int('dense1', min_value=10, max_value=3000, step=10)
        hp_dense2 = hp.Int('dense2', min_value=10, max_value=3000, step=10)
        hp_dense3 = hp.Int('dense3', min_value=10, max_value=3000, step=10)


        hp_dropout1 = hp.Float('dropout1', min_value=0.01, max_value=0.99, step=0.01)
        hp_dropout2 = hp.Float('dropout2', min_value=0.01, max_value=0.99, step=0.01)
        hp_dropout3 = hp.Float('dropout3', min_value=0.01, max_value=0.99, step=0.01)


        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1,1e-2, 1e-3, 1e-4,1e-5])

        model.add(Dense(units=hp_dense1, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
        model.add(Dropout(hp_dropout1))
        model.add(Dense(units=hp_dense2, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(hp_dropout2))
        model.add(Dense(units=hp_dense3, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(hp_dropout3))

        model.add(Dense(300, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=500,
                         factor=3,
                         )


    es_callback = EarlyStopping(monitor='val_loss', patience=3)
    threads = cpu_count() - 1

    tuner.search(train_features, train_labels, epochs=300, batch_size=300, verbose=1, callbacks=[es_callback],
              validation_data=(val_features, val_labels), workers=threads)

    best_hps = tuner.get_best_hyperparameters(num_trials=10)[0]

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_features, train_labels, epochs=300, batch_size=300, verbose=1, callbacks=[es_callback],
              validation_data=(val_features, val_labels), workers=threads,use_multiprocessing=True)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

    model.fit(train_features, train_labels, epochs=best_epoch, batch_size=300, verbose=1, callbacks=[es_callback],
              validation_data=(val_features, val_labels), workers=threads,use_multiprocessing=True)

    loss, acc = model.evaluate(test_features, test_labels, verbose=1)
    acc*=100

    print(f"""
        dense1 = {best_hps.get('dense1')}\n
        dense2 = {best_hps.get('dense2')}\n
       
        
        drop1 = {best_hps.get('dropout1')}\n
        drop2 = {best_hps.get('dropout2')}\n
        
        
        learning_rate = {best_hps.get('learning_rate')}
        """)

    print('Best epoch: %d' % (best_epoch,))

    print('Test Accuracy: %.3f' % acc)


if __name__ == '__main__':
    main()





