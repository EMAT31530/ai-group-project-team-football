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

def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

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

    cur.execute("SELECT WHH,WHA,WHD FROM Match")
    gambling_odds = cur.fetchall()
    imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
    gambling_odds = imp_mean.fit_transform(gambling_odds)
    odds_cols = ["whh","wha","whd"]
    matches_df[odds_cols] = gambling_odds

    scaler = MinMaxScaler()
    columns_to_norm = ['home_elo','away_elo','whh','wha','whd']
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

    model = Sequential()
    es_callback = EarlyStopping(monitor='val_loss', patience=3)

    model.add(Dense(1000, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(Dropout(0.7))
    model.add(Dense(800, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.7))
    model.add(Dense(300, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_features, train_labels, epochs=150, batch_size=300,verbose=1,callbacks=[es_callback],validation_data=(val_features,val_labels))

    #model.save(r"D:\intro2ai\ai-group-project-team-football\model")

    loss, acc = model.evaluate(test_features, test_labels, verbose=1)

    print('Test Accuracy: %.3f' % acc)


if __name__ == '__main__':
    main()





