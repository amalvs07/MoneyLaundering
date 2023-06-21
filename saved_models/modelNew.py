import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

logging.basicConfig(filename='logs/model_development.txt',
                    filemode='a',
                    format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("----------")
logging.warning("MODEL CREATION STAGE")

logging.warning("Reading Final Dataset...")

data_path = "data_processed/final_data.csv"
dataMat = pd.read_csv(data_path)

logging.warning("Read Final Dataset")

logging.warning("Checking Categorical Features...")

cat_feat = [i for i in dataMat.columns if dataMat[i].dtypes == 'O']

logging.warning("Checking Missing Values...")

a = dict(dataMat.isnull().sum())
b = [[i, a[i]] for i in a.keys()]
missing = pd.DataFrame(b, columns=['features', 'null_values_count'])

logging.warning("Storing Missing Values...")

missing.to_csv("reports/missing_values.csv", index=False)

logging.warning("Storing Missing Values Done")

logging.warning("Encoding Categorical Features...")

encoder = LabelEncoder()
for i in cat_feat:
    dataMat[i] = encoder.fit_transform(dataMat[i])

logging.warning("Features Encoding Done")

logging.warning("Creating X and y variables ...")

X = dataMat.iloc[:, :-1].values
y = dataMat['isFraud'].values

logging.warning(f"Shape of X: {X.shape} and Shape of y: {y.shape}")

logging.warning("Splitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logging.warning("Reshaping X_train and X_test for LSTM...")

# Reshape input data for LSTM
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

logging.warning("Instantiating LSTM Model...")

modelLSTM = Sequential()
modelLSTM.add(LSTM(128, input_shape=(1, X.shape[1])))
modelLSTM.add(Dense(1, activation='sigmoid'))

logging.warning("Compiling and Fitting LSTM Model...")

modelLSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
modelLSTM.fit(X_train, y_train, epochs=10, batch_size=32)

logging.warning("Saving Model...")

model_path = "saved_models/modelNew.pkl"
pickle.dump(modelLSTM, open(model_path, 'wb'))

logging.warning("Saving Model Metrics...")

metric_file_path = "reports/performance.json"
with open(metric_file_path, "r") as f:
    data = json.load(f)

y_pred_lstm = np.argmax(modelLSTM.predict(X_test), axis=-1)

model_metric = {
    "time_stamp": datetime.now().strftime("%d-%m-%Y_%H:%M:%S"),
    "confusion_matrix": confusion_matrix(y_test, y_pred_lstm).tolist(),
    "precision": precision_score(y_test, y_pred_lstm,zero_division=1),
    "recall": recall_score(y_test, y_pred_lstm),
    "f1_score": f1_score(y_test, y_pred_lstm)
}

data['model_metric'].append(model_metric)
with open(metric_file_path, "w") as f:
    json.dump(data, f, indent=4)

logging.warning("Model Metrics Stored")
