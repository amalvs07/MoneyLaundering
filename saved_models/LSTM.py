import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Load data
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

# Preprocess data
df.drop(['step', 'nameOrig', 'nameDest'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['type'])
le = LabelEncoder()
df['isFraud'] = le.fit_transform(df['isFraud'])
scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Define LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(1, df.shape[1]-1)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
model.fit(X, y, epochs=10, batch_size=32)

# Take user input
user_input = pd.DataFrame(columns=['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])
user_input.loc[0] = ['TRANSFER', 10000, 50000, 40000, 10000, 20000]
user_input = pd.get_dummies(user_input, columns=['type'])
user_input = scaler.transform(user_input)

# Reshape input for LSTM input
user_input = np.reshape(user_input, (user_input.shape[0], 1, user_input.shape[1]))

# Predict using trained model on user input
prediction = model.predict(user_input)
prediction = (prediction > 0.5)

# Print prediction
if prediction[0][0]:
    print('Fraudulent transaction detected')
else:
    print('Transaction is not fraudulent')
