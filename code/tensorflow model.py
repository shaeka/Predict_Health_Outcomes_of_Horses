### The competition the code is for: https://www.kaggle.com/competitions/playground-series-s3e22

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

dataPath = os.path.dirname(os.getcwd()) + '/data/playground-series-s3e22/'
data = pd.read_csv(dataPath + 'train.csv')
data = data[[x for x in data.columns if 'id' not in x and 'hospital_number' not in x]]

### Convert columns into numerical columns
encoder_dicts = {}
cleaned_data = pd.DataFrame()
numerical_data = pd.DataFrame()
min_max_scaler = preprocessing.MinMaxScaler()

for eachcol, eachtype in zip(data.dtypes.index, data.dtypes):
    if eachtype == 'object':
        temp_le = LabelEncoder()
        data['{}_encoded'.format(eachcol)] = temp_le.fit_transform(data[eachcol])
        encoder_dicts[eachcol] = temp_le
        cleaned_data['{}_encoded'.format(eachcol)] = data['{}_encoded'.format(eachcol)]
    elif 'int' in str(eachtype):
        numerical_data['{}_minmax'.format(eachcol)] = data[eachcol]
    else:
        cleaned_data['{}_original'.format(eachcol)] = data[eachcol]
        
numerical_data_scaled = min_max_scaler.fit_transform(numerical_data.values)        
numerical_data = pd.DataFrame(numerical_data_scaled, columns=numerical_data.columns)
cleaned_data = pd.concat([cleaned_data, numerical_data], axis=1)

X = cleaned_data[[x for x in cleaned_data.columns if 'outcome' not in x]]
y = cleaned_data[['outcome_encoded']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Use tensorflow to do classification
import tensorflow as tf
from tensorflow.keras import layers

tf.random.set_seed(42)

model = tf.keras.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1])),
    layers.Dense(2048, activation='relu'),
    # layers.Dropout(0.1),
    # layers.Dense(1024, activation='relu'),
    # layers.Dropout(0.1),
    # layers.Dense(512, activation='relu'),
    # layers.Dropout(0.1),
    # layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
    ], name='model')

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              metrics=['accuracy'])

model_history = model.fit(X_train.values,
                          y_train.values,
                          epochs=1000,
                          validation_data=(X_test.values, y_test.values),
                          verbose=0,
                          callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
                          )

y_proba = model.predict(X_test.values)
y_pred = y_proba.argmax(axis=1)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

### Apply on test data
test_data = pd.read_csv(dataPath + 'test.csv')
test_data_id = test_data['id']
test_data = test_data[[x for x in test_data.columns if 'id' not in x and 'hospital_number' not in x]]

cleaned_test_data = pd.DataFrame()
numerical_test_data = pd.DataFrame()
### Apply same encoders and min_max_scaler on the test data
test_data.loc[(test_data['pain'] == 'moderate'), 'pain'] = 'slight'

for eachcol, eachtype in zip(test_data.dtypes.index, test_data.dtypes):
    if eachtype == 'object':
        temp_le = encoder_dicts[eachcol]
        test_data['{}_encoded'.format(eachcol)] = temp_le.transform(test_data[eachcol])
        cleaned_test_data['{}_encoded'.format(eachcol)] = test_data['{}_encoded'.format(eachcol)]
    elif 'int' in str(eachtype):
        numerical_test_data['{}_minmax'.format(eachcol)] = test_data[eachcol]
    else:
        cleaned_test_data['{}_original'.format(eachcol)] = test_data[eachcol]
        
numerical_test_data_scaled = min_max_scaler.transform(numerical_test_data.values)        
numerical_test_data = pd.DataFrame(numerical_test_data_scaled, columns=numerical_test_data.columns)
cleaned_test_data = pd.concat([cleaned_test_data, numerical_test_data], axis=1)

y_test_proba = model.predict(cleaned_test_data.values)
y_test_pred = y_test_proba.argmax(axis=1)

### Save results
y_test_pred_transform = encoder_dicts['outcome'].inverse_transform(y_test_pred)
result = pd.concat([test_data_id, pd.DataFrame(y_test_pred_transform)], axis=1)
result.columns = ['id', 'outcome']
result.to_csv(os.path.dirname(os.getcwd()) + '/output/' + 'submission.csv', index=False)