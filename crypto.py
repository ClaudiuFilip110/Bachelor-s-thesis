import pandas as pd

# df = pd.read_csv('crypto_data/BCH-USD.csv', names=['time', 'low', 'high', 'open', 'close', 'volume'])
# print(df.head())

# read the csv

# create an empty main_df and add the close and volume from all of them
main_df = pd.DataFrame()

ratios = ["BCH-USD", "BTC-USD", "ETH-USD", "LTC-USD"]

for ratio in ratios:
    dataset = pd.read_csv(
        f"crypto_data/{ratio}.csv",
        names=["time", "low", "high", "open", "close", "volume"],
    )
    # we only need the close and volume
    dataset.rename(
        columns={"close": f"{ratio}-close", "volume": f"{ratio}-volume"}, inplace=True
    )
    # print(dataset.head())

    # set the time as the index
    dataset.set_index("time", inplace=True)

    # drop the ones we don't need
    dataset.drop(["low", "high", "open"], axis=1, inplace=True)

    # print(dataset.head())
    # merge all the columns on the index
    if main_df.empty:
        main_df = dataset
    else:
        main_df = main_df.join(dataset)

# now we have the table that we want
# print(main_df.head())

# print(main_df.columns)
# as we can see the time is the index, as we requested.

# define the parameters.
"""
The data that we have is updated every 1 min. so we will first try to predict the next 3 minutes based on the last 60 minutes. 
The algorithm will learn by shifting through the learning set. 
"""

SEQ_LEN = 60
FUTURE_PERIOD_TO_PREDICT = 3
COIN_TO_PREDICT = "BTC-USD"
VALIDATION_PCT = 0.05

# we will need a clasifier so that we will know whether the price goes up or down
def classify(current, future):
    if float(current) <= float(future):
        return 1
    else:
        return 0


# add a future column that will be the price in 3 mins
main_df["future"] = main_df[f"{COIN_TO_PREDICT}-close"].shift(-FUTURE_PERIOD_TO_PREDICT)
# print(main_df[[f'{COIN_TO_PREDICT}-close', 'future']].head())

# add a target column that will be classification of the price
# print(classify(main_df[f'{COIN_TO_PREDICT}-close'],main_df['future']))
main_df["target"] = list(
    map(classify, main_df[f"{COIN_TO_PREDICT}-close"], main_df["future"])
)
"""
Now we need to separate into training and validation sets.
We will do that with a 95 - 5 split.
We will not shuffle the dataset because that would not be in our advantage.
"""
times = sorted(main_df.index.values)
last_5_pct = times[-int(VALIDATION_PCT * len(times))]

validation_main_df = main_df[main_df.index >= last_5_pct]
main_df = main_df[main_df.index < last_5_pct]
# print(validation_main_df.head())
# print(main_df.head())
# print(main_df['future'].isna().sum() * 100 /len(main_df['future']))
# print(validation_main_df['future'].isna().sum() * 100 /len(validation_main_df['future']))
""" 
as we can see we have a looot of missing values, about 5.8% of them in the training set and 0.08% in the validation set
so we will have to figure a way how to deal with them.
"""

"""
Now that we've separated the data into validation and training sets, should we shuffle the datasets?
The answer is yes. WHY? Because the model will overfit really quickly if the data is in order. We can safely 
shuffle it now because the order won't matter now.

This preprocessing part is really hard and I don't understand it completly myself yet.
"""

from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time

# we need a function that processes the data so we can use it in the model
# normalize and scale the data
"""
We use pct_change so we only use the change in percentage, not the value itself.. that will be useful when 
the NN tries to learn from COIN to COIN.
"""

def preprocessing_df(df):
    df.drop("future", axis=1)  # we drop the future column

    # use pct change to figure the change in price
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)

            # normalize the data.. NOTE: normalization is a form of scaling
            df[col] = preprocessing.scale(df[col])
    df.dropna(inplace=True)

    # the last bit of preprocessing we need to do is balancing the dataset.
    # if we leave it like this it will not learn very well.

    """
    Basically, with SEQ_LEN we are going to create batches of data.. i believe
    I was wrong. figure that.
    My next theory is that this is used to split the data in 1 hour data.
    """
    sequencial_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for row in df.values:
        prev_days.append([x for x in row[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequencial_data.append([np.array(prev_days), row[-1]])

    random.shuffle(sequencial_data)

    """
    the last step in our preprocessing will be to balance the learning set
    """

    buys = []
    sells = []

    for seq, target in sequencial_data:
        if target == 0:
            sells.append([seq, target])
        else:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    # how do we balance? Simple. we remove the excess

    lower_nr = min(len(buys), len(sells))

    buys = buys[:lower_nr]
    sells = sells[:lower_nr]

    sequencial_data = buys + sells

    random.shuffle(sequencial_data)

    # split into x and Y

    x = []
    Y = []

    for seq, target in sequencial_data:
        x.append(seq)
        Y.append(target)
    return np.array(x), np.array(Y)

train_x, train_Y = preprocessing_df(main_df)
test_x, test_Y = preprocessing_df(validation_main_df)

#Use these to verify that your data is split correctly.
# print(f"Dont buys: {train_Y.count(0)}, buys: {train_Y.count(1)}")
# print(f"validation dont buys: {test_Y.count(0)} buys: {test_Y.count(1)}")

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
# pt CuDNNLSTM look at this https://stackoverflow.com/questions/60468385/is-there-cudnnlstm-or-cudnngru-alternative-in-tensorflow-2-0

'''
model:
lstm
dropout
normalization
x3
dense
dropout
optimizer --
loss -------
accuracy ---
'''

EPOCHS = 10
BATCH_SIZE = 32
NAME = f"{COIN_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_TO_PREDICT}-PRED-{int(time.time())}"

model = Sequential()
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = Adam(learning_rate=0.001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(x=train_x, y=train_Y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_x, test_Y))

# model.save('')