import os

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split

def create_dataset():
    file_out = pd.read_csv("./dataset/dataset2.csv")
    x = file_out.iloc[0:301, 1:9].values   # NOT CONSIDERING THE PROJECT_ID COLUMN
    y = file_out.iloc[0:301, 8].values

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    sc = Normalizer()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    if os.path.exists("D:\Concordia Courses\ENCS 6931 - Industrial Stage & Training\Project\ANN_model.h5"):

        ann = tf.keras.models.load_model("ANN_model.h5")

    else:

        # Initialising ANN
        ann = tf.keras.models.Sequential()

        # Layers
        leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.01)
        ann.add(tf.keras.layers.Dense(units=25, activation="relu"))
        ann.add(tf.keras.layers.Dense(units=25, activation=leakyrelu))

        # Output layer
        ann.add(tf.keras.layers.Dense(units=1, activation="linear"))

        # Compiling ANN
        ann.compile(optimizer="adam", loss="mse", metrics=tf.keras.metrics.RootMeanSquaredError())

        # Fitting ANN
        ann.fit(X_train, Y_train, batch_size=32, epochs=100)

        # Saving created neural network
        ann.save("ANN_model.h5")



    # EVALUATION of the keras model
    _, accuracy = ann.evaluate(X_test, Y_test)
    print('Accuracy: %.2f' % (accuracy))

    # Scale for parameters of complexity calculation:
    # design change - 10p
    # config. change - 7p
    # no. of system - 5p
    # store change - 2p

    # No. of teams, Team size, design ch, config. ch, store ch,
    # No. of systems, Actual duration, Delayed days
    nt = 6
    ts = 19
    dc = 2
    cc = 1
    stc = 0
    nos = 2
    ad = 114

    #complexity: in percentage
    complexity = ((dc / 10) + (cc / 7) + (nos / 5) + (stc / 2))/10 * 100
    print('complexity of project: %.2f percent' % (complexity))

    # Predicting result for Single Observation
    predicted_delay = ann.predict(sc.transform([[nt, ts, dc, cc, stc, nos, _, _]]))
    print('predicted delayed time is %.2f days' % (predicted_delay))

    # risk factor scale: 0-1
    risk_factor = (predicted_delay) / complexity
    print('risk factor for this project: %.2f' % (risk_factor))


if __name__ == "__main__":
    create_dataset()
