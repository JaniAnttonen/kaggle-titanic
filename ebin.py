import numpy
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder


class TitanicClassifier(object):

    def __init__(self):
        self.model = Sequential()
        self.model.add(
            Dense(32, input_dim=5, init='normal', activation='relu'))
        self.model.add(Dense(2, init='normal', activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def load_train_data(self):
        self.all_train_data = pandas.read_csv(
            "data/train.csv", usecols=[1, 2, 4, 5, 6, 9])

        # Replace sex labels with mapped values
        self.all_train_data = self.all_train_data.replace(
            to_replace='male', value=1)
        self.all_train_data = self.all_train_data.replace(
            to_replace='female', value=2)

        # Drop any missing data
        self.all_train_data = self.all_train_data.dropna(
            axis=0, how='any', thresh=None, subset=None, inplace=False)

        # Load subdata for neural network
        survived = pandas.read_csv("data/train.csv", usecols=[1]).values[1:]
        survived = np_utils.to_categorical(survived)
        trainData = pandas.read_csv("data/train.csv", usecols=[2, 4, 5, 6, 9])
        trainData = trainData.replace(to_replace='male', value=1)
        trainData = trainData.replace(to_replace='female', value=2)

        self.trainData = trainData.values[1:].astype(float)
        self.dataLabels = survived.astype(float)

    def train(self):
        self.model.fit(self.trainData, self.dataLabels,
                       nb_epoch=10, batch_size=80, verbose=2)

    def descriptive_statistics(self):
        print self.all_train_data.groupby('Survived').describe()
        self.all_train_data.groupby('Survived').hist()
        plt.show()
