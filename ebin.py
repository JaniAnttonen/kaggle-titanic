import numpy
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


class TitanicClassifier(object):
    """
    A Keras classifier / Pandas descriptives wrapper for
    a Kaggle contest where the object is to predict if
    a passenger in the test data survives the sinking or not
    """

    def __init__(self):
        self.all_train_data, self.trainData, self.dataLabels = [None, None, None]

        self.model = Sequential()
        self.model.add(
            Dense(32, input_dim=5, init='normal', activation='relu'))
        self.model.add(Dense(2, init='normal', activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def load_train_data(self):
        """
        Loads Titanic train data to memory from train.csv
        """
        self.all_train_data = pandas.read_csv(
            "data/train.csv", usecols=[1, 2, 4, 5, 6, 9])

        # Drop any missing data
        self.all_train_data = self.all_train_data.dropna(
            axis=0, how='any', thresh=None, subset=None, inplace=False)

        # Replace sex labels with mapped values
        self.all_train_data = self.all_train_data.replace(
            to_replace='male', value=1)
        self.all_train_data = self.all_train_data.replace(
            to_replace='female', value=2)

        # Load subdata for neural network
        survived = self.all_train_data.filter(items=['Survived']).values[1:]
        survived = np_utils.to_categorical(survived)
        trainData = self.all_train_data.filter(items=['Pclass', 'Sex', 'Age', 'SibSp', 'Fare'])
        trainData = trainData.replace(to_replace='male', value=1)
        trainData = trainData.replace(to_replace='female', value=2)

        self.trainData = trainData.values[1:].astype(float)
        self.dataLabels = survived.astype(int)
        print self.trainData, self.dataLabels

    def train(self):
        """
        Fits the model to the train data
        """
        self.model.fit(self.trainData, self.dataLabels,
                       nb_epoch=50, batch_size=80, verbose=2)

    def descriptive_statistics(self):
        """
        Prints out basic descriptives of the distribution
        """
        print self.all_train_data.groupby('Survived').describe()
        self.all_train_data.groupby('Survived').hist()
        plt.show()
