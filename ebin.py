import numpy
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import optimizers


class TitanicClassifier(object):
    """
    A Keras classifier / Pandas descriptives wrapper for
    a Kaggle contest where the object is to predict if
    a passenger in the test data survives the sinking or not
    """

    def __init__(self):
        # Assignments to satisfy pylint lul
        self.all_train_data, self.train_data, self.data_labels, self.test_data_size = [
            None, None, None, None]

        # Construct the computational graph
        self.model = Sequential()
        self.model.add(
            Dense(25, input_dim=5, init='normal', activation='relu'))
        self.model.add(
            Dense(120, input_dim=5, init='normal', activation='relu'))
        self.model.add(Dense(2, init='normal', activation='softmax'))

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        # Compile the neural network model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def load_train_data(self):
        """
        Loads Titanic train data to memory from train.csv
        """
        # Load the whole train data file
        self.all_train_data = pandas.read_csv(
            "data/train.csv", usecols=[0, 1, 2, 4, 5, 6, 9])

        # Drop any missing data
        self.all_train_data = self.all_train_data.dropna(
            axis=0, how='any', thresh=None, subset=None, inplace=False)

        # Replace sex labels with mapped values
        self.all_train_data = self.all_train_data.replace(
            to_replace='male', value=1)
        self.all_train_data = self.all_train_data.replace(
            to_replace='female', value=2)

        # Get remaining indeces
        self.train_indeces = self.all_train_data.filter(
            items=['PassengerId']).values[1:]

        ###################################
        # Load subdata for neural network #
        ###################################

        # Split boolean for survived (true, false) as training labels
        survived = self.all_train_data.filter(
            items=['Survived']).values[1:].astype(int)
        # Convert to categorical data (boolean values for both 0 and 1)
        self.train_labels = np_utils.to_categorical(survived)

        # Load correlating data points as training data
        self.train_data = self.all_train_data.filter(
            items=['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']).values[1:].astype(float)

    def train(self):
        """
        Fits the model to the train data
        """
        self.model.fit(self.train_data, self.train_labels,
                       nb_epoch=2048, batch_size=20, verbose=2)

    def descriptive_statistics(self):
        """
        Prints out basic descriptives of the distribution
        """
        print self.all_train_data.groupby('Survived').describe()
        self.all_train_data.groupby('Survived').hist()
        plt.show()
