import numpy
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

class TitanicClassifier(object):
  def __init__(self):
    self.model = Sequential()
    self.model.add(Dense(32, input_shape=(5,), init='normal', activation='relu'))
    self.model.add(Dense(128, activation='relu'))
    self.model.add(Dense(1, activation='sigmoid'))
    # Compile model
    self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  def load_train_data(self):
    self.all_train_data = pandas.read_csv("data/train.csv", usecols=[1,2,4,5,6,9])

    # Replace sex labels with mapped values
    self.all_train_data = self.all_train_data.replace(to_replace='male', value=1)
    self.all_train_data = self.all_train_data.replace(to_replace='female', value=2)

    # Drop any missing data
    self.all_train_data = self.all_train_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    # Load subdata for neural network
    survived = pandas.read_csv("data/train.csv", usecols=[1])
    trainData = pandas.read_csv("data/train.csv", usecols=[2,4,5,6,9])
    trainData = trainData.replace(to_replace='male', value=1)
    trainData = trainData.replace(to_replace='female', value=2)

    self.trainData = trainData
    self.dataLabels = survived

  def train(self):
    self.model.fit(self.trainData.values[1:], self.dataLabels.values[1:], nb_epoch=22, batch_size=10)

  def descriptive_statistics(self):
    print self.all_train_data.groupby('Survived').describe()
    self.all_train_data.groupby('Survived').hist()
    plt.show()

def main(argv=None):
  classifier = TitanicClassifier()
  classifier.load_train_data()
  # classifier.train()
  classifier.descriptive_statistics()

  # load test data
  testData = pandas.read_csv("data/test.csv", header=None, usecols=[1,3,4,5,8])
  testData = testData.replace(to_replace='male', value=1)
  testData = testData.replace(to_replace='female', value=2)

  # Slice the test data without column labels
  testData = testData.values[1:6]

  # Predict the scores
  # results = classifier.model.predict(testData)

  #print results

if __name__ == '__main__':
  main()
