import numpy
import pandas
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
    survived = pandas.read_csv("data/train.csv", header=None, usecols=[1])
    trainData = pandas.read_csv("data/train.csv", header=None, usecols=[2,4,5,6,9])
    sexlabels = {'male': 1, 'female': 2}
    trainData = trainData.replace(to_replace='male', value=1)
    trainData = trainData.replace(to_replace='female', value=2)

    self.trainData = trainData.values[1:]
    self.dataLabels = survived.values[1:]

    print self.trainData, self.dataLabels

  def train(self):
    self.model.fit(self.trainData, self.dataLabels, nb_epoch=22, batch_size=10)

def main(argv=None):
  classifier = TitanicClassifier()
  classifier.load_train_data()
  classifier.train()

  # load test data
  testData = pandas.read_csv("data/test.csv", header=None, usecols=[1,3,4,5,8])
  testData = testData.replace(to_replace='male', value=1)
  testData = testData.replace(to_replace='female', value=2)

  # Slice the test data without column labels
  testData = testData.values[1:6]

  # Predict the scores
  results = classifier.model.predict(testData)

  print results

if __name__ == '__main__':
  main()
