import numpy as np # to use numpy arrays instead of lists
import pandas as pd # DataFrame (table)
import matplotlib.pyplot as plt # to plot

from sklearn import preprocessing # to normalize data
from sklearn.model_selection import train_test_split # to split data in train and test sets

np.random.seed(0)

# --------------------
def leeDatos(fileName):
    """
    Read data from a CSV file
    """

    dataSet = pd.read_csv(fileName, header=0, index_col=0)

    return dataSet

# --------------------
def graficaDatos(dataSet=0, label='', rangeScale=1, axis_label=''):

    plt.figure(figsize=(15, 6))
    for i in range(len(dataSet.columns)):
        plt.plot(dataSet.iloc[:,i]*rangeScale,label=dataSet.columns[i])
    plt.grid(color='b',linestyle='dashed')
    plt.gcf().autofmt_xdate()
    plt.title(label, fontweight='bold')
    plt.xlabel('Date (t)')
    plt.ylabel(axis_label +' Cases')
    plt.legend(loc='upper left')
    plt.show()

# --------------------
def selectData(dataSet=0):

    newDataset = dataSet.select_dtypes(exclude=['object']).copy()

    newDataset['SQ_Low'] = [1 if x == 'Low' else 0 for x in dataSet['Sleep_Quality']]
    newDataset['SQ_Fair'] = [1 if x == 'Fair' else 0 for x in dataSet['Sleep_Quality']]
    newDataset['SQ_Good'] = [1 if x == 'Good' else 0 for x in dataSet['Sleep_Quality']]
    newDataset['SQ_Excellent'] = [1 if x == 'Excellent' else 0 for x in dataSet['Sleep_Quality']]

    return newDataset

# --------------------
def preprocess(dataSet=0):
    """
    Preprocess data
    """
    
    # fit the scaler
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(dataSet)

    # Transform the data using the fitted scaler
    np_dataSet = scaler.transform(dataSet)

    # Convert the numpy array back to a DataFrame
    dataSet = pd.DataFrame(np_dataSet, columns=dataSet.columns, index=dataSet.index)

    # Fill NaN values after rolling mean
    dataSet = dataSet.fillna(method='bfill')

    return dataSet

# --------------------
def describeData(dataSet=0):
    """
    Data descriptive statistics
    """
    pd.set_option.precision = 4

    print(dataSet.head())
    print("\n")
    print(dataSet.info())
    print("\n")
    print(dataSet.describe())
    print("\n")
    print(dataSet.nunique())
    print("\n")


# --------------------
# split data
def splitDataSet(dataSet=0, test_size=.2, randSplit=True):
    """
    Split data in train and test sets
    """

    train, test = train_test_split(dataSet, test_size=test_size, shuffle=randSplit, random_state=0)

    return [train, test]