import os
import pandas as pd
import numpy as np

def getRow(df, colname, value, columns):
    return list(df.loc[df[colname]==value, columns].values[0])

def getColumn(df, colname):
    columnValues = list(df[colname].unique())
    return [value for value in columnValues if value != 'nan']

def filterData(df, colname):
    return df[df[colname]]

def getIndex(df):
    return df.index.tolist()

def tolist(df, column):
    return df[column].values.tolist()

def toListMultiColumns(df, columnList):
    result = set() 
    for col in columnList:
        result.update(tolist(df, col))
    return result

def getValue(df, column):
    value = df[column].tolist()
    if value != []:
        return value[0]
    else:
        return 'nan'

def createNumericFeature(df, column):
    category = 0
    for value in df[column].unique():
        df.loc[df[column]==value, column] = category
        category += 1
    toNumeric(df, column)

def toNumeric(df, column):
    df[column] = df[column].astype(int)

def changeStringsInColumn(df, column, old, new):
    oldValues = df[column].tolist()
    newValues = [string.replace(old, new) for string in oldValues]
    df[column] = newValues


def save(df, path):
    df.to_pickle(path)

def toCSV(df, path):
    createDirectory(path)
    df.to_csv(path)

def createDirectory(path):
    directories = path.split('/')
    path = '/'.join(directories[0:len(directories)-1])
    if not os.path.exists(path):
        os.makedirs(path)

def arrayColumnToDataframe(dataframe, col):
    data = dataframe[col]
    columnNames = createColumnNames(col, len(data[0]))
    test = pd.DataFrame([elem for elem in data], columns=columnNames)
    return test


def createColumnNames(name, n):
    return [name+str(number) for number in range(n)]


def flattenDataframe(dataframe):
    flatDF = pd.DataFrame(dataframe)
    arrayColumns = getArrayColumns(dataframe)
    for col in arrayColumns:
        flatCol = arrayColumnToDataframe(dataframe, col)
        flatDF.drop(col, axis=1, inplace=True)
        flatDF = pd.concat([flatDF,flatCol], axis=1)
    return flatDF 

def getArrayColumns(data):
    arrayColumns = []
    for col in data.columns:
        columnList = data[col].tolist()
        if type(columnList[0])==list: 
            arrayColumns.append(col)
    return arrayColumns

def combineColumnValues(dataframe, columns):
    values = dataframe[columns].as_matrix()
    valueList = [flattenArray(array) for array in values]
    return valueList 

def flattenArray(array):
    flatArray = []
    for elem in array:
        if type(elem) in (list,tuple,np.ndarray):
            [flatArray.append(value) for value in elem]
        else:
            flatArray.append(elem)
    return flatArray 


    

