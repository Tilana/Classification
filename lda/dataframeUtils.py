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


