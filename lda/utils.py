#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unittest import TestCase
import numpy as np

def flattenList(l):
	return [item for sublist in l for item in sublist]

def lowerList(wordList):
        return [word.lower() for word in wordList]

def listDifference(l):
    return [(elem[1]-elem[0], elem[0]) for elem in zip(l[:-1],l[1:])]

def removeAll(l, value):
    return [elem for elem in l if elem != value]

def containsAny(str, specialChars):
    for letter in str:
        if letter in specialChars:
            return 1;
    return 0;

def absoluteTupleList(tupleList):
    return [(abs(elem[0]), abs(elem[1])) for elem in tupleList]


def joinSublists(l1, l2):
    resultList = l1
    for index, sublist in enumerate(l2):
        [resultList[index].append(elem) for elem in sublist]
    return sortSublist(resultList)


def sortSublist(l):
    return [sorted(sublist) for sublist in l]


def countOccurance(text, l):
    return [(word, text.lower().count(word)) for word in l if text.lower().count(word)>0]


def listToNumpy(l):
    return np.asarray(l)

def histogram(l):
    return np.histogram(listToNumpy(l), range=(0,1))

def getMedian(l):
    return np.median(listToNumpy(l))

def getMean(l):
    return np.mean(listToNumpy(l))

def indicesOfReverseSorting(indices):
    return np.argsort(listToNumpy(indices))[::-1]

def getUpperSymmetrixMatrix(matrix):
    matrix = listToNumpy(matrix)
    return list(matrix[np.triu_indices(len(matrix), 1)])

def sortTupleList(tupleList):
    return sorted(tupleList, reverse=True, key=lambda x: x[1])

def getBigrams(wordList):
    return [tuple(word.split()) for word in wordList if ' ' in word]

def convertTupleToString(tupleList):
    return [(' '.join(elem[0]), elem[1]) for elem in tupleList]

