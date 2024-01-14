import numpy as np
import pandas as pd

from copy import deepcopy
from math import inf

class TOPSIS:
    def __init__(self, generated_data, criterions_list, weights) -> None:
        self.generated_data = generated_data
        # self.criterions = criterions
        self.weights = np.array(weights)
        self.matrix = generated_data.to_numpy()
        self.criterionsList = criterions_list

    # def criterionsDfToList(self):
    #     numOfCriterions = self.criterions.shape[0]
    #     self.criterionsList = [1 if self.criterions.loc[i].Kierunek == 'Max' else 0 for i in range(numOfCriterions)]
    #     return self.criterionsList

    def normalize(self):
        self.matrix = self.matrix / ((self.matrix ** 2).sum(axis=0)) ** (1/2)
        # return self.matrix

    def applyMaxim(self):
        # one when maximalize
        # maxims 0 0 1
        v = [1 if x == 0 else -1 for x in self.criterionsList]
        # v2 1 1 -1
        self.matrix = self.matrix * v
        self.matrix = self.matrix + self.criterionsList
        # matrix a b c -> a 1-b 1-c
        # return self.matrix

    def applyWeights(self):
        self.matrix = self.matrix * self.weights
    
    def getMin(self):
        return self.matrix.min(0)

    def getMax(self):
        return self.matrix.max(0)

    def getDig(self, matMin):
        matrix = deepcopy(self.matrix)
        matrix = matrix - matMin
        matrix = matrix ** 2
        return sum(matrix.T)

    def getDid(self, matMax):
        matrix = deepcopy(self.matrix)
        matrix = matrix - matMax
        matrix = matrix ** 2
        return sum(matrix.T)

    def getCi(self, dig, did):
        return did / (did + dig)

    def sortRes(self, ci):
        listRes = list(enumerate(ci, 1))
        return sorted(listRes, key=lambda x: x[1], reverse=True)

    def printRank(self, rank):
        print(*rank, sep='\n')

    def get_max_min(self, struct):
        min_ = [inf] * 3
        max_ = [-inf] * 3
        for val in struct.values():
            for i in (0, 1, 2):
                min_[i] = min(min_[i], val[i])
                max_[i] = max(max_[i], val[i])
        return {"MIN": min_, "MAX": max_}

    def topsis(self, generated_data, criterions, weights):
        # self.criterionsDfToList()
        self.normalize()
        self.applyWeights()
        self.applyMaxim()
        min_ = self.getMin()
        max_ = self.getMax()
        did = self.getDid(max_)
        dig = self.getDig(min_)
        ci = self.getCi(dig, did)
        print(ci)
        rank = self.sortRes(ci)
        self.printRank(rank)
        return rank, ci

    def run(self):
        rank = self.topsis(self.generated_data, self.criterionsList, self.weights)
        return rank