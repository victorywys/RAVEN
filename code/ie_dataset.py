import os
import torch
import torch.utils.data as Data
import numpy as np
import cPickle as pickle

import ie_data_loader
from ie_data_loader import fetch_data

from consts import global_consts as gc

def mid(a):
    return (a[0] + a[1]) / 2.0

class IESubdata():
    def __init__(self, name = "train"):
        self.name = name
        self.wordInput = []
        self.wordLen = []
        self.facetInput = []
        self.facetLen = []
        self.covarepInput = []
        self.covarepLen = []
        self.labelOutput = []

class IEDataset(Data.Dataset):
    trainset = IESubdata("train")
    testset = IESubdata("test")
    validset = IESubdata("valid")

    def __init__(self, root, cls="train", src="csd", save=False):
        self.root = root
        self.cls = cls
        if len(IEDataset.trainset.labelOutput) != 0:
            print "Data has been preiviously loaded, fetching from previous lists."
        else:
            dataset = IEDataset.trainset
            dataset.wordInput, dataset.wordLen, dataset.covarepInput, dataset.covarepLen, dataset.facetInput, dataset.facetLen, dataset.labelOutput = fetch_data("train")
            dataset = IEDataset.testset
            dataset.wordInput, dataset.wordLen, dataset.covarepInput, dataset.covarepLen, dataset.facetInput, dataset.facetLen, dataset.labelOutput = fetch_data("test")
            dataset = IEDataset.validset
            dataset.wordInput, dataset.wordLen, dataset.covarepInput, dataset.covarepLen, dataset.facetInput, dataset.facetLen, dataset.labelOutput = fetch_data("valid")

            gc.wordDim = len(IEDataset.trainset.wordInput[0][0])
            gc.facetDim = len(IEDataset.trainset.facetInput[0][0][0])
            gc.covarepDim = len(IEDataset.trainset.covarepInput[0][0][0])
        if self.cls == "train":
            self.dataset = IEDataset.trainset
        elif self.cls == "test":
            self.dataset = IEDataset.testset
        elif self.cls == "valid":
            self.dataset = IEDataset.validset
        if gc.debug:
            self.covarepInput = self.dataset.covarepInput[:min(20, len(self.dataset.covarepInput))]
            self.wordInput = self.dataset.wordInput[:min(20, len(self.dataset.wordInput))]
            self.facetInput = self.dataset.facetInput[:min(20, len(self.dataset.facetInput))]
            self.labelOutput = self.dataset.labelOutput[:min(20, len(self.dataset.labelOutput))]
        else:
            self.covarepInput = self.dataset.covarepInput[:]
            self.covarepLen = self.dataset.covarepLen[:]
            self.wordInput = self.dataset.wordInput[:]
            self.wordLen = self.dataset.wordLen[:]
            self.facetInput = self.dataset.facetInput[:]
            self.facetLen = self.dataset.facetLen[:]
            self.labelOutput = self.dataset.labelOutput[:]

    def __getitem__(self, index):
        return torch.zeros(1),\
            torch.tensor(self.wordInput[index], dtype=torch.float),\
            torch.tensor(self.covarepInput[index], dtype=torch.float),\
            torch.tensor(self.covarepLen[index], dtype=torch.long),\
            torch.tensor(self.facetInput[index], dtype=torch.float),\
            torch.tensor(self.facetLen[index], dtype=torch.long),\
            self.wordLen[index],\
            torch.tensor(self.labelOutput[index][0])

    def __len__(self):
        return len(self.labelOutput)

if __name__ == "__main__":
    dataset = IEDataset('/home/victorywys/Documents/data/', src="csd", save=False)
