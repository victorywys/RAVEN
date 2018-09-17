import os
import torch
import torch.utils.data as Data
import numpy as np
import cPickle as pickle

from MOSI_data_loader import load_word_level_features

from consts import global_consts as gc

def mid(a):
    return (a[0] + a[1]) / 2.0

class MOSISubdata():
    def __init__(self, name = "train"):
        self.name = name
        self.covarepInput = []
        self.covarepLength = []
        self.wordid = []
        self.wordInput = []
        self.wordLength = []
        self.facetInput = []
        self.facetLength = []
        self.labelOutput = []

class MOSIDataset(Data.Dataset):
    trainset = MOSISubdata("train")
    testset = MOSISubdata("test")
    validset = MOSISubdata("valid")

    def __init__(self, root, cls="train", src="csd", save=False):
        self.root = root
        self.cls = cls
        if len(MOSIDataset.trainset.labelOutput) != 0 and cls != "train":
            print "Data has been preiviously loaded, fetching from previous lists."
        else:
            train, valid, test = load_word_level_features(gc.padding_len, 1.0)
            dataset = MOSIDataset.trainset
            dataset.wordInput = train['text']
            dataset.wordid = train['textid']
            dataset.wordLength = train['length']
            dataset.facetInput = train['facet']
            dataset.facetLength = train['facet_len']
            dataset.labelOutput = train['label']
            dataset.covarepInput = train['covarep']
            dataset.covarepLength = train['covarep_len']
            dataset = MOSIDataset.testset
            dataset.wordInput = test['text']
            dataset.wordid = test['textid']
            dataset.wordLength = test['length']
            dataset.facetInput = test['facet']
            dataset.facetLength = test['facet_len']
            dataset.labelOutput = test['label']
            dataset.covarepInput = test['covarep']
            dataset.covarepLength = test['covarep_len']
            dataset = MOSIDataset.validset
            dataset.wordInput = valid['text']
            dataset.wordid = valid['textid']
            dataset.wordLength = valid['length']
            dataset.facetInput = valid['facet']
            dataset.facetLength = valid['facet_len']
            dataset.labelOutput = valid['label']
            dataset.covarepInput = valid['covarep']
            dataset.covarepLength = valid['covarep_len']
            gc.wordDim = len(MOSIDataset.trainset.wordInput[0][0])
            gc.facetDim = len(MOSIDataset.trainset.facetInput[0][0][0])
            gc.covarepDim = len(MOSIDataset.trainset.covarepInput[0][0][0])
        if self.cls == "train":
            self.dataset = MOSIDataset.trainset
        elif self.cls == "test":
            self.dataset = MOSIDataset.testset
        elif self.cls == "valid":
            self.dataset = MOSIDataset.validset
        self.covarepInput = self.dataset.covarepInput[:]
        self.covarepLength = self.dataset.covarepLength[:]
        self.wordLength = self.dataset.wordLength[:]
        self.wordInput = self.dataset.wordInput[:]
        self.wordid = self.dataset.wordid[:]
        self.facetInput = self.dataset.facetInput[:]
        self.facetLength = self.dataset.facetLength[:]
        self.labelOutput = self.dataset.labelOutput[:]

    def mergeDataset(self):
        MOSIDataset.trainset.covarepInput.extend(MOSIDataset.testset.covarepInput)
        MOSIDataset.trainset.covarepLength.extend(MOSIDataset.testset.covarepLength)
        MOSIDataset.trainset.wordInput.extend(MOSIDataset.testset.wordInput)
        MOSIDataset.trainset.facetInput.extend(MOSIDataset.testset.facetInput)
        MOSIDataset.trainset.facetLength.extend(MOSIDataset.testset.facetLength)
        MOSIDataset.trainset.labelOutput.extend(MOSIDataset.testset.labelOutput)

    def __getitem__(self, index):
        inputLen = self.wordLength[index]
        return torch.tensor(self.wordid[index] + [0] * (gc.padding_len - len(self.wordid[index]))),\
            torch.cat((torch.tensor([list(x) for x in self.wordInput[index]], dtype=torch.float), torch.zeros((gc.padding_len - len(self.wordInput[index]), gc.wordDim))), 0),\
            torch.cat((torch.tensor(self.covarepInput[index], dtype=torch.float), torch.zeros((gc.padding_len - len(self.covarepInput[index]), gc.shift_padding_len, gc.covarepDim))), 0),\
            torch.cat((torch.tensor(self.covarepLength[index], dtype=torch.long), torch.zeros(gc.padding_len - len(self.covarepLength[index]), dtype=torch.long)), 0),\
            torch.cat((torch.tensor(self.facetInput[index], dtype=torch.float), torch.zeros((gc.padding_len - len(self.facetInput[index]), gc.shift_padding_len, gc.facetDim))), 0),\
            torch.cat((torch.tensor(self.facetLength[index], dtype=torch.long), torch.zeros(gc.padding_len - len(self.facetLength[index]), dtype=torch.long)), 0),\
            inputLen, torch.tensor(self.labelOutput[index])

    def __len__(self):
        return len(self.labelOutput)

if __name__ == "__main__":
    dataset = MOSIDataset('/home/victorywys/Documents/data/', src="csd", save=False)
