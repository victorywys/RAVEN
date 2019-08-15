import os
import torch
import torch.utils.data as Data
import numpy as np
import cPickle as pickle


from mmsdk import mmdatasdk
from mmsdk.mmdatasdk import computational_sequence
from mmsdk.mmdatasdk import mmdataset
import mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI.cmu_mosi_std_folds as std_folds

from consts import global_consts as gc

def mid(a):
    return (a[0] + a[1]) / 2.0

class MOSISubdata():
    def __init__(self, name = "train"):
        self.name = name
        self.covarepInput = []
        self.covarepLength = []
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
            self.readFromCSD()
            self.alignment()

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
        self.facetInput = self.dataset.facetInput[:]
        self.facetLength = self.dataset.facetLength[:]
        self.labelOutput = self.dataset.labelOutput[:]

    def readFromCSD(self):
        labelCompSeq = computational_sequence(self.root+'CMU_MOSI_Opinion_Labels.csd').data
        facetCompSeq = computational_sequence(self.root+'CMU_MOSI_VisualFacet_4.1.csd').data
        wordCompSeq = computational_sequence(self.root+'CMU_MOSI_TimestampedWordVectors.csd').data
        covarepCompSeq = computational_sequence(self.root+'CMU_MOSI_COVAREP.csd').data

        self.vidList = []
        self.sidList = []
        self.labelList = []
        self.facetList = []
        self.facetInterval = []
        self.covarepList = []
        self.covarepInterval = []
        self.wordList = []
        self.wordInterval = []
        for i, vid in enumerate(labelCompSeq):
            if gc.debug:
                if i > 5:
                    break
            if i == 88 or i == 66:
                continue
            print "processing video %d, uid %s" % (i, vid)
            labels = labelCompSeq[vid]['features']
            sen_intervals = labelCompSeq[vid]['intervals']
            facet = facetCompSeq[vid]['features']
            facet_intervals = facetCompSeq[vid]['intervals']
            covarep = covarepCompSeq[vid]['features']
            covarep_intervals = covarepCompSeq[vid]['intervals']
            words = wordCompSeq[vid]['features']
            word_intervals = wordCompSeq[vid]['intervals']

            #add basic infomation
            sen_num = 0
            while sen_num < len(labels):
                self.labelList.append(labels[sen_num])
                self.vidList.append(vid)
                self.sidList.append(sen_num)
                sen_num += 1

            #add word vectors
            start, end = 0, 0
            sen_num = 0
            while sen_num < len(labels):
                while start < len(words) and mid(word_intervals[start]) < sen_intervals[sen_num][0]:
                    start += 1
                end = start
                while end < len(words) and mid(word_intervals[end]) < sen_intervals[sen_num][1]:
                    end += 1
                toAppend = []
                toAppendInterval = []
                for k in range(start, end):
                    toAppend.append(words[k])
                    toAppendInterval.append(word_intervals[k])
                self.wordList.append(toAppend)
                self.wordInterval.append(toAppendInterval)
                if len(toAppend) > 50:
                    print len(toAppend)
                start = end
                sen_num += 1

            #add facets
            start, end = 0, 0
            sen_num = 0
            while sen_num < len(labels):
                while start < len(facet) and mid(facet_intervals[start]) < sen_intervals[sen_num][0]:
                    start += 1
                end = start
                while end < len(facet) and mid(facet_intervals[end]) < sen_intervals[sen_num][1]:
                    end += 1
                self.facetList.append(facet[start:end])
                self.facetInterval.append(facet_intervals[start:end])
                start = end
                sen_num += 1

            #add covarep
            start, end = 0, 0
            sen_num = 0
            while sen_num < len(labels):
                while start < len(covarep) and mid(covarep_intervals[start]) < sen_intervals[sen_num][0]:
                    start += 1
                end = start
                while end < len(covarep) and mid(covarep_intervals[end]) < sen_intervals[sen_num][1]:
                    end += 1
                self.covarepList.append(covarep[start:end])
                self.covarepInterval.append(covarep_intervals[start:end])
                start = end
                sen_num += 1

    def alignment(self):
        trainvid = std_folds.standard_train_fold
        testvid = std_folds.standard_test_fold
        validvid = std_folds.standard_valid_fold
        gc.wordDim = len(self.wordList[0][0])
        gc.facetDim = len(self.facetList[0][0])
        gc.covarepDim = len(self.covarepList[0][0])
        timescale = 1.0 / gc.sub_freq
        for i in range(len(self.labelList)):
            if self.vidList[i] in trainvid:
                dataset = MOSIDataset.trainset
            elif self.vidList[i] in testvid:
                dataset = MOSIDataset.testset
            elif self.vidList[i] in validvid:
                dataset = MOSIDataset.validset

            if (len(self.wordList[i]) > gc.padding_len) or (len(self.wordList[i]) == 0):
                continue
            dataset.wordLength.append(len(self.wordList[i]))
            dataset.wordInput.append(self.wordList[i])
            dataset.labelOutput.append(self.labelList[i])

            toAppend = []
            lengthToAppend = []
            start, end = 0, 0
            for j in range(len(self.wordList[i])):
                facetV = []
                while start < len(self.facetInterval[i]) and mid(self.facetInterval[i][start]) < self.wordInterval[i][j][0]:
                    start += 1
                end = start

                if start < len(self.facetInterval[i]):
                    startTime = mid(self.facetInterval[i][start])
                tempV = np.zeros(gc.facetDim)
                num = 0
                while end < len(self.facetInterval[i]) and mid(self.facetInterval[i][end]) < self.wordInterval[i][j][1]:
                    if mid(self.facetInterval[i][end]) < startTime + timescale:
                        tempV = tempV + self.facetList[i][end]
                        num += 1
                    else:
                        if num > 0:
                            facetV.append(tempV / num)
                        while mid(self.facetInterval[i][end]) > startTime + timescale:
                            startTime = startTime + timescale
                        num = 1
                        tempV = self.facetList[i][end]
                    end += 1
                if len(facetV) > gc.shift_padding_len:
                    facetV = facetV[:gc.shift_padding_len]
                lengthToAppend.append(len(facetV))
                while len(facetV) < gc.shift_padding_len:
                    facetV.append(np.zeros(gc.facetDim))
                start = end
                toAppend.append(facetV[:])
            dataset.facetInput.append(toAppend[:])
            dataset.facetLength.append(lengthToAppend[:])

            toAppend = []
            lengthToAppend = []
            start, end = 0, 0
            for j in range(len(self.wordList[i])):
                covarepV = []
                while start < len(self.covarepInterval[i]) and mid(self.covarepInterval[i][start]) < self.wordInterval[i][j][0]:
                    start += 1
                end = start

                if start < len(self.covarepInterval[i]):
                    startTime = mid(self.covarepInterval[i][start])
                tempV = np.zeros(gc.covarepDim)
                num = 0
                while end < len(self.covarepInterval[i]) and mid(self.covarepInterval[i][end]) < self.wordInterval[i][j][1]:
                    if mid(self.covarepInterval[i][end]) < startTime + timescale:
                        tempV = tempV + self.covarepList[i][end]
                        num += 1
                    else:
                        if num > 0:
                            covarepV.append(tempV / num)
                        while mid(self.covarepInterval[i][end]) > startTime + timescale:
                            startTime = startTime + timescale
                        num = 1
                        tempV = self.covarepList[i][end]
                    end += 1
                if len(covarepV) > gc.shift_padding_len:
                    covarepV = covarepV[:gc.shift_padding_len]
                lengthToAppend.append(len(covarepV))
                while len(covarepV) < gc.shift_padding_len:
                    covarepV.append(np.zeros(gc.covarepDim))
                start = end
                toAppend.append(covarepV[:])
            dataset.covarepInput.append(toAppend[:])
            dataset.covarepLength.append(lengthToAppend[:])

    def __getitem__(self, index):
        inputLen = self.wordLength[index]
        return torch.cat((torch.tensor(self.wordInput[index], dtype=torch.float32), torch.zeros((gc.padding_len - len(self.wordInput[index]), gc.wordDim))), 0),\
                torch.cat((torch.tensor(self.covarepInput[index], dtype=torch.float32), torch.zeros((gc.padding_len - len(self.covarepInput[index]), gc.shift_padding_len, gc.covarepDim))), 0),\
                torch.cat((torch.tensor(self.covarepLength[index], dtype=torch.long), torch.zeros(gc.padding_len - len(self.covarepLength[index]), dtype=torch.long)), 0),\
                torch.cat((torch.tensor(self.facetInput[index], dtype=torch.float32), torch.zeros((gc.padding_len - len(self.facetInput[index]), gc.shift_padding_len, gc.facetDim))), 0),\
                torch.cat((torch.tensor(self.facetLength[index], dtype=torch.long), torch.zeros(gc.padding_len - len(self.facetLength[index]), dtype=torch.long)), 0),\
                inputLen, torch.tensor(self.labelOutput[index]).squeeze()

    def __len__(self):
        return len(self.labelOutput)

if __name__ == "__main__":
    dataset = MOSIDataset(gc.data_path, src="csd", save=False)
