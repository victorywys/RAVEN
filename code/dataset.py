import os
import torch
import torch.utils.data as Data
import numpy as np
import cPickle as pickle

from mmsdk import mmdatasdk
from mmsdk.mmdatasdk import computational_sequence
from mmsdk.mmdatasdk import mmdataset
from consts import global_consts as gc

def mid(a):
    return (a[0] + a[1]) / 2.0

class MOSISubdata():
    def __init__(self, name = "train"):
        self.name = name
        self.phoInput = []
        self.phoLength = []
        self.wordInput = []
        self.facetInput = []
        self.openfaceInput = []
        self.labelOutput = []
        self.smileInput = []


class MOSIDataset(Data.Dataset):
    trainset = MOSISubdata("train")
    testset = MOSISubdata("test")
    validset = MOSISubdata("valid")

    def __init__(self, root, cls="train", src="csd", save=False):
        self.root = root
        self.cls = cls
        if len(MOSIDataset.trainset.labelOutput) != 0:
            print "Data has been preiviously loaded, fetching from previous lists."
        else:
            self.readFromCSD()
            self.alignment()

            # the proportion of datasets is quite unreasonable, I'll merge the trainset and the testset to try to get better results.
            # this shouldn't happen in the final version
#            self.mergeDataset()

            if gc.useWord:
                gc.wordDim = len(MOSIDataset.trainset.wordInput[0][0])
            gc.facetDim = len(MOSIDataset.trainset.facetInput[0][0])
            gc.openfaceDim = len(MOSIDataset.trainset.openfaceInput[0][0])
            gc.smileDim = len(MOSIDataset.trainset.smileInput[0])
        if self.cls == "train":
            self.dataset = MOSIDataset.trainset
        elif self.cls == "test":
            self.dataset = MOSIDataset.testset
        elif self.cls == "valid":
            self.dataset = MOSIDataset.validset
        self.phoInput = self.dataset.phoInput[:]
        self.phoLength = self.dataset.phoLength[:]
        self.wordInput = self.dataset.wordInput[:]
        self.facetInput = self.dataset.facetInput[:]
        self.openfaceInput = self.dataset.openfaceInput[:]
        self.labelOutput = self.dataset.labelOutput[:]
        self.smileInput = self.dataset.smileInput[:]
        print len(self.labelOutput)
        print gc.padding_len

    def mergeDataset(self):
        MOSIDataset.trainset.phoInput.extend(MOSIDataset.testset.phoInput)
        MOSIDataset.trainset.phoLength.extend(MOSIDataset.testset.phoLength)
        MOSIDataset.trainset.wordInput.extend(MOSIDataset.testset.wordInput)
        MOSIDataset.trainset.facetInput.extend(MOSIDataset.testset.facetInput)
        MOSIDataset.trainset.openfaceInput.extend(MOSIDataset.testset.openfaceInput)
        MOSIDataset.trainset.labelOutput.extend(MOSIDataset.testset.labelOutput)
        MOSIDataset.trainset.smileInput.extend(MOSIDataset.testset.smileInput)

    def alignment(self):
        f = open(self.root + "train.pkl", 'r')
        trainvid = pickle.load(f)
        f.close()
        f = open(self.root + "test.pkl", 'r')
        testvid = pickle.load(f)
        f.close()
        f = open(self.root + "valid.pkl", 'r')
        validvid = pickle.load(f)
        f.close()

        phoLen = self.buildPhoVocab()
        gc.phoDim = phoLen
        facetLen = len(self.facetList[0][0])
        openfaceLen = len(self.openfaceList[0][0])
        maxPhoPerWord = 0
        for i in range(len(self.labelList)):
            if self.vidList[i] in trainvid:
                dataset = MOSIDataset.trainset
            elif self.vidList[i] in testvid:
                dataset = MOSIDataset.testset
            elif self.vidList[i] in validvid:
                dataset = MOSIDataset.validset
            if gc.useWord == False:
                # Word vectors are not used, so all vectors should be aligned to phomemes
                if (len(self.phoList[i]) > gc.padding_len) or (len(self.phoList[i]) == 0):
                    continue
                dataset.phoInput.append([self.phoDict[phoName] for phoName in self.phoList[i]])
                dataset.smileInput.append(self.smileList[i])
                dataset.labelOutput.append(self.labelList[i])
                toAppend = []
                start, end = 0, 0
                for j in range(len(self.phoList[i])):
                    facetV = np.zeros(facetLen)
                    while start < len(self.facetInterval[i]) and mid(self.facetInterval[i][start]) < self.phoInterval[i][j][0]:
                        start += 1
                    end = start
                    while end < len(self.facetInterval[i]) and mid(self.facetInterval[i][end]) < self.phoInterval[i][j][1]:
                        facetV = facetV + self.facetList[i][end]
                        end += 1
                    if end != start:
                        facetV = facetV / (end - start)
                    start = end
                    toAppend.append(facetV)
                dataset.facetInput.append(toAppend[:])

                start, end = 0, 0
                toAppend = []
                for j in range(len(self.phoList[i])):
                    openfaceV = np.zeros(openfaceLen)
                    while start < len(self.openfaceInterval[i]) and mid(self.openfaceInterval[i][start]) < self.phoInterval[i][j][0]:
                        start += 1
                    end = start
                    while end < len(self.openfaceInterval[i]) and mid(self.openfaceInterval[i][end]) < self.phoInterval[i][j][1]:
                        openfaceV = openfaceV + self.openfaceList[i][end]
                        end += 1
                    if end != start:
                        openfaceV = openfaceV / (end - start)
                    start = end
                    toAppend.append(openfaceV)
                dataset.openfaceInput.append(toAppend[:])
            else:
                # Otherwise, the vectors should be aligned to word vectors.
                if (len(self.wordList[i]) > gc.padding_len) or (len(self.wordList[i]) == 0):
                    continue
                dataset.wordInput.append(self.wordList[i])
                dataset.smileInput.append(self.smileList[i])
                dataset.labelOutput.append(self.labelList[i])
                toAppend = []
                lengthToAppend = []
                start, end = 0, 0
                for j in range(len(self.wordList[i])):
                    phoV = []
                    while start < len(self.phoInterval[i]) and mid(self.phoInterval[i][start]) < self.wordInterval[i][j][0]:
                        start += 1
                    end = start

                    while end < len(self.phoInterval[i]) and mid(self.phoInterval[i][end]) < self.wordInterval[i][j][1]:
                        phoV.append(self.phoDict[self.phoList[i][end]])
                        end += 1
                    if end - start > maxPhoPerWord:
                        maxPhoPerWord = end - start
                    if len(phoV) > gc.pho_padding_len:
                        phoV = phoV[:gc.pho_padding_len]
                    lengthToAppend.append(len(phoV))
                    while len(phoV) < gc.pho_padding_len:
                        phoV.append(torch.zeros(phoLen))
                    start = end
                    toAppend.append(phoV[:])
                dataset.phoInput.append(toAppend[:])
                dataset.phoLength.append(lengthToAppend[:])

                toAppend = []
                start, end = 0, 0
                for j in range(len(self.wordList[i])):
                    facetV = np.zeros(facetLen)
                    while start < len(self.facetInterval[i]) and mid(self.facetInterval[i][start]) < self.wordInterval[i][j][0]:
                        start += 1
                    end = start
                    while end < len(self.facetInterval[i]) and mid(self.facetInterval[i][end]) < self.wordInterval[i][j][1]:
                        facetV = facetV + self.facetList[i][end]
                        end += 1
                    if end != start:
                        facetV = facetV / (end - start)
                    start = end
                    toAppend.append(facetV)
                dataset.facetInput.append(toAppend[:])

                toAppend = []
                start, end = 0, 0
                for j in range(len(self.wordList[i])):
                    openfaceV = np.zeros(openfaceLen)
                    while start < len(self.openfaceInterval[i]) and mid(self.openfaceInterval[i][start]) < self.wordInterval[i][j][0]:
                        start += 1
                    end = start
                    while end < len(self.openfaceInterval[i]) and mid(self.openfaceInterval[i][end]) < self.wordInterval[i][j][1]:
                        openfaceV = openfaceV + self.openfaceList[i][end]
                        end += 1
                    if end != start:
                        openfaceV = openfaceV / (end - start)
                    start = end
                    toAppend.append(openfaceV)
                dataset.openfaceInput.append(toAppend[:])

        print "max phonemes per word:", maxPhoPerWord

    def buildPhoVocab(self):
        self.phoDict = {}
        num = 0
        for sen in self.phoList:
            for pho in sen:
                if not self.phoDict.has_key(pho):
                    self.phoDict[pho] = num
                    num += 1
        print "total phomemes: %d" % num
        for pho in self.phoDict:
            self.phoDict[pho] = torch.zeros(1, num).scatter_(1, torch.tensor([[self.phoDict[pho]]]), 1.0).squeeze()
        return num

    def readFromCSD(self):
        phoCompSeq = computational_sequence(self.root+'CMU_MOSI_TimestampedPhones.csd').data
        labelCompSeq = computational_sequence(self.root+'CMU_MOSI_Opinion_Labels.csd').data
        facetCompSeq = computational_sequence(self.root+'CMU_MOSI_VisualFacet_4.1.csd').data
        openfaceCompSeq = computational_sequence(self.root+'CMU_MOSI_OpenFace_V1.csd').data
        smileCompSeq = computational_sequence(self.root+'CMU_MOSI_OpenSmile.csd').data
        wordCompSeq = computational_sequence(self.root+'CMU_MOSI_TimestampedWordVectors.csd').data
        rawCompSeq = computational_sequence(self.root+'CMU_MOSI_TimestampedWords.csd').data
        self.vidList = []
        self.sidList = []
        self.labelList = []
        self.phoList = []
        self.phoInterval = []
        self.facetList = []
        self.facetInterval = []
        self.openfaceList = []
        self.openfaceInterval = []
        self.smileList = []
        self.wordList = []
        self.wordInterval = []
        self.rawList = []
        fout = open("raw_data.txt", 'w')
        fout2 = open("raw_data_no_sp.txt", 'w')
        for i, vid in enumerate(labelCompSeq):
            if gc.debug:
                if i > 2:
                    break
            if vid == "c5xsKMxpXnc":
                continue
            print "processing video %d, uid %s" % (i, vid)
            labels = labelCompSeq[vid]['features']
            sen_intervals = labelCompSeq[vid]['intervals']
            phos = phoCompSeq[vid]['features']
            pho_intervals = phoCompSeq[vid]['intervals']
            facet = facetCompSeq[vid]['features']
            facet_intervals = facetCompSeq[vid]['intervals']
            openface = openfaceCompSeq[vid]['features']
            openface_intervals = openfaceCompSeq[vid]['intervals']
            smile = smileCompSeq[vid]['features']
            words = wordCompSeq[vid]['features']
            word_intervals = wordCompSeq[vid]['intervals']
            raws = rawCompSeq[vid]['features']

            timescale = sen_intervals[-1][1]

            #add phomemes and basic infomation
            start, end = 0, 0
            sen_num = 0
            while sen_num < len(labels):
                while start < len(phos) and mid(pho_intervals[start]) < sen_intervals[sen_num][0]:
                    start += 1
                end = start
                while end < len(phos) and mid(pho_intervals[end]) < sen_intervals[sen_num][1]:
                    end += 1
                self.phoList.append([e[0] for e in phos[start:end]])
                self.phoInterval.append(pho_intervals[start:end])
                self.labelList.append(labels[sen_num])
                #add smiles here, for the intervals of opensmile is the same as the labels
                self.smileList.append(smile[sen_num])
                self.vidList.append(vid)
                self.sidList.append(sen_num)
                start = end
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
                    if raws[k] != 'sp' or not gc.no_sp:
                        toAppend.append(words[k])
                        toAppendInterval.append(word_intervals[k])
                self.wordList.append(toAppend)
                self.wordInterval.append(toAppendInterval)
                if len(toAppend) > 50:
                    print len(toAppend)

                fout.write("%s: %s %f\n" % (vid, ''.join([x[0] + ' ' for x in raws[start:end]]), labels[sen_num]))
                fout2.write("%s: %s %f\n" % (vid, ''.join([x[0] + ' ' if x[0] != "sp" else '' for x in raws[start:end]]), labels[sen_num]))
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

            #add openface
            start, end = 0, 0
            sen_num = 0
            while sen_num < len(labels):
                while start < len(openface) and mid(openface_intervals[start]) < sen_intervals[sen_num][0]:
                    start += 1
                end = start
                while end < len(openface) and mid(openface_intervals[end]) < sen_intervals[sen_num][1]:
                    end += 1
                self.openfaceList.append(openface[start:end])
                self.openfaceInterval.append(openface_intervals[start:end])
                start = end
                sen_num += 1

    def __getitem__(self, index):
        inputLen = len(self.wordInput[index]) if gc.useWord else len(self.phoInput[index])
        return torch.cat((torch.tensor(self.wordInput[index], dtype=torch.float), torch.zeros((gc.padding_len - len(self.wordInput[index]), gc.wordDim))), 0) if gc.useWord else torch.zeros(1),\
            torch.cat((torch.tensor([d.tolist() for d in self.phoInput[index]], dtype=torch.float), torch.zeros((gc.padding_len - len(self.phoInput[index]), gc.phoDim))), 0) if not gc.useWord else torch.cat((torch.tensor([[pho.tolist() for pho in phos] for phos in self.phoInput[index]], dtype=torch.float), torch.zeros((gc.padding_len - len(self.phoInput[index]), gc.pho_padding_len, gc.phoDim))), 0),\
            torch.cat((torch.tensor(self.phoLength[index]), torch.zeros(gc.padding_len - len(self.phoLength[index]), dtype=torch.long)), 0) if gc.useWord else torch.zeros(1),\
            torch.cat((torch.tensor(self.facetInput[index], dtype=torch.float), torch.zeros((gc.padding_len - len(self.facetInput[index]), gc.facetDim))), 0),\
            torch.cat((torch.tensor(self.openfaceInput[index], dtype=torch.float), torch.zeros((gc.padding_len - len(self.openfaceInput[index]), gc.openfaceDim))), 0),\
            inputLen, self.smileInput[index], self.labelOutput[index]

    def __len__(self):
        return len(self.labelOutput)

if __name__ == "__main__":
    dataset = MOSIDataset('/home/victorywys/Documents/data/', src="csd", save=False)
