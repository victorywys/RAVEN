from consts import global_consts as gc

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import layer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        features = 0
        input_size = 0

        self.normcovarep = nn.BatchNorm2d(gc.padding_len, track_running_stats=False)
        self.dropcovarep = nn.Dropout(p=gc.dropProb)
        self.fc_rszcovarep = nn.Linear(gc.covarepDim, gc.normDim)
        self.covarepLSTM = nn.LSTM(gc.normDim, gc.cellDim, batch_first = True)
        self.covarepW = nn.Linear(gc.cellDim + gc.wordDim, 1)

        self.normFacet = nn.BatchNorm2d(gc.padding_len, track_running_stats=False)
        self.dropFacet = nn.Dropout(p=gc.dropProb)
        self.fc_rszFacet = nn.Linear(gc.facetDim, gc.normDim)
        self.facetLSTM = nn.LSTM(gc.normDim, gc.cellDim, batch_first = True)
        self.facetW = nn.Linear(gc.cellDim + gc.wordDim, 1)

        self.calcAddon = nn.Linear(2 * gc.cellDim, gc.wordDim)

        self.dropWord = nn.Dropout(p=gc.dropProb)
        input_size += gc.wordDim

        self.lstm1 = layer.LSTM(input_size, gc.hiddenDim, layer=gc.layer)

        if gc.lastState:
            self.fc_afterLSTM = nn.Linear(gc.hiddenDim, 1)
        else:
            self.fc_afterLSTM = nn.Linear(gc.hiddenDim * gc.padding_len, 1)

    def forward(self, words, covarep, covarepLens, facet, facetLens, inputLens):
        batch = covarep.size()[0]
        inputs = None
        covarep = self.normcovarep(covarep)
        covarepInput = self.fc_rszcovarep(self.dropcovarep(covarep))
        covarepFlat = covarepInput.data.contiguous().view(-1, gc.shift_padding_len, gc.normDim)
        output, _ = self.covarepLSTM(covarepFlat)
        output = torch.cat([torch.zeros(batch * gc.padding_len, 1, gc.cellDim).to(gc.device), output], 1)
        covarepLensFlat = covarepLens.data.contiguous().view(-1)
        covarepSelector = torch.zeros(batch * gc.padding_len, 1, gc.shift_padding_len + 1).to(gc.device).scatter_(2, covarepLensFlat.unsqueeze(1).unsqueeze(1), 1.0)
        covarepState = torch.matmul(covarepSelector, output).squeeze()

        #facet = self.normFacet(facet)
        facetInput = self.fc_rszFacet(self.dropFacet(facet))
        facetFlat = facetInput.data.contiguous().view(-1, gc.shift_padding_len, gc.normDim)
        output, _ = self.facetLSTM(facetFlat)
        output = torch.cat([torch.zeros(batch * gc.padding_len, 1, gc.cellDim).to(gc.device), output], 1)
        facetLensFlat = facetLens.data.contiguous().view(-1)
        facetSelector = torch.zeros(batch * gc.padding_len, 1, gc.shift_padding_len + 1).to(gc.device).scatter_(2, facetLensFlat.unsqueeze(1).unsqueeze(1), 1.0)
        facetState = torch.matmul(facetSelector, output).squeeze()

        wordFlat = words.data.contiguous().view(-1, gc.wordDim)
        covarepWeight = self.covarepW(torch.cat([covarepState, wordFlat], 1))
        facetWeight = self.facetW(torch.cat([facetState, wordFlat], 1))
        covarepState = covarepState * covarepWeight
        facetState = facetState * facetWeight
        addon = self.calcAddon(torch.cat([covarepState, facetState], 1))

        addonL2 = torch.norm(addon, 2, 1)
        addonL2 = torch.max(addonL2, torch.tensor([1.0]).to(gc.device)) / torch.tensor([gc.shift_weight]).to(gc.device)
        addon = addon / addonL2.unsqueeze(1)
        addon = addon.data.contiguous().view(batch, gc.padding_len, gc.wordDim)

        wordsL2 = torch.norm(words, 2, 2).unsqueeze(2)
        wordInput = self.dropWord(words + addon * wordsL2)

        inputs = wordInput

        output, _ = self.lstm1(inputs)
        if gc.lastState:
            self.selector = torch.zeros(batch, 1, gc.padding_len).to(gc.device).scatter_(2, (inputLens-1).unsqueeze(1).unsqueeze(1), 1.0)
            spec_output = torch.matmul(self.selector, output).squeeze()
        else:
            spec_output = output.data.contiguous().view(-1, gc.hiddenDim * gc.padding_len)
        final = self.fc_afterLSTM(spec_output)
        return final.squeeze()
