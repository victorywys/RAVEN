from consts import global_consts as gc

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    # Define a Deep LSTM Network
    #
    # Parameters:
    #   input_dim: the dimension of the input
    #   hidden_dim: the dimension of the hidden states
    #   layer = 1: the number of layers
    #   TODO:device = None
    #
    # Inputs: inputs, (h_0, c_0) = (0, 0)
    #   inputs (shape: batch * seq_len * input_dim): the inputs vector, the same as torch.nn.LSTM(batch_first = True)
    #   h_0 (shape: batch * layer * hidden_dim): tensor containing the initial hidden state for each element in the batch.
    #   c_0 (shape: batch * layer * hidden_dim): tensor containing the initial cell state for each element in the batch.
    #
    # Outputs: outputs, cv_outputs, (h_n, c_n, cv_n)
    #   outputs (shape: batch * seq_len * hidden_dim): tensor containing the output features (h_t) from the last layer of the LSTM, for each t.
    #   h_n (shape: batch * layer * hidden_dim): the last hidden state of every layer for each element in the batch
    #   c_n (shape: batch * layer * hidden_dim): the last cell state of every layer for each element in the batch

    def __init__(self, input_dim, hidden_dim, layer=1, device=None):
        super(LSTM, self).__init__()
        self.i_dim = input_dim
        self.h_dim = hidden_dim
        self.layer = layer
        if layer <= 0:
            print "Error: The number of layer should be a positive integer, but got [%s] instead." % str(layer)

        self.add_module("layer_1", LSTMLayer(self.i_dim, self.h_dim))
        for i in range(1, self.layer):
            self.add_module("layer_%d" % (i + 1), LSTMLayer(self.h_dim, self.h_dim))

    def forward(self, inputs, hc_0=None):
        batch = inputs.size()[0]
        if hc_0 == None:
            h, c = torch.zeros(batch, self.layer, self.h_dim).to(gc.device), torch.zeros(batch, self.layer, self.h_dim).to(gc.device)
        else:
            h = hc_0[0].clone()
            c = hc_0[1].clone()

        for i in range(self.layer):
            inputs, hc_n = self.__getattr__("layer_%d" % (i + 1))(inputs, map(lambda x:x[:, i, :].squeeze(), [h, c]))
            if i == 0:
                h_out, c_out = map(lambda x:x.unsqueeze(1), hc_n)
            else:
                h_out, c_out = map(lambda x, y:torch.cat([y, x.unsqueeze(1)], 1), list(hc_n), [h_out, c_out])
        return inputs, (h_out, c_out)



class LSTMLayer(nn.Module):
    # Define a normal LSTM Layer
    # This should work the same as torch.nn.LSTM(), but seems it can train faster.
    #
    # Parameters:
    #   input_dim: the dimension of the input
    #   hidden_dim: the dimension of the hidden states
    #
    # Inputs: inputs, (h_0, c_0) = (0, 0)
    #   inputs (shape: batch * seq_len * input_dim): the inputs vector, the same as torch.LSTM(batch_first = True)
    #   h_0 (shape: batch * hidden_dim): tensor containing the initial hidden state for each element in the batch. Only one_layer with unidirectional LSTM is supported now
    #   c_0 (shape: batch * hidden_dim): tensor containing the initial cell state for each element in the batch.
    #
    # Outputs: outputs, (h_n, c_n)
    #   outputs (shape: batch * seq_len * hidden_dim): tensor containing the output features (h_t) from the last layer of the LSTM, for each t.
    #   h_n (shape: batch * hidden_dim): the last hidden state for each element in the batch
    #   c_n (shape: batch * hidden_dim): the last cell state for eache element in the batch

    def __init__(self, input_dim, hidden_dim):
        super(LSTMLayer, self).__init__()
        self.i_dim = input_dim
        self.h_dim = hidden_dim
        self.lstm = nn.LSTMCell(self.i_dim, self.h_dim)

    def forward(self, inputs, hc_0 = None):
        #inputs: batch * len * input_dim
        batch, seq_len, _ = inputs.size()
        if hc_0 == None:
            h, c = torch.zeros(batch, self.h_dim).to(gc.device), torch.zeros(batch, self.h_dim).to(gc.device)
        else:
            h = hc_0[0].clone().to(gc.device)
            c = hc_0[1].clone().to(gc.device)

        for i in range(seq_len):
            if i == 0:
                h, c = self.lstm(inputs[:, i, :], (h, c))
                outputs = torch.unsqueeze(h, 1)
            else:
                # there are two attention coffecients calculation methods:
                #   1. alpha_i = softmax(MLP(h'_i, h(')_{t-1}))
                #   2. alpha_i = softmax(h'_i, h'_{t-1})
                # now we're trying to use the second method, so self.calc_alpha isn't used.
                #
                # torch.tensor(h_primes).size(): batch * (t - 1) * hidden_dim
                h, c = self.lstm(inputs[:, i, :], (h, c))
                outputs = torch.cat([outputs, torch.unsqueeze(h, 1)], 1)
        return outputs, (h, c)
