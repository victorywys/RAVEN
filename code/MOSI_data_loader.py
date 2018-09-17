import os
from collections import defaultdict
import numpy as np
import random
import scipy.io as sio
import cPickle
import h5py
from consts import global_consts as gc

data_path = gc.data_path
dataset_path = data_path
truth_path = dataset_path + 'Meta_data/boundaries_sentimentint_avg.csv'
#openface_path = dataset_path + "Features/Visual/OPEN_FACE/Segmented/"
openface_path = dataset_path + "Features/Visual/OpenfaceRaw/"
facet_path = dataset_path + "Features/Visual/FACET_GIOTA/"
covarep_path = dataset_path + "Features/Audio/raw/"
transcript_path = dataset_path + 'Transcript/SEGMENT_ALIGNED/'
word2ix_path = data_path + 'glove_word_embedding/word2ix_300_mosi.pkl'
word_embedding_path = data_path + "glove_word_embedding/glove_300_mosi.pkl"
def load_word_embedding():
    with open(word_embedding_path) as f:
        return cPickle.load(f)

def load_word2ix():
    with open(word2ix_path) as f:
        word2ix = cPickle.load(f)
    return word2ix

def load_truth():
    truth_dict = defaultdict(dict)
    with open(truth_path) as f:
        lines = f.read().split("\r\n")
    for line in lines:
        if line != '':
            line = line.split(",")
            truth_dict[line[2]][line[3]] = {'start_time': float(line[0]), 'end_time':float(line[1]), 'sentiment':float(line[4])}
    return truth_dict

def load_facet(truth_dict):
    for video_index in truth_dict:
        file_name = facet_path + video_index + '.FACET_out.csv'
        #print file_name
        with open(file_name) as f:
            lines = f.read().split('\r\n')[1:]
            lines = [[float(x) for x in line.split(',')]  for line in lines if line != '']
            for seg_index in truth_dict[video_index]:
                for w in truth_dict[video_index][seg_index]['data']:
                    start_frame = int(w['start_time_clip']*30)
                    end_frame = min(int(w['end_time_clip']*30), len(lines) - 1)
                    frame_interval = int(30. / gc.sub_freq)
                    ft = [list(np.mean(lines[f:min(f+frame_interval, end_frame)], 0))[5:] for f in range(start_frame, end_frame, frame_interval)]
                    num = len(ft)
                    if num > gc.shift_padding_len:
                        ft = ft[:gc.shift_padding_len]
                        num = gc.shift_padding_len
                    for i, vec in enumerate(ft):
                        temp = np.array(vec)
                        temp[np.isnan(vec)] = 0
                        temp[np.isneginf(vec)] = 0
                        ft[i] = list(temp)
                    while len(ft) < gc.shift_padding_len:
                        ft.append([0 for _ in np.zeros(len(lines[0]) - 5)])
                    w['facet'] = ft
                    w['facet_len'] = num

def load_covarep(truth_dict):
    for video_index in truth_dict:
        file_name = covarep_path + video_index + '.mat'
        fts = sio.loadmat(file_name)['features']
        #print fts.shape
        dim = fts.shape[-1]
        for seg_index in truth_dict[video_index]:
            for w in truth_dict[video_index][seg_index]['data']:
                start_frame = int(w['start_time_clip']*100)
                end_frame = min(int(w['end_time_clip']*100), len(fts) - 1)
                frame_interval = int(100. / gc.sub_freq)
                ft = [list(np.mean(fts[f:min(f+frame_interval, end_frame)], 0)) for f in range(start_frame, end_frame, frame_interval)]
                num = len(ft)
                if num > gc.shift_padding_len:
                    num = gc.shift_padding_len
                    ft = ft[:gc.shift_padding_len]
                for i, vec in enumerate(ft):
                    temp = np.array(vec)
                    temp[np.isnan(vec)] = 0
                    temp[np.isneginf(vec)] = 0
                    ft[i] = list(temp)
                while len(ft) < gc.shift_padding_len:
                    ft.append([0 for _ in range(dim)])
                w['covarep'] = ft
                w['covarep_len'] = num

def load_transcript(truth_dict, word2ix, word_embed):
    for video_index in truth_dict:
        for seg_index in truth_dict[video_index]:
            file_name = transcript_path + video_index + '_' + seg_index
            truth_dict[video_index][seg_index]['data'] = []
            with open(file_name) as f:
                lines = f.read().split("\n")
                for line in lines:
                    if line == '':
                        continue
                    line = line.split(',')
                    truth_dict[video_index][seg_index]['data'].append({'word_ix': word2ix[line[1]], 'word': line[1], 'start_time_seg': float(line[2]), 'end_time_seg':float(line[3]), 'start_time_clip':float(line[4]), 'end_time_clip':float(line[5]), 'word_embed': word_embed[word2ix[line[1]]]})
def split_data(tr_proportion, truth_dict):
    data = [(vid, truth_dict[vid]) for vid in truth_dict]
    data.sort(key = lambda x: x[0])
    tr_split = int(round(len(data) * tr_proportion))

    train = data[:52]
    valid = data[52:62]
    test = data[62:]
    print len(train)
    print len(valid) #0.1514 62 -> 52, 10, 31
    print len(test)
    #assert False
    return train, valid, test
def get_data(dataset, max_segment_len):
    data = {'facet': [], 'facet_len': [], 'covarep': [], 'covarep_len': [], 'text': [], 'textid': [], 'label': [], 'id': [], 'length': []}
    for i in range(len(dataset)):
        v = dataset[i][1]
        for seg_id in v:
            fts = v[seg_id]['data']
            facet, facet_len, text, textid, covarep, covarep_len = [], [], [], [], [], []
            length = 0
            if max_segment_len >= len(fts):
                for w in fts:
                    textid.append(w['word_ix'])
                    text.append(w['word_embed'])
                    covarep_len.append(w['covarep_len'])
                    covarep.append(w['covarep'])
                    facet_len.append(w['facet_len'])
                    facet.append(w['facet'])
                length = len(fts)
            else:
                for w in fts[len(fts)-max_segment_len:]:
                    text.append(w['word_embed'])
                    textid.append(w['word_ix'])
                    covarep.append(w['covarep'])
                    covarep_len.append(w['covarep_len'])
                    facet.append(w['facet'])
                    facet_len.append(w['facet_len'])
                length = max_segment_len
            data['facet'].append(facet)
            data['facet_len'].append(facet_len)
            data['covarep'].append(covarep)
            data['covarep_len'].append(covarep_len)
            data['text'].append(text)
            data['textid'].append(textid)
            data['label'].append(v[seg_id]['sentiment'])
            data['id'].append(dataset[i][0]+'_'+seg_id)
            data['length'].append(length)
    data['facet'] = np.array(data['facet'])
    data['facet_len'] = np.array(data['facet_len'])
    data['covarep'] = np.array(data['covarep'])
    data['covarep_len'] = np.array(data['covarep_len'])
    data['text'] = np.array(data['text'])
    data['textid'] = np.array(data['textid'])
    data['label'] = np.array(data['label'])
    return data
def load_word_level_features(max_segment_len, tr_proportion):
    word2ix = load_word2ix()
    word_embed = load_word_embedding()
    truth_dict = load_truth()
    load_transcript(truth_dict, word2ix, word_embed)
    load_facet(truth_dict)
    load_covarep(truth_dict)
    train, valid, test = split_data(tr_proportion, truth_dict)
    train = get_data(train, max_segment_len)
    valid = get_data(valid, max_segment_len)
    test = get_data(test, max_segment_len)
    return train, valid, test
