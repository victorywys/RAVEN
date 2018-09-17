from consts import global_consts as gc
import numpy as np
seed = 123
np.random.seed(seed)
import random
import pickle
import time, argparse
import csv
import sys, os
sys.path.append('/media/bighdd7/yansen/code/tools/iemocap/')
sys.path.append('/media/bighdd7/yansen/data/CMU-MultimodalDataSDK/lib/')

from collections import defaultdict, OrderedDict

data_path = gc.data_path
raw_path = gc.raw_path

def get_data():
        print "fetching labels..."
	labels = pickle.load(open("%s/labels_old.p" % raw_path,"rb"))
	e_labels = pickle.load(open("%s/e_labels_old.p" % raw_path,"rb"))
	happy_labels = pickle.load(open("%s/happy_labels.p" % raw_path,"rb"))
	angry_labels = pickle.load(open("%s/angry_labels.p" % raw_path,"rb"))
	sad_labels = pickle.load(open("%s/sad_labels.p" % raw_path,"rb"))
	neutral_labels = pickle.load(open("%s/neutral_labels.p" % raw_path,"rb"))
	fru_labels = pickle.load(open("%s/fru_labels.p" % raw_path,"rb"))
	exc_labels = pickle.load(open("%s/exc_labels.p" % raw_path,"rb"))

        print "loading texts..."
	text_dict = pickle.load(open("%s/text_dict_new.p" % data_path,"rb"))
        print "loading audios..."
	audio_dict = pickle.load(open("%s/audio_dict_new.p" % data_path,"rb"))
        print "loading videos..."
	video_dict = pickle.load(open("%s/video_dict_new.p" % data_path,"rb"))

        train_vids = [video_id for video_id in text_dict if 'Ses03' in video_id or 'Ses04' in video_id or 'Ses05' in video_id]
	valid_vids = [video_id for video_id in text_dict if 'Ses02' in video_id]
	test_vids = [video_id for video_id in text_dict if 'Ses01' in video_id]

	all_labels = happy_labels
	train_i = []
	for video_id in train_vids:
		for segment_id in text_dict[video_id]:
			if video_id in all_labels:
				if segment_id in all_labels[video_id]:
                                    if len(text_dict[video_id][segment_id]) > 0:
					train_i.append((video_id,segment_id))
	valid_i = []
	for video_id in valid_vids:
		for segment_id in text_dict[video_id]:
			if video_id in all_labels:
				if segment_id in all_labels[video_id]:
                                    if len(text_dict[video_id][segment_id]) > 0:
					valid_i.append((video_id,segment_id))
	test_i = []
	for video_id in test_vids:
		for segment_id in text_dict[video_id]:
			if video_id in all_labels:
				if segment_id in all_labels[video_id]:
                                    if len(text_dict[video_id][segment_id]) > 0:
					test_i.append((video_id,segment_id))



	def pad(data,max_segment_len,t):
		curr = []
                if np.array(data).shape[0] == 0:
			if t == 1:
				return np.zeros((max_segment_len,300))
			if t == 2:
				return np.zeros((max_segment_len, gc.shift_padding_len, 74))
			if t == 3:
				return np.zeros((max_segment_len, gc.shift_padding_len, 36))

                try:
                    if t == 1:
                        dim = 300
                    elif t == 2:
                        dim = 74
                    elif t == 3:
                        dim = 36
		except:
			if t == 1:
				return np.zeros((max_segment_len,300))
			if t == 2:
				return np.zeros((max_segment_len, gc.shift_padding_len, 74))
			if t == 3:
				return np.zeros((max_segment_len, gc.shift_padding_len, 36))
		if max_segment_len >= len(data):
		    if t == 1:
			for vec in data:
                                curr.append(vec)
                        for i in xrange(max_segment_len-len(data)):
                            curr.append([0 for i in xrange(dim)])
                    else:
                        data = data.tolist()
                        for i in xrange(len(data)):
                            if np.array(data[i]).shape[0] == 0:
                                data[i] = np.zeros([gc.shift_padding_len, dim])
                            else:
                                data[i] = np.concatenate((data[i], np.zeros([gc.shift_padding_len-len(data[i]), dim])), axis=0)
			for veclist in data:
                                curr.append(veclist)
                        for i in xrange(max_segment_len-len(data)):
			        curr.append(np.zeros_like(data[0]))

                else:	# max_segment_len < len(text), take last max_segment_len of text
                    if t != 1:
                        data = data.tolist()
                        for i in xrange(len(data)):
                            if np.array(data[i]).shape[0] == 0:
                                data[i] = np.zeros([gc.shift_padding_len, dim])
                            else:
                                data[i] = np.concatenate((data[i], np.zeros([gc.shift_padding_len-len(data[i]), dim])), axis=0)

                    for vec in data[:max_segment_len]:
		    	curr.append(vec)

                curr = np.array(curr)
		return curr

        max_segment_len = gc.padding_len

        print "padding train set ..."
        covarep_len_train = [] # num * padding_len
        for video_id, segment_id in train_i:
            curr = []
            data = audio_dict[video_id][segment_id]
            for i in xrange(len(data)):
                curr.append(len(data[i]))
            if len(curr) >= gc.padding_len:
                curr = curr[:gc.padding_len]
            else:
                num = gc.padding_len - len(curr)
                curr += [0 for i in range(num)]
            covarep_len_train.append(curr)
        covarep_len_train = np.array(covarep_len_train)

        facet_len_train = [] # num * padding_len
        for video_id, segment_id in train_i:
            curr = []
            data = audio_dict[video_id][segment_id]
            for i in xrange(len(data)):
                curr.append(len(data[i]))
            if len(curr) >= gc.padding_len:
                curr = curr[:gc.padding_len]
            else:
                num = gc.padding_len - len(curr)
                curr += [0 for i in range(num)]
            facet_len_train.append(curr)
        facet_len_train = np.array(facet_len_train)

        text_len_train = np.array([min(len(text_dict[video_id][segment_id]), gc.padding_len) for (video_id,segment_id) in train_i])
	text_train_emb = [pad(text_dict[video_id][segment_id],max_segment_len,1) for (video_id,segment_id) in train_i]
        text_train_emb = np.array(text_train_emb)
        covarep_train = np.array([pad(audio_dict[video_id][segment_id],max_segment_len,2) for (video_id,segment_id) in train_i])
	facet_train = np.array([pad(video_dict[video_id][segment_id],max_segment_len,3) for (video_id,segment_id) in train_i])
	y_train = np.nan_to_num(np.array([labels[video_id][segment_id] for (video_id,segment_id) in train_i]))
	ey_train = np.nan_to_num(np.array([e_labels[video_id][segment_id] for (video_id,segment_id) in train_i]))

        print "padding valid set ..."
        covarep_len_valid = [] # num * padding_len
        for video_id, segment_id in valid_i:
            curr = []
            data = audio_dict[video_id][segment_id]
            for i in xrange(len(data)):
                curr.append(len(data[i]))
            if len(curr) >= gc.padding_len:
                curr = curr[:gc.padding_len]
            else:
                num = gc.padding_len - len(curr)
                curr += [0 for i in range(num)]
            covarep_len_valid.append(curr)
        covarep_len_valid = np.array(covarep_len_valid)

        facet_len_valid = [] # num * padding_len
        for video_id, segment_id in valid_i:
            curr = []
            data = audio_dict[video_id][segment_id]
            for i in xrange(len(data)):
                curr.append(len(data[i]))
            if len(curr) >= gc.padding_len:
                curr = curr[:gc.padding_len]
            else:
                num = gc.padding_len - len(curr)
                curr += [0 for i in range(num)]
            facet_len_valid.append(curr)
        facet_len_valid = np.array(facet_len_valid)

        text_len_valid = np.array([min(len(text_dict[video_id][segment_id]), gc.padding_len) for (video_id,segment_id) in valid_i])
        text_valid_emb = np.array([pad(text_dict[video_id][segment_id],max_segment_len,1) for (video_id,segment_id) in valid_i])
	covarep_valid = np.array([pad(audio_dict[video_id][segment_id],max_segment_len,2) for (video_id,segment_id) in valid_i])
	facet_valid = np.array([pad(video_dict[video_id][segment_id],max_segment_len,3) for (video_id,segment_id) in valid_i])
	y_valid = np.nan_to_num(np.array([labels[video_id][segment_id] for (video_id,segment_id) in valid_i]))
	ey_valid = np.nan_to_num(np.array([e_labels[video_id][segment_id] for (video_id,segment_id) in valid_i]))

        print "padding test set ..."
        covarep_len_test = [] # num * padding_len
        for video_id, segment_id in test_i:
            curr = []
            data = audio_dict[video_id][segment_id]
            for i in xrange(len(data)):
                curr.append(len(data[i]))
            if len(curr) >= gc.padding_len:
                curr = curr[:gc.padding_len]
            else:
                num = gc.padding_len - len(curr)
                curr += [0 for i in range(num)]
            covarep_len_test.append(curr)
        covarep_len_test = np.array(covarep_len_test)

        facet_len_test = [] # num * padding_len
        for video_id, segment_id in test_i:
            curr = []
            data = audio_dict[video_id][segment_id]
            for i in xrange(len(data)):
                curr.append(len(data[i]))
            if len(curr) >= gc.padding_len:
                curr = curr[:gc.padding_len]
            else:
                num = gc.padding_len - len(curr)
                curr += [0 for i in range(num)]
            facet_len_test.append(curr)
        facet_len_test = np.array(facet_len_test)

        text_len_test = np.array([min(gc.padding_len, len(text_dict[video_id][segment_id])) for (video_id,segment_id) in test_i])

        text_test_emb = np.array([pad(text_dict[video_id][segment_id],max_segment_len,1) for (video_id,segment_id) in test_i])
	covarep_test = np.array([pad(audio_dict[video_id][segment_id],max_segment_len,2) for (video_id,segment_id) in test_i])
	facet_test = np.array([pad(video_dict[video_id][segment_id],max_segment_len,3) for (video_id,segment_id) in test_i])
	y_test = np.nan_to_num(np.array([labels[video_id][segment_id] for (video_id,segment_id) in test_i]))
	ey_test = np.nan_to_num(np.array([e_labels[video_id][segment_id] for (video_id,segment_id) in test_i]))

	happy_train = np.nan_to_num(np.array([happy_labels[video_id][segment_id] for (video_id,segment_id) in train_i]))
	happy_valid = np.nan_to_num(np.array([happy_labels[video_id][segment_id] for (video_id,segment_id) in valid_i]))
	happy_test = np.nan_to_num(np.array([happy_labels[video_id][segment_id] for (video_id,segment_id) in test_i]))

	angry_train = np.nan_to_num(np.array([angry_labels[video_id][segment_id] for (video_id,segment_id) in train_i]))
	angry_valid = np.nan_to_num(np.array([angry_labels[video_id][segment_id] for (video_id,segment_id) in valid_i]))
	angry_test = np.nan_to_num(np.array([angry_labels[video_id][segment_id] for (video_id,segment_id) in test_i]))

	sad_train = np.nan_to_num(np.array([sad_labels[video_id][segment_id] for (video_id,segment_id) in train_i]))
	sad_valid = np.nan_to_num(np.array([sad_labels[video_id][segment_id] for (video_id,segment_id) in valid_i]))
	sad_test = np.nan_to_num(np.array([sad_labels[video_id][segment_id] for (video_id,segment_id) in test_i]))

	neutral_train = np.nan_to_num(np.array([neutral_labels[video_id][segment_id] for (video_id,segment_id) in train_i]))
	neutral_valid = np.nan_to_num(np.array([neutral_labels[video_id][segment_id] for (video_id,segment_id) in valid_i]))
	neutral_test = np.nan_to_num(np.array([neutral_labels[video_id][segment_id] for (video_id,segment_id) in test_i]))

	print text_train_emb.shape	# n x seq x 300
        print text_len_train.shape      # n
	print covarep_train.shape       # n x seq x shift_len x 74
        print covarep_len_train.shape   # n x seq
        print facet_train.shape         # n x seq x shift_len x 35
        print facet_len_train.shape     # n x seq
        X_train = (text_train_emb, text_len_train, covarep_train, covarep_len_train, facet_train, facet_len_train)

	print text_valid_emb.shape	# n x seq x 300
        print text_len_valid.shape      # n
	print covarep_valid.shape       # n x seq x shift_len x 74
        print covarep_len_valid.shape   # n x seq
	print facet_valid.shape         # n x seq x shift_len x 35
        print facet_len_valid.shape     # n x seq
        X_valid = (text_valid_emb, text_len_valid, covarep_valid, covarep_len_valid, facet_valid, facet_len_valid)

	print text_test_emb.shape       # n x seq x 300
        print text_len_test.shape       # n
	print covarep_test.shape        # n x seq x shift_len x 74
        print covarep_len_test.shape    # n x seq
	print facet_test.shape          # n x seq x shift_len x 35
        print facet_len_test.shape      # n x seq
        X_test = (text_test_emb, text_len_test, covarep_test, covarep_len_test, facet_test, facet_len_test)

	if gc.sentiment == 'happy':
		return X_train, happy_train, X_valid, happy_valid, X_test, happy_test
	if gc.sentiment == 'angry':
		return X_train, angry_train, X_valid, angry_valid, X_test, angry_test
	if gc.sentiment == 'sad':
		return X_train, sad_train, X_valid, sad_valid, X_test, sad_test
	if gc.sentiment == 'neutral':
		return X_train, neutral_train, X_valid, neutral_valid, X_test, neutral_test

def fetch_data(cls):
    global X_train
    global y_train
    global X_valid
    global y_valid
    global X_test
    global y_test
    if X_train == None:
        X_train, y_train, X_valid, y_valid, X_test, y_test = get_data()
    if cls == "train":
        text, text_len, covarep, covarep_len, facet, facet_len = X_train
        label = y_train
    elif cls == "valid":
        text, text_len, covarep, covarep_len, facet, facet_len = X_valid
        label = y_valid
    elif cls == "test":
        text, text_len, covarep, covarep_len, facet, facet_len = X_test
        label = y_test
    else:
        print "Data class should be in {train, valid, test}!}"
        assert False
    return text, text_len, covarep, covarep_len, facet, facet_len, label

X_train = None
y_train = None
X_valid = None
y_valid = None
X_test = None
y_test = None
if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data()
    print y_train.shape
    print y_valid.shape
    print y_test.shape
