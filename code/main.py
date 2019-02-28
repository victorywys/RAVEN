from consts import global_consts as gc
from model import Net

if gc.dataset == "iemocap":
    from ie_dataset import IEDataset as ds
else:
    from MOSI_dataset import MOSIDataset as ds
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from sklearn.metrics import f1_score
import sys
import signal
import os

def cor(X, Y):
    return np.mean(((X - np.mean(X)) * (Y - np.mean(Y)))) / (np.var(X) ** 0.5) / (np.var(Y) ** 0.5)

def logSummary():
    if gc.dataset == "iemocap":
        print "best epoch: %d" % gc.best_epoch
        print "highest training F1: %f" % gc.max_train_f1
        print "highest testing F1: %f" % gc.max_test_f1
        print "highest validation F1: %f" % gc.max_valid_f1
        print "test F1 when validation F1 is the highest: %f" % gc.test_f1_at_valid_max
        print "highest training accuracy: %f" % gc.max_train_acc
        print "highest testing accuracy: %f" % gc.max_test_acc
        print "highest validation accuracy: %f" % gc.max_valid_acc
        print "test accuracy when validation accuracy is the highest: %f" % gc.test_acc_at_valid_max
        print "highest training precision: %f" % gc.max_train_prec
        print "highest testing precision: %f" % gc.max_test_prec
        print "highest validation precision: %f" % gc.max_valid_prec
        print "test precision when validation precision is the highest: %f" % gc.test_prec_at_valid_max
        print "highest training recall: %f" % gc.max_train_recall
        print "highest testing recall: %f" % gc.max_test_recall
        print "highest validation recall: %f" % gc.max_valid_recall
        print "test recall when validation recall is the highest: %f" % gc.test_recall_at_valid_max
    elif gc.dataset == "MOSI":
        print "best epoch: %d" % gc.best_epoch
        print "lowest training MAE: %f" % gc.min_train_mae
        print "lowest testing MAE: %f" % gc.min_test_mae
        print "lowest validation MAE: %f" % gc.min_valid_mae
        print "test MAE when validation MAE is the lowest: %f" % gc.test_mae_at_valid_min
        print "highest testing correlation: %f" % gc.max_test_cor
        print "highest validation correlation: %f" % gc.max_valid_cor
        print "test correlation when validation correlation is the highest: %f" % gc.test_cor_at_valid_max
        print "highest testing accuracy: %f" % gc.max_test_acc
        print "highest validation accuracy: %f" % gc.max_valid_acc
        print "test accuracy when validation accuracy is the highest: %f" % gc.test_acc_at_valid_max

def stopTraining(signum, frame):
    global savedStdout
    logSummary()
    sys.stdout = savedStdout
    sys.exit()

def train_model(arg_dict):
    try:
        signal.signal(signal.SIGINT, stopTraining)
        signal.signal(signal.SIGTERM, stopTraining)
    except:
        pass

    for key in arg_dict:
        if gc.__dict__.has_key(key):
            gc.__dict__[key] = arg_dict[key]
    global savedStdout
    savedStdout = sys.stdout

    if gc.log_path != None:
        dir_path = "%s%d" % (gc.log_path, gc.HPID)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        log_file = "%s/print.log" % dir_path
        f = open(log_file, "w+")
        sys.stdout = f

    train_dataset = ds(gc.data_path, cls="train")
    train_loader = Data.DataLoader(
                dataset=train_dataset,
                batch_size=gc.batch_size,
                shuffle=True,
                num_workers=1,
        )

    test_dataset = ds(gc.data_path, cls="test")
    test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size = gc.batch_size,
            shuffle=False,
            num_workers=1,
        )

    valid_dataset = ds(gc.data_path, cls="valid")
    valid_loader = Data.DataLoader(
            dataset=valid_dataset,
            batch_size = gc.batch_size,
            shuffle=False,
            num_workers=1,
        )

    print >> savedStdout, "HPID:%d:Data Successfully Loaded." % gc.HPID

    device = torch.device("cuda:%d" % gc.cuda if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gc.cuda)
    gc.device = device
    print "running device: ", device

    net = Net()
    print net
    net.to(device)

    if gc.dataset == "iemocap":
        criterion = nn.BCELoss()
    elif gc.dataset == "MOSI":
        criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=gc.learning_rate)

    gc().logParameters()

    running_loss = 0.0
    for epoch in range(gc.epoch_num):
        if epoch % 10 == 0:
            print >> savedStdout, "HPID:%d:Training Epoch %d." % (gc.HPID, epoch)
        if epoch % 100 == 0:
            logSummary()
        with torch.no_grad():
            print "Epoch #%d results:" % epoch
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            tot_num = 0
            tot_err = 0
            tot_right = 0
            label_all = []
            output_all = []
            for data in test_loader:
                words, covarep, covarepLen, facet, facetLen, inputLen, labels = data
                labels = labels.to(torch.float)
                if covarep.size()[0] == 1:
                    continue
                words, covarep, covarepLen, facet, facetLen, inputLen, labels = words.to(device), covarep.to(device), covarepLen.to(device), facet.to(device), facetLen.to(device), inputLen.to(device), labels.to(device)
                outputs = net(words, covarep, covarepLen, facet, facetLen, inputLen)
                if gc.dataset == "iemocap":
                    output_all.extend(torch.ge(outputs, 0).to(torch.long))
                else:
                    output_all.extend(outputs.tolist())
                label_all.extend(labels.tolist())

                if gc.dataset == "iemocap":
                    tp += torch.sum(torch.ge(outputs, 0).to(torch.long) * labels.to(torch.long))
                    fp += torch.sum(torch.ge(outputs, 0).to(torch.long) * (1 - labels.to(torch.long)))
                    tn += torch.sum(torch.le(outputs, 0).to(torch.long) * (1 - labels.to(torch.long)))
                    fn += torch.sum(torch.le(outputs, 0).to(torch.long) * labels.to(torch.long))
                elif gc.dataset == "MOSI":
                    err = torch.sum(torch.abs(outputs - labels))
                    tot_err += err
                    tot_num += covarep.size()[0]
                    labels = torch.ge(labels, 0)
                    pred = torch.ge(outputs, 0)
                    tot_right += torch.sum(torch.eq(labels, pred))
            if gc.dataset == "iemocap":
                tp, fp, tn, fn = tp.item(), fp.item(), tn.item(), fn.item()
                if tp == 0:
                    test_prec = 0
                    test_recall = 0
                    test_f1 = 0
                    test_acc = 0
                else:
                    test_prec = float(tp) / (tp + fp)
                    test_recall = float(tp) / (tp + fn)
                    test_f1 = f1_score(label_all, output_all, average="weighted")
                    test_acc = float(tp + tn) / (tp + tn + fp + fn)
                print "\ttest precision: %f" % test_prec
                print "\ttest recall: %f" % test_recall
                print "\ttest F1: %f" % test_f1
                print "\ttest accuracy: %f" % test_acc
            elif gc.dataset == "MOSI":
                test_mae = tot_err / tot_num
                test_cor = cor(output_all, label_all)
                test_acc = float(tot_right) / tot_num
                print "\ttest Correlation coefficient: %f" % cor(output_all, label_all)
                print "\ttest mean error: %f" % test_mae
                print "\ttest accuracy: %f" % test_acc

            tp = 0
            fp = 0
            tn = 0
            fn = 0
            tot_num = 0
            tot_err = 0
            tot_right = 0
            label_all = []
            output_all = []
            for data in valid_loader:
                words, covarep, covarepLen, facet, facetLen, inputLen, labels = data
                labels = labels.to(torch.float)
                if covarep.size()[0] == 1:
                    continue
                words, covarep, covarepLen, facet, facetLen, inputLen, labels = words.to(device), covarep.to(device), covarepLen.to(device), facet.to(device), facetLen.to(device), inputLen.to(device), labels.to(device)
                outputs = net(words, covarep, covarepLen, facet, facetLen, inputLen)
                if gc.dataset == "iemocap":
                    output_all.extend(torch.ge(outputs, 0).to(torch.long))
                else:
                    output_all.extend(outputs.data.cpu().tolist())
                label_all.extend(labels.data.cpu().tolist())
                if gc.dataset == "iemocap":
                    tp += torch.sum(torch.ge(outputs, 0).to(torch.long) * labels.to(torch.long))
                    fp += torch.sum(torch.ge(outputs, 0).to(torch.long) * (1 - labels.to(torch.long)))
                    tn += torch.sum(torch.le(outputs, 0).to(torch.long) * (1 - labels.to(torch.long)))
                    fn += torch.sum(torch.le(outputs, 0).to(torch.long) * labels.to(torch.long))
                elif gc.dataset == "MOSI":
                    err = torch.sum(torch.abs(outputs - labels))
                    tot_err += err
                    labels = torch.ge(labels, 0)
                    pred = torch.ge(outputs, 0)
                    tot_right += torch.sum(torch.eq(labels, pred)).data.cpu().tolist()
                    tot_num += covarep.size()[0]
            if gc.dataset == "iemocap":
                tp, fp, tn, fn = tp.item(), fp.item(), tn.item(), fn.item()
                if tp == 0:
                    valid_prec = 0
                    valid_recall = 0
                    valid_f1 = 0
                    valid_acc = 0
                else:
                    valid_prec = tp / float(tp + fp)
                    valid_recall = tp / float(tp + fn)
                    valid_f1 = f1_score(label_all, output_all, average="weighted")
                    valid_acc = float(tp + tn) / (tp + tn + fp + fn)
                if (tp + fp + tn + fn) != 0:
                    print "\tvalid precision: %f" % valid_prec
                    print "\tvalid recall: %f" % valid_recall
                    print "\tvalid F1: %f" % valid_f1
                    print "\tvalid accuracy: %f" % valid_acc
                    if valid_f1 > gc.max_valid_f1:
                        gc.max_valid_f1 = valid_f1
                        gc.test_f1_at_valid_max = test_f1
                        gc.best_epoch = epoch + 1
                    if valid_prec > gc.max_valid_prec:
                        gc.max_valid_prec = valid_prec
                        gc.test_prec_at_valid_max = test_prec
                    if valid_recall > gc.max_valid_recall:
                        gc.max_valid_recall = valid_recall
                        gc.test_recall_at_valid_max = test_recall
                    if valid_acc > gc.max_valid_acc:
                        gc.max_valid_acc = valid_acc
                        gc.test_acc_at_valid_max = test_acc
                    if test_f1 > gc.max_test_f1:
                        gc.max_test_f1 = test_f1
                    if test_prec > gc.max_test_prec:
                        gc.max_test_prec = test_prec
                    if test_recall > gc.max_test_recall:
                        gc.max_test_recall = test_recall
                    if test_acc > gc.max_test_acc:
                        gc.max_test_acc = test_acc
            elif gc.dataset == "MOSI":
                if tot_num != 0:
                    valid_mae = tot_err / tot_num
                    valid_cor = cor(output_all, label_all)
                    valid_acc = float(tot_right) / tot_num
                    print "\tvalid Correlation coefficient: %f" % cor(output_all, label_all)
                    print "\tvalid mean error: %f" % (valid_mae)
                    print "\tvalid accuracy: %f" % (valid_acc)
                    if valid_mae < gc.min_valid_mae:
                        gc.min_valid_mae = valid_mae
                        gc.test_mae_at_valid_min = test_mae
                        gc.best_epoch = epoch + 1
                    if valid_cor > gc.max_valid_cor:
                        gc.max_valid_cor = valid_cor
                        gc.test_cor_at_valid_max = test_cor
                    if valid_acc > gc.max_valid_acc:
                        gc.max_valid_acc = valid_acc
                        gc.test_acc_at_valid_max = test_acc
                    if test_mae < gc.min_test_mae:
                        gc.min_test_mae = test_mae
                    if test_cor > gc.max_test_cor:
                        gc.max_test_cor = test_cor
                    if test_acc > gc.max_test_acc:
                        gc.max_test_acc = test_acc

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        tot_num = 0
        tot_err = 0
        tot_right = 0
        label_all = []
        output_all = []
        for i, data in enumerate(train_loader):
            words, covarep, covarepLen, facet, facetLen, inputLen, labels = data
            if covarep.size()[0] == 1:
                continue
            labels = labels.to(torch.float)
            words, covarep, covarepLen, facet, facetLen, inputLen, labels = words.to(device), covarep.to(device), covarepLen.to(device), facet.to(device), facetLen.to(device), inputLen.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(words, covarep, covarepLen, facet, facetLen, inputLen)
            if gc.dataset == "iemocap":
                for l in torch.ge(outputs, 0).to(torch.long):
                    output_all.append(l)
                for l in labels:
                    label_all.append(l)
                tp += torch.sum(torch.ge(outputs, 0).to(torch.long) * labels.to(torch.long))
                fp += torch.sum(torch.ge(outputs, 0).to(torch.long) * (1 - labels.to(torch.long)))
                tn += torch.sum(torch.le(outputs, 0).to(torch.long) * (1 - labels.to(torch.long)))
                fn += torch.sum(torch.le(outputs, 0).to(torch.long) * labels.to(torch.long))
            elif gc.dataset == "MOSI":
                err = torch.sum(torch.abs(outputs - labels))
                tot_right += torch.sum(torch.eq(torch.sign(labels), torch.sign(outputs)))
                tot_err += err
                tot_num += covarep.size()[0]
            if gc.dataset == "iemocap":
                loss = criterion(torch.sigmoid(outputs), labels)
            elif gc.dataset == "MOSI":
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        if gc.dataset == "iemocap":
            tp, fp, tn, fn = tp.item(), fp.item(), tn.item(), fn.item()
            if tp + fp + tn + fn != 0:
                if tp == 0:
                    train_prec = 0
                    train_recall = 0
                    train_f1 = 0
                    train_acc = 0
                else:
                    train_prec = float(tp) / (tp + fp)
                    train_recall = float(tp) / (tp + fn)
                    train_f1 = f1_score(label_all, output_all, average="weighted")
                    train_acc = float(tp + tn) / (tp + tn + fp + fn)
                print "\ttrain precision: %f" % train_prec
                print "\ttrain recall: %f" % train_recall
                print "\ttrain F1: %f" % train_f1
                print "\ttrain accuracy: %f" % train_acc
                if train_f1 > gc.max_train_f1:
                    gc.max_train_f1 = train_f1
                if train_prec > gc.max_train_prec:
                    gc.max_train_prec = train_prec
                if train_recall > gc.max_train_recall:
                    gc.max_train_recall = train_recall
                if train_acc > gc.max_train_acc:
                    gc.max_train_acc = train_acc
        elif gc.dataset == "MOSI":
            train_mae = tot_err / tot_num
            train_acc = float(tot_right) / tot_num
            print "\ttrain mean error: %f" % train_mae
            print "\ttrain acc: %f" % train_acc
            if train_mae < gc.min_train_mae:
                gc.min_train_mae = train_mae
            if train_acc > gc.max_train_acc:
                gc.max_train_acc = train_acc

    logSummary()
    if gc.log_path != None:
        sys.stdout = savedStdout
        with open("%ssummary.csv" % gc.log_path, "a+") as f:
            for arg in arg_dict:
                f.write("%s," % str(arg_dict[arg]))
            f.write("%d," % best_epoch)
            if gc.dataset == "iemocap":
                f.write("%f," % max_test_f1)
                f.write("%f," % max_test_prec)
                f.write("%f," % max_test_recall)
                f.write("%f," % max_test_acc)
                f.write("%f," % test_f1_at_valid_max)
                f.write("%f\n" % test_acc_at_valid_max)
            else:
                f.write("%f," % min_test_mae)
                f.write("%f," % max_test_cor)
                f.write("%f," % max_test_acc)
                f.write("%f," % test_mae_at_valid_min)
                f.write("%f," % test_cor_at_valid_max)
                f.write("%f,\n" % test_acc_at_valid_max)

if __name__ == "__main__":
    train_model({ "log_path" : None })
