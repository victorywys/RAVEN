class global_consts():
    debug = False

    platform = "server"
    cuda = 1

    dataset = "MOSI"
    data_path = None
    raw_path = None
    if dataset == "MOSI":
        data_path = "/home/data/wangyansen/cmumosi/"
    elif dataset == "iemocap":
        data_path = "/media/bighdd7/yansen/code/tools/iemocap/"
        raw_path =  "/media/bighdd4/Paul/mosi2/experiments/iemocap/"
    sentiment = "sad" # for IEMOCAP, choose from happy, angry, sad and neutral

    old_data = True

    log_path = "/media/bighdd7/yansen/code/DALSTM/result/iemocap/happy/"
    HPID = -1

    lastState = True
    layer = 1

    no_sp = True

    batch_size = 20
    epoch_num = 500
    learning_rate = 0.001

    padding_len = 50

    shift = True
    sub_freq = 20
    shift_padding_len = 20
    shift_weight = 0.2

    dropProb = 0.2

    cellDim = 150
    normDim = 150
    hiddenDim = 300

    device = None

    covarepDim = 0
    wordDim = 0
    facetDim = 0
    smileDim = 0

    best_epoch = 0

    max_train_f1 = 0
    max_test_f1 = 0
    max_valid_f1 = 0
    max_test_prec = 0
    max_valid_prec = 0
    max_train_prec = 0
    max_train_recall = 0
    max_test_recall = 0
    max_valid_recall = 0
    max_train_acc = 0
    max_test_acc = 0
    max_valid_acc = 0
    test_f1_at_valid_max = 0
    test_prec_at_valid_max = 0
    test_recall_at_valid_max = 0
    test_acc_at_valid_max = 0

    min_train_mae = 10
    min_test_mae = 10
    max_test_cor = 0
    min_valid_mae = 10
    max_valid_cor = 0
    test_mae_at_valid_min = 10
    test_cor_at_valid_max = 0
    test_acc_at_valid_max = 0

    def logParameters(self):
        print "Hyperparameters:"
        for name in dir(global_consts):
            if name.find("__") == -1 and name.find("max") == -1 and name.find("min") == -1:
                print "\t%s: %s" % (name, str(getattr(global_consts, name)))
