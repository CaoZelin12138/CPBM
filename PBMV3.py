import os
import sys
from Utils.LoadData import load_click_log
from Utils.LoadData import load_para
from Utils.LoadData import load_propensity
from collections import defaultdict, Counter
import numpy as np
import scipy.optimize as opt
from PrepareData.ProcessDataV3 import process_data
from PrepareData.ProcessDataV3 import weight
import json
import random
def intervention_harvesting(click_log_list, weight, topK_position):
    log0 = click_log_list[0]
    log1 = click_log_list[1]
    # log2 = click_log_list[2]
    # log3 = click_log_list[3]
    # log4 = click_log_list[4]
    # log5 = click_log_list[5]
    # log6 = click_log_list[6]
    # log7 = click_log_list[7]

    qid_different_pair = {}
    qid_different = {}

    signal = set()

    click = np.zeros(shape=(topK_position, topK_position))
    not_click = np.zeros(shape=(topK_position, topK_position))

    for i in range(len(click_log_list)):
        logger1 = eval("log{}".format(i))
        logger1_str = "log{}".format(i + 1)
        # print(logger1_str)
        for j in range(i + 1, len(click_log_list)):

            logger2 = eval('log{}'.format(j))
            logger2_str = "log{}".format(j + 1)

            for query0, query1 in zip(logger1, logger2):
                diff = 0
                assert query0.qid == query1.qid
                qid = query0.qid
                docs0 = query0.doc
                docs1 = query1.doc
                for rk0, doc0 in enumerate(docs0, start=1):
                    if rk0 > topK_position:
                        break
                    doc_id0, click_label0 = doc0
                    for rk1, doc1 in enumerate(docs1, start=1):
                        if rk1 > topK_position:
                            break
                        if rk1 == rk0:
                            continue
                        doc_id1, click_label1 = doc1
                        if doc_id0 == doc_id1:

                            doc_id0_click = click_label0 / weight[(qid, doc_id0, rk0)]
                            doc_id0_not_click = (1 - click_label0) / float(weight[(qid, doc_id0, rk0)])

                            doc_id1_click = click_label1 / weight[(qid, doc_id1, rk1)]
                            doc_id1_not_click = (1 - click_label1) / float(weight[(qid, doc_id1, rk1)])

                            if (qid, doc_id0, rk0, rk1, logger1_str) not in signal:

                                click[rk0-1][rk1-1] += doc_id0_click
                                not_click[rk0-1][rk1-1] += doc_id0_not_click
                                signal.add((qid, doc_id0, rk0, rk1, logger1_str))

                            if (qid, doc_id1, rk1, rk0, logger2_str) not in signal:
                                click[rk1 - 1][rk0 - 1] += doc_id1_click
                                not_click[rk1 - 1][rk0 - 1] += doc_id1_not_click

                                signal.add((qid, doc_id1, rk1, rk0, logger2_str))

    return click, not_click

def likelihood(p, r, c, not_c, M):
    pr = p.reshape([-1, 1]) * r
    obj = np.sum(c * np.log10(pr) + not_c * np.log10(1 - pr))
    return obj

def all_pair_estimator(deminsion, click, not_click, propensity_path, out_path):
    a, b = 1e-6, 1 - 1e-6
    x0 = np.random.uniform(a, b, deminsion * deminsion + deminsion)
    bnds = np.array([(a, b)] * (deminsion * deminsion + deminsion))
    def f(x):
        p = x[:deminsion]
        r = x[deminsion:].reshape([deminsion, deminsion])
        r_symm = (r + r.transpose()) / 2
        return -likelihood(p, r_symm, click, not_click, deminsion)

    ret = opt.minimize(f, x0, method='L-BFGS-B', bounds=bnds)
    predict_prop = ret.x[:deminsion]
    predict_prop = predict_prop / predict_prop[0]

    np.savetxt(out_path, predict_prop)



    true_prop = load_propensity(propensity_path)

    print('Relative Error on training set: {}'.format(avg_rel_err(predict_prop, true_prop)))


def avg_rel_err(predict_prop, true_prop):
    c = true_prop - predict_prop
    d = np.absolute((true_prop - predict_prop) / true_prop)
    return np.sum(np.absolute((true_prop - predict_prop) / true_prop))

def test(test_propensity_path, out_path):

    true_propensity = load_propensity(test_propensity_path)
    predict_propensity = np.loadtxt(out_path)

    N = true_propensity.shape[0]
    predict_propensity = np.tile(predict_propensity, (N, 1))

    print('Relative Error on test set: {}'.format(avg_rel_err(predict_propensity, true_propensity)))

def evaluation(predict_prop, true_prop):
    true_propensity = load_propensity(true_prop)
    predict_propensity = np.loadtxt(predict_prop)

    N = true_propensity.shape[0]
    predict_propensity = np.tile(predict_propensity, (N, 1))

    print('Relative Error on test set: {}'.format(avg_rel_err(predict_propensity, true_propensity)))
def setup_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    setup_seed(30)
    deminsion = 10
    click_log1_input_path = "../Data/SliceData/w=10_relevance=9_author/train_slice_click_log1.txt"
    click_log2_input_path = "../Data/SliceData/w=10_relevance=9_author/train_slice_click_log2.txt"
    # click_log3_input_path = "../Data/SliceData/train_slice_click_log3.txt"
    # click_log4_input_path = "../Data/SliceData/train_slice_click_log4.txt"
    # click_log5_input_path = "../Data/SliceData/train_slice_click_log5.txt"
    # click_log6_input_path = "../Data/SliceData/train_slice_click_log6.txt"
    # click_log7_input_path = "../Data/SliceData/train_slice_click_log7.txt"
    # click_log8_input_path = "../Data/SliceData/train_slice_click_log8.txt"
    context_feature_input_path = "../Data/SliceData/w=10_relevance=9_author/train_slice_feature.txt"
    out_path = "../NetWorkResult/w=10_relevance=9_author/PBM.txt"
    train_propensity_path = "../Data/trainData/w=10_relevance=9_author/train_slice_propensity.txt"
    test_propensity_path = "../Data/trainData/w=10_relevance=9_author/test_propensity.txt"
    topK_position = 10
    click_log_file_path_list = []
    click_log_file_path_list.append(click_log1_input_path)
    click_log_file_path_list.append(click_log2_input_path)
    # click_log_file_path_list.append(click_log3_input_path)
    # click_log_file_path_list.append(click_log4_input_path)
    # click_log_file_path_list.append(click_log5_input_path)
    # click_log_file_path_list.append(click_log6_input_path)
    # click_log_file_path_list.append(click_log7_input_path)
    # click_log_file_path_list.append(click_log8_input_path)

    click_log_list = process_data(click_log_file_path_list)
    weight = weight(click_log_list, topK_position)
    click, not_click = intervention_harvesting(click_log_list, weight, topK_position)
    all_pair_estimator(deminsion, click, not_click, train_propensity_path, out_path)
    test(test_propensity_path, out_path)
    # evaluation("../NetWorkResult/uniformRange=4_relevancePart=9_contextFeatureIsAuthor/twoRanker_PBM_prediction.txt",
    #            "../Data/trainData/uniformRange=4_relevancePart=9_contextFeatureIsAuthor/test_propensity.txt")
