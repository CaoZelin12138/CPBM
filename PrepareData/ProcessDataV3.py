import time
import numpy as np
import random
from Utils.LoadData import load_context_feature
from Utils.LoadData import load_click_log
from Utils.QueryStructure import Query
from collections import defaultdict
from collections import Counter
import copy
import json

def process_data(click_log_path_list):
    random.seed()
    res = []
    total = 0
    for temp in click_log_path_list:
        log = load_click_log(temp)
        for each_query in log:
            total += len(each_query.doc)
            # if each_query.qid == 3:
            #     print("aaaa")
        res.append(log)
    return res

def write_dataset_number(array, out_path):
    with open(out_path, "w") as f:
        f.write(str(array.shape[0]))

def weight(click_log_list, topK_position):
    log0 = click_log_list[0]
    log1 = click_log_list[1]
    # log2 = click_log_list[2]
    # log3 = click_log_list[3]
    # log4 = click_log_list[4]
    # log5 = click_log_list[5]
    # log6 = click_log_list[6]
    # log7 = click_log_list[7]

    # 每份点击日志有多少条搜索query
    n0 = len(log0)
    n1 = len(log1)
    # n2 = len(log2)
    # n3 = len(log3)
    # n4 = len(log4)
    # n5 = len(log5)
    # n6 = len(log6)
    # n7 = len(log7)
    # n8 = len(log8)

    total = float(n0 + n1)
    assert n0 == n1
    # assert n2 == n3

    w = Counter()
    for i in range(len(click_log_list)):
        logger = eval('log{}'.format(i))
        for q in logger:
            qid = q.qid
            docs = q.doc
            for rk, doc in enumerate(docs, start=1):
                if rk > topK_position:
                    break
                doc_id, _ = doc
                # before = w[(qid, doc_id, rk)]
                w.update({(qid, doc_id, rk): eval('n{}'.format(i)) / total})
                # after = w[(qid, doc_id, rk)]
    return w

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
    qid_position_different = {}
    c_cnt = Counter()
    not_c_cnt = Counter()
    signal = set()
    # 支持多份点击日志查找，假如有三份日志需要将上方对应log的注释打开，每份log数量一样
    for i in range(len(click_log_list)):
        logger1 = eval("log{}".format(i))
        logger1_str = "log{}".format(i + 1)

        for j in range(i + 1, len(click_log_list)):

            logger2 = eval('log{}'.format(j))
            logger2_str = "log{}".format(j + 1)

            for query0, query1 in zip(logger1, logger2):
                assert query0.qid == query1.qid
                qid = query0.qid

                docs0 = query0.doc
                docs1 = query1.doc
                for rk0, doc0 in enumerate(docs0, start=1):

                    if rk0 > topK_position:
                        break
                    doc_id0, click_label0 = doc0
                    # 将前10个位置同一个doc被排在不同位置的doc加入干预数据
                    for rk1, doc1 in enumerate(docs1, start=1):
                        # if qid == 3 and doc_id0 == 25:
                        #     print("aaa")
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

                                c_cnt.update({(rk0, rk1, qid): doc_id0_click})
                                not_c_cnt.update({(rk0, rk1, qid): doc_id0_not_click})
                                signal.add((qid, doc_id0, rk0, rk1, logger1_str))

                            if (qid, doc_id1, rk1, rk0, logger2_str) not in signal:

                                c_cnt.update({(rk1, rk0, qid): doc_id1_click})
                                not_c_cnt.update({(rk1, rk0, qid): doc_id1_not_click})
                                signal.add((qid, doc_id1, rk1, rk0, logger2_str))

    return c_cnt, not_c_cnt

def generate_click_non_click_data(c_cnt, not_c_cnt, topK_position, context_feature_input_path, numpy_output_path):
    train_queries = load_context_feature(context_feature_input_path)
    qid_different = {}
    how_many_query = len(train_queries)

    click = np.zeros(shape=(how_many_query, topK_position, topK_position))
    not_click = np.zeros(shape=(how_many_query, topK_position, topK_position))

    for idx, query in enumerate(train_queries):
        qid = query.qid
        is_not_zero = 0
        feat = query.context_feature

        for k in range(topK_position):
            for k_ in range(topK_position):
                if k == k_:
                    continue
                aa = c_cnt[(k + 1, k_ + 1, qid)]
                bb = not_c_cnt[(k + 1, k_ + 1, qid)]
                if c_cnt[(k + 1, k_ + 1, qid)] != 0.0 or not_c_cnt[(k + 1, k_ + 1, qid)] != 0.0:
                    is_not_zero += 1
                click[idx][k][k_] = c_cnt[(k + 1, k_ + 1, qid)]
                not_click[idx][k][k_] = not_c_cnt[(k + 1, k_ + 1, qid)]
        if qid in qid_different:
            qid_different[qid] = qid_different[qid] + is_not_zero
        else:
            qid_different[qid] = is_not_zero

    click_npy_path = numpy_output_path + 'train_click.npy'

    np.save(click_npy_path, (click, not_click))

def generate_feature_data(context_feature_input_path, numpy_output_path):
    train_feat_queries = load_context_feature(context_feature_input_path)
    X = np.array([q.context_feature for q in train_feat_queries])
    train_feat_npy_path = numpy_output_path + 'train_slice_feature.npy'
    np.save(train_feat_npy_path, X)

    valid_feat_path = "../Data/SliceData/w=0.7_relevance=9_author/valid_feature.txt"
    valid_feat_queries = load_context_feature(valid_feat_path)
    X = np.array([q.context_feature for q in valid_feat_queries])

    write_dataset_number(X, numpy_output_path + "valid_feature_number.txt")
    valid_feat_npy_path = numpy_output_path + "valid_feature.npy"
    np.save(valid_feat_npy_path, X)

    test_feat_path = "../Data/SliceData/w=0.7_relevance=9_author/test_feature.txt"
    test_feat_queries = load_context_feature(test_feat_path)
    X = np.array([q.context_feature for q in test_feat_queries])

    write_dataset_number(X, numpy_output_path + "test_feature_number.txt")
    test_feat_npy_path = numpy_output_path + "test_feature.npy"
    np.save(test_feat_npy_path, X)

if __name__=="__main__":
    start_time = time.time()
    click_log1_input_path = "../Data/SliceData/w=0.7_relevance=9_author/train_slice_click_log1.txt"
    click_log2_input_path = "../Data/SliceData/w=0.7_relevance=9_author/train_slice_click_log2.txt"
    # click_log3_input_path = "../Data/SliceData/train_slice_click_log3.txt"
    # click_log4_input_path = "../Data/SliceData/train_slice_click_log4.txt"
    # click_log5_input_path = "../Data/SliceData/train_slice_click_log5.txt"
    # click_log6_input_path = "../Data/SliceData/train_slice_click_log6.txt"
    # click_log7_input_path = "../Data/SliceData/train_slice_click_log7.txt"
    # click_log8_input_path = "../Data/SliceData/train_slice_click_log8.txt"

    context_feature_input_path = "../Data/SliceData/w=0.7_relevance=9_author/train_slice_feature.txt"
    numpy_output_path = "../Data/trainData/w=0.7_relevance=9_author/"

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
    # intervention_harvesting寻找同一个query下，同一个doc被排在不同位置的情况
    c_cnt, not_c_cnt = intervention_harvesting(click_log_list, weight, topK_position)
    # 将intervention_harvesting找到的文档转换为2个10*10的矩阵，一个是点击矩阵，一个是未点击矩阵
    generate_click_non_click_data(c_cnt, not_c_cnt, topK_position, context_feature_input_path,
                                  numpy_output_path)
    # 下面是把生成的上下文信息转换为numpy存储
    generate_feature_data(context_feature_input_path, numpy_output_path)
    end_time = time.time()
    print("Process Data Usage time %f" % (end_time - start_time))