# -*- coding: utf-8 -*-
from Utils.QueryStructure import Query
import json
import os
import random
from sklearn.datasets import load_svmlight_file
import numpy as np
def load_query_item(file_path):
    res = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            content = line.split(" ", 2)
            relevance_label = int(content[0])
            query_id = int(content[1].split(":")[1])
            query_doc_feature = content[2]
            if not res or res[-1].qid != query_id:
                q = Query(query_id)
                q.doc.append((relevance_label, query_doc_feature))
                res.append(q)
            else:
                res[-1].doc.append((relevance_label, query_doc_feature))
    return res

def read_index(index_file_path, dimension):
    if os.path.exists(index_file_path):
        with open(index_file_path, "r") as f:
            index = json.load(f)
            return index
    candis = [362, 431, 668, 425, 331, 344, 646, 384, 90, 186, 407, 647, 243, 392, 561, 413, 145, 506, 620, 488, 151, 676, 501, 417, 503, 536, 69, 479, 689, 617]
    # candis = range(90, 687)
    index = random.sample(candis, dimension)
    # idx = random.sample(candis, dim)
    print('Select %s' % (index))

    with open(index_file_path, "w") as f:
        json.dump(index, f)
    return index

# 这里的复现方式与原文好像不太一样
def prepare_context_feature(input_file_path, dimension, index):
    res = []
    X, y, qids = load_svmlight_file(input_file_path, query_id=True)
    last_qid = 0
    temp_X = []
    click_situation = []

    for i in range(y.shape[0]):
        qid = qids[i]
        if qid != last_qid:
            if temp_X:
                # if qid == 46:
                #     print("aaa")
                a = np.concatenate(temp_X, axis=0)
                b = np.mean(np.concatenate(temp_X, axis=0), axis=0)
                c = np.take(np.mean(np.concatenate(temp_X, axis=0), axis=0), index[:dimension])
                # feat = np.mean(np.concatenate(temp_X, axis=0), axis=0)
                feat = np.take(np.mean(np.concatenate(temp_X, axis=0), axis=0), index[:dimension])
                res.append(Query(last_qid, context_feature=feat))
            elif len(click_situation) != 0:
                click_situation_array = np.array(click_situation)
                false_num = np.sum(click_situation_array == 0)
                if false_num == len(click_situation):
                    res.append(Query(last_qid, context_feature=np.zeros(dimension)))
            last_qid = qid
            temp_X.clear()
            click_situation.clear()
        if y[i]:
            temp_X.append(X[i].toarray())
            click_situation.append(True)
        else:
            click_situation.append(False)
    # feat = np.mean(np.concatenate(temp_X, axis=0), axis=0)
    a = res[-1]
    feat = np.take(np.mean(np.concatenate(temp_X, axis=0), axis=0), index[:dimension])
    res.append(Query(last_qid, context_feature=feat))
    return res

def load_context_feature(input_file_path):
    res = []
    with open(input_file_path, 'r') as fin:
        for line in fin:
            line = line.strip()
            content = line.split(' ')
            qid = int(content[0].split(':')[1])
            feat = np.array([float(tok.split(':')[1]) for tok in content[1:]])
            res.append(Query(qid, context_feature=feat))
    return res

def load_para(input_file_path, dimension, range):
    if os.path.exists(input_file_path):
        w = np.loadtxt(input_file_path)
        if w[0] == range:
            return w[1:]
    w = np.random.uniform(-range, range, dimension)
    w = w - np.mean(w)
    x = np.hstack((range, w))
    np.savetxt(input_file_path, x)
    print('Select %s' % (w))
    return w

def load_click_log(input_file_path):
    logs = []
    with open(input_file_path, 'r') as fin:
        for line_ in fin:
            line = line_.strip()
            toks = line.split(' ')
            assert len(toks) == 3
            delta = int(toks[0])
            qid = int(toks[1].split(':')[1])
            doc_id = int(toks[2])
            if not logs or not logs[-1].qid == qid:
                q = Query(qid)
                q.doc.append((doc_id, delta))
                logs.append(q)
            else:
                logs[-1].doc.append((doc_id, delta))
    return logs

def load_propensity(input_file_path):
    p = []
    with open(input_file_path, 'r') as f:
        for line_ in f:
            line = line_.strip()
            toks = line.split(' ', 1)
            qid = int(toks[0].split(':')[1])
            propensity = json.loads(toks[1])
            prop = np.asarray([float(tok) for tok in propensity])
            p.append(prop)
    p = np.array(p)
    return p
