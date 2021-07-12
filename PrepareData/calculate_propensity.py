import time
from Utils.LoadData import load_context_feature
from Utils.LoadData import load_para
from Utils.QueryStructure import Query
import numpy as np
import random
import json
def cal_propensity(input_feature_path, input_para_path, out_propensity_path, topK_position, feature_dimension, weight_range):
    start_time = time.time()
    random.seed()
    query = load_context_feature(input_feature_path)
    w = load_para(input_para_path, feature_dimension, weight_range)
    with open(out_propensity_path, "w") as f:
        for temp in query:
            query_id = temp.qid
            context_feature = temp.context_feature
            power = max(np.dot(w, context_feature) + 1, 0)
            propensity = [pow(1 / k, power) for k in range(1, topK_position + 1)]
            propensity_str = json.dumps([str(p) for p in propensity])
            content = "qid:%d %s\n" % (query_id, propensity_str)
            f.write(content)
    end_time = time.time()
    print("calculate propensity usage time %f" % (end_time - start_time))

if __name__=="__main__":
    topK_position = 10
    feature_dimension = 10
    weight_range = 0.7

    train_slice_para = "../Data/SliceData/w=0.7_relevance=9_author/train_slice_para.txt"
    valid_slice_para = "../Data/SliceData/w=0.7_relevance=9_author/valid_slice_para.txt"
    test_slice_para = "../Data/SliceData/w=0.7_relevance=9_author/test_slice_para.txt"

    cal_propensity("../Data/SliceData/w=0.7_relevance=9_author/train_slice_feature.txt",
                   train_slice_para,
                   "../Data/trainData/w=0.7_relevance=9_author/train_slice_propensity.txt", topK_position, feature_dimension, weight_range)

    cal_propensity("../Data/SliceData/w=0.7_relevance=9_author/test_feature.txt",
                   train_slice_para,
                   "../Data/trainData/w=0.7_relevance=9_author/test_propensity.txt", topK_position, feature_dimension, weight_range)

    cal_propensity("../Data/SliceData/w=0.7_relevance=9_author/valid_feature.txt",
                   train_slice_para,
                   "../Data/trainData/w=0.7_relevance=9_author/valid_propensity.txt", topK_position, feature_dimension, weight_range)
