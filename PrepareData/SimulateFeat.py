import os
import sys
import random
import time
import numpy as np
from Utils.LoadData import read_index
from Utils.LoadData import prepare_context_feature
from sklearn.linear_model import LogisticRegression
from Utils.LoadData import load_context_feature
def simulate_feature(index_file_path, dimension, relevant_random_control, data_input_path, data_output_path):
    random.seed()
    np.random.seed()
    start_time = time.time()

    index = read_index(index_file_path, dimension)
    query = prepare_context_feature(data_input_path, dimension, index)
    relevant_part = int(dimension * relevant_random_control)
    random_part = dimension - relevant_part

    for temp in query:
        # relevant_context_feature = 2 * temp.context_feature - 1

        random_context_feature = np.random.normal(0, 0.35, random_part)
        if relevant_part == 0:
            temp.context_feature = random_context_feature.tolist()
        elif random_part == 0:
            # temp.context_feature = 2 * temp.context_feature - 1
            temp.context_feature = temp.context_feature
        else:
            temp.context_feature = np.concatenate((temp.context_feature[:relevant_part], random_context_feature)).tolist()

    write_to_file(query, data_output_path)
    end_time = time.time()

    print("Simulate feature usage time %f" % (end_time - start_time))

def write_to_file(data, out_path):
    with open(out_path, "w") as f:
        for temp in data:
            content = "qid:%d" % (temp.qid)
            f.write(content)
            for index, val in enumerate(temp.context_feature, start=1):
                f.write(" %d:%s" % (index, val))
            f.write("\n")


if __name__ == '__main__':
    train_index = "../Data/SliceData/w=0.7_relevance=9_author/train_index.json"
    relevant_random_control = 0.9
    simulate_feature(train_index, 10, relevant_random_control,
                     "../Data/SliceData/w=0.7_relevance=9_author/train_slice.txt",
                     "../Data/SliceData/w=0.7_relevance=9_author/train_slice_feature.txt")

    simulate_feature(train_index, 10, relevant_random_control,
                     "../Data/vali.txt", "../Data/SliceData/w=0.7_relevance=9_author/valid_feature.txt")

    simulate_feature(train_index, 10, relevant_random_control,
                     "../Data/test.txt", "../Data/SliceData/w=0.7_relevance=9_author/test_feature.txt")
