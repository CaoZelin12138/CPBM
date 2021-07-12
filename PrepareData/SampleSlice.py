# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from Utils.LoadData import load_query_item
import time
import copy
def sample_data(input_file_path, fraction, overlap, output_file_path):
    start_time = time.time()

    random.seed()
    query = load_query_item(input_file_path)
    res = 0
    for temp in query:
        res += len(temp.doc)
    print(res)
    random.shuffle(query)

    train_size = len(query)
    slice_size = int(train_size * fraction)
    overlap_size = int(slice_size * overlap)
    non_overlap_size = slice_size - overlap_size
    # print("aaaa")
    # overlap_data = copy.deepcopy(query[0:overlap_size])

    svm_rank_one_train_data = copy.deepcopy(query[0:slice_size])
    svm_rank_two_train_data = copy.deepcopy(svm_rank_one_train_data[0:overlap_size] + query[slice_size: slice_size + non_overlap_size])
    # svm_rank_three_train_data = copy.deepcopy(svm_rank_one_train_data[0:overlap_size] + query[slice_size + non_overlap_size: slice_size + 2 * non_overlap_size])
    # svm_rank_four_train_data = copy.deepcopy(svm_rank_one_train_data[0:overlap_size] + query[slice_size + 2 * non_overlap_size: slice_size + 3 * non_overlap_size])

    # svm_rank_five_train_data = copy.deepcopy(svm_rank_one_train_data[0:overlap_size] + query[slice_size + 3 * non_overlap_size: slice_size + 4 * non_overlap_size])
    # svm_rank_six_train_data = copy.deepcopy(svm_rank_one_train_data[0:overlap_size] + query[slice_size + 4 * non_overlap_size: slice_size + 5 * non_overlap_size])
    # svm_rank_seven_train_data = copy.deepcopy(query[0:overlap_size] + query[slice_size + 5 * non_overlap_size: slice_size + 6 * non_overlap_size])
    # svm_rank_eight_train_data = copy.deepcopy(query[0:overlap_size] + query[slice_size + 6 * non_overlap_size: slice_size + 7 * non_overlap_size])

    train_slice = copy.deepcopy(query[slice_size + non_overlap_size:])

    svm_rank_one_train_data = sorted(svm_rank_one_train_data, key=lambda x:x.qid)
    svm_rank_two_train_data = sorted(svm_rank_two_train_data, key=lambda x:x.qid)
    # svm_rank_three_train_data = sorted(svm_rank_three_train_data, key=lambda x: x.qid)
    # svm_rank_four_train_data = sorted(svm_rank_four_train_data, key=lambda x: x.qid)
    #
    # svm_rank_five_train_data = sorted(svm_rank_five_train_data, key=lambda x: x.qid)
    # svm_rank_six_train_data = sorted(svm_rank_six_train_data, key=lambda x: x.qid)
    # svm_rank_seven_train_data = sorted(svm_rank_seven_train_data, key=lambda x: x.qid)
    # svm_rank_eight_train_data = sorted(svm_rank_eight_train_data, key=lambda x: x.qid)

    train_slice = sorted(train_slice, key=lambda x: x.qid)
    # a = len(svm_rank_one_train_data)
    # b = len(svm_rank_two_train_data)
    # c = len(svm_rank_three_train_data)
    # d = len(svm_rank_four_train_data)
    # e = len(train_slice)
    svm_rank_one_train_data_output_path = output_file_path + "svm_rank_1_train_data.txt"
    svm_rank_two_train_data_output_path = output_file_path + "svm_rank_2_train_data.txt"
    # svm_rank_three_train_data_output_path = output_file_path + "svm_rank_3_train_data.txt"
    # svm_rank_four_train_data_output_path = output_file_path + "svm_rank_4_train_data.txt"
    #
    # svm_rank_five_train_data_output_path = output_file_path + "svm_rank_5_train_data.txt"
    # svm_rank_six_train_data_output_path = output_file_path + "svm_rank_6_train_data.txt"
    # svm_rank_seven_train_data_output_path = output_file_path + "svm_rank_7_train_data.txt"
    # svm_rank_eight_train_data_output_path = output_file_path + "svm_rank_8_train_data.txt"

    train_slice_output_path = output_file_path + "train_slice.txt"

    write_to_file(svm_rank_one_train_data_output_path, svm_rank_one_train_data)
    write_to_file(svm_rank_two_train_data_output_path, svm_rank_two_train_data)
    # write_to_file(svm_rank_three_train_data_output_path, svm_rank_three_train_data)
    # write_to_file(svm_rank_four_train_data_output_path, svm_rank_four_train_data)
    #
    # write_to_file(svm_rank_five_train_data_output_path, svm_rank_five_train_data)
    # write_to_file(svm_rank_six_train_data_output_path, svm_rank_six_train_data)
    # write_to_file(svm_rank_seven_train_data_output_path, svm_rank_seven_train_data)
    # write_to_file(svm_rank_eight_train_data_output_path, svm_rank_eight_train_data)
    write_to_file(train_slice_output_path, train_slice)

    end_time = time.time()
    print("Slice data usage time %f" % (end_time - start_time))

def write_to_file(output_file_path, data):
    with open(output_file_path, "w") as f:
        for query in data:
            for each_doc in query.doc:
                relevance_label, feature = each_doc
                content = "%d qid:%d %s\n" % (relevance_label, query.qid, feature)
                f.write(content)

if __name__ == "__main__":
    # use train data to sample
    file_path = "../Data/train.txt"
    slice_size = 0.01
    overlap = 0.2
    output_file_path = "../Data/SliceData/w=0.7_relevance=9_author/"
    sample_data(file_path, slice_size, overlap, output_file_path)