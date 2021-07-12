# -*- coding: utf-8 -*-
import os,sys
import time
import subprocess
# Windows .exe
SVMRANK_TRAIN = r'start ../SVMModel/svm_rank_learn.exe'
SVMRANK_TEST = r'start ../SVMModel/svm_rank_classify.exe'
# Linux
# SVMRANK_TRAIN = '../SVMModel/svm_rank_learn'
# SVMRANK_TEST = '../SVMModel/svm_rank_classify'

def svm_rank(input_file_path, svm_rank_path):
    rank_file = input_file_path + "train_slice.txt"
    i = 1
    model_file = svm_rank_path + "svmRank" + str(i + 1) + ".model"
    # train a SVMRank as production ranker
    train_command = SVMRANK_TRAIN + " -c 3" + " " + input_file_path + "svm_rank_" + str(i + 1) + "_train_data.txt" + " " + model_file
    print(train_command)
    # Windows下调用
    child = subprocess.Popen(["cmd", "/c", train_command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Linux下调用
    # child = subprocess.Popen(["/bin/sh", "-c", train_command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # print(child.stdout.readlines()[0].decode("gbk"))
    child.wait()
    time.sleep(10)
    # use trained svmRank to get initialized rank prediction
    command = SVMRANK_TEST + " " + rank_file + " " + model_file + " " + input_file_path + "train_score_" + str(i + 1) + ".txt"
    child = subprocess.Popen(["cmd", "/c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(command)
    # Linux下调用
    # child = subprocess.Popen(["/bin/sh", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # print(child.stdout.readlines())
    child.wait()

if __name__== "__main__":
    input_file_path = "../Data/SliceData/w=0.7_relevance=9_author/"
    svm_rank_path = "../Data/RankRes/w=0.7_relevance=9_author/"
    svm_rank(input_file_path, svm_rank_path)