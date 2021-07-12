# -*- coding: utf-8 -*-
import time
import random
from Utils.LoadData import load_context_feature
from Utils.QueryStructure import Query
from Utils.LoadData import load_para
import numpy as np
import random
def calaulate_probability(w, context_feature, rank_position):
    power = max(np.dot(w, context_feature) + 1, 0)
    return pow(1 / rank_position, power)

def simulate_click(data_input_path, context_feature_input_path, score_input_path, para_input_path, click_log_output_path, weight_range, sweep, dimension, click_on_relevent_doc, click_on_irrelevent_doc):
    start_time = time.time()
    random.seed()
    query = load_context_feature(context_feature_input_path)
    index = 0
    with open(data_input_path, "r") as f1, open(score_input_path, "r") as f2:
        for line1, line2 in zip(f1, f2):
            line1 = line1.strip()
            line2 = line2.strip()
            contents = line1.split(' ', 2)
            assert len(contents) == 3
            relevance = int(contents[0])
            qid = int(contents[1].split(':')[1])
            score = float(line2)
            if not query[index].qid == qid:
                index += 1
            assert query[index].qid == qid
            doc_id = len(query[index].doc)
            query[index].doc.append((doc_id, score, relevance))

    w = load_para(para_input_path, dimension, range=weight_range)
    with open(click_log_output_path, "w") as f:
        for sweep_temp in range(sweep):
            for temp in query:
                query_id = temp.qid

                context_feat = temp.context_feature
                docs = sorted(temp.doc, key=lambda x: x[1], reverse=True)
                for rank_position, sr in enumerate(docs, start=1):
                    doc_id, _, relevance = sr
                    observation_probability = calaulate_probability(w, context_feat, rank_position)
                    click = False

                    if random.random() <= observation_probability:
                        if relevance:
                            if random.random() <= click_on_relevent_doc:
                                click = True
                        else:
                            if random.random() <= click_on_irrelevent_doc:
                                click = True
                    if click:
                        write_doc_id = sweep_temp * len(temp.doc) + doc_id
                        content = "%d qid:%d %d\n" % (1, query_id, write_doc_id)
                        f.write(content)
                    else:
                        write_doc_id = sweep_temp * len(temp.doc) + doc_id
                        content = "%d qid:%d %d\n" % (0, query_id, write_doc_id)
                        f.write(content)

    end_time = time.time()
    print("simulate click usage time %f" % (end_time - start_time))

if __name__=="__main__":
    sweep = 1
    dimension = 10
    click_on_relevent_doc = 1.0
    click_on_irrelevent_doc = 0.1
    weight_range = 0.7
    for i in range(2):
        data_input_path = "../Data/SliceData/w=0.7_relevance=9_author/train_slice.txt"
        context_feature_input_path = "../Data/SliceData/w=0.7_relevance=9_author/train_slice_feature.txt"
        score_input_path = "../Data/SliceData/w=0.7_relevance=9_author/train_score_%d.txt" % (i + 1)
        para_input_path = "../Data/SliceData/w=0.7_relevance=9_author/train_slice_para.txt"
        click_log_output_path = "../Data/SliceData/w=0.7_relevance=9_author/train_slice_click_log%d.txt" % (i + 1)
        simulate_click(data_input_path, context_feature_input_path, score_input_path, para_input_path,
                       click_log_output_path, weight_range, sweep, dimension, click_on_relevent_doc, click_on_irrelevent_doc)




