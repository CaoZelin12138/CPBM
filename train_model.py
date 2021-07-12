import torch
import numpy
import sys
sys.path.append("/home/zelin_cao/PythonFile/InterventionHarvesting/")
from TrainModel.SigMMOEMLP_RELP import Net
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from Utils.LoadData import load_propensity
from TrainTestValidLoader import train_data_maker
from TrainTestValidLoader import test_data_maker
import numpy as np
import random
def train(model, train_loader_list, optimizer):
    train_loader = train_loader_list[0]
    model.train()
    for batch_index, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss_error_predict = model(batch[0], batch[1], batch[2], batch[3])


        loss_error_predict[0].backward()
        optimizer.step()

def eval(model, eval_loader_list, mode):
    model.eval()
    scenario_error = []
    input = [temp[0] for temp in eval_loader_list]
    y_label = [temp[1] for temp in eval_loader_list]
    click = [temp[2] for temp in eval_loader_list]
    not_click = [temp[3] for temp in eval_loader_list]

    with torch.no_grad():
        for i in range(len(input)):
            loss_error_predict = model(input[i], y_label[i], click[i], not_click[i])
            print("Scenario %d: %s Error %f" % (i, mode, loss_error_predict[1]))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__=="__main__":
    batch_size = 16
    use_cuda = False
    topK_position = 10
    epoch = 400
    model = Net()
    learning_rate = 2e-4
    gamma = 0.4
    scenario = 1
    model_para_path = "../NetWorkResult/CPBM_model_w=0.6.pkt"
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[30, 40])

    train_feature_path_list = [
       #"../Data/trainData/w=0.1_relevance=9_author/train_slice_feature.npy",
         # "../Data/trainData/w=0.2_relevance=9_author/train_slice_feature.npy",
        # "../Data/trainData/w=0.3_relevance=9_author/train_slice_feature.npy",
        # "../Data/trainData/w=0.4_relevance=9_author/train_slice_feature.npy",
        # "../Data/trainData/w=0.5_relevance=9_author/train_slice_feature.npy",
        "../Data/trainData/w=0.6_relevance=9_author/train_slice_feature.npy",
        # "../Data/trainData/w=0.7_relevance=9_author/train_slice_feature.npy",
        # "../Data/trainData/w=10_relevance=9_author/train_slice_feature.npy"
    ]
    train_click_path_list = [
        #"../Data/trainData/w=0.1_relevance=9_author/train_click.npy",
         # "../Data/trainData/w=0.2_relevance=9_author/train_click.npy",
        # "../Data/trainData/w=0.3_relevance=9_author/train_click.npy",
        # "../Data/trainData/w=0.4_relevance=9_author/train_click.npy",
        # "../Data/trainData/w=0.5_relevance=9_author/train_click.npy",
        "../Data/trainData/w=0.6_relevance=9_author/train_click.npy",
        # "../Data/trainData/w=0.7_relevance=9_author/train_click.npy",
        # "../Data/trainData/w=10_relevance=9_author/train_click.npy"
    ]
    train_true_propensity_path_list = [
        #"../Data/trainData/w=0.1_relevance=9_author/train_slice_propensity.txt",
         # "../Data/trainData/w=0.2_relevance=9_author/train_slice_propensity.txt",
        # "../Data/trainData/w=0.3_relevance=9_author/train_slice_propensity.txt",
        # "../Data/trainData/w=0.4_relevance=9_author/train_slice_propensity.txt",
        # "../Data/trainData/w=0.5_relevance=9_author/train_slice_propensity.txt",
        "../Data/trainData/w=0.6_relevance=9_author/train_slice_propensity.txt",
        # "../Data/trainData/w=0.7_relevance=9_author/train_slice_propensity.txt",
        # "../Data/trainData/w=10_relevance=9_author/train_slice_propensity.txt"
    ]

    test_feature_path_list = [
        #"../Data/trainData/w=0.1_relevance=9_author/test_feature.npy",
         # "../Data/trainData/w=0.2_relevance=9_author/test_feature.npy",
        # "../Data/trainData/w=0.3_relevance=9_author/test_feature.npy",
        # "../Data/trainData/w=0.4_relevance=9_author/test_feature.npy",
        # "../Data/trainData/w=0.5_relevance=9_author/test_feature.npy",
        "../Data/trainData/w=0.6_relevance=9_author/test_feature.npy",
        # "../Data/trainData/w=0.7_relevance=9_author/test_feature.npy",
        # "../Data/trainData/w=10_relevance=9_author/test_feature.npy"
    ]
    test_true_propensity_path_list = [
        #"../Data/trainData/w=0.1_relevance=9_author/test_propensity.txt",
         # "../Data/trainData/w=0.2_relevance=9_author/test_propensity.txt",
        # "../Data/trainData/w=0.3_relevance=9_author/test_propensity.txt",
        # "../Data/trainData/w=0.4_relevance=9_author/test_propensity.txt",
        # "../Data/trainData/w=0.5_relevance=9_author/test_propensity.txt",
        "../Data/trainData/w=0.6_relevance=9_author/test_propensity.txt",
        # "../Data/trainData/w=0.7_relevance=9_author/test_propensity.txt",
        # "../Data/trainData/w=10_relevance=9_author/test_propensity.txt"
    ]

    valid_feature_path_list = [
        #"../Data/trainData/w=0.1_relevance=9_author/valid_feature.npy",
         # "../Data/trainData/w=0.2_relevance=9_author/valid_feature.npy",
        # "../Data/trainData/w=0.3_relevance=9_author/valid_feature.npy",
        # "../Data/trainData/w=0.4_relevance=9_author/valid_feature.npy",
        # "../Data/trainData/w=0.5_relevance=9_author/valid_feature.npy",
        "../Data/trainData/w=0.6_relevance=9_author/valid_feature.npy",
        # "../Data/trainData/w=0.7_relevance=9_author/valid_feature.npy",
        # "../Data/trainData/w=10_relevance=9_author/valid_feature.npy"
    ]
    valid_true_propensity_path_list = [
        #"../Data/trainData/w=0.1_relevance=9_author/valid_propensity.txt",
         # "../Data/trainData/w=0.2_relevance=9_author/valid_propensity.txt",
        # "../Data/trainData/w=0.3_relevance=9_author/valid_propensity.txt",
        # "../Data/trainData/w=0.4_relevance=9_author/valid_propensity.txt",
        # "../Data/trainData/w=0.5_relevance=9_author/valid_propensity.txt",
        "../Data/trainData/w=0.6_relevance=9_author/valid_propensity.txt",
        # "../Data/trainData/w=0.7_relevance=9_author/valid_propensity.txt",
        # "../Data/trainData/w=10_relevance=9_author/valid_propensity.txt"
    ]


   
    # train_data = TrainDataset(train_feature_path_list, train_click_path_list, train_true_propensity_path_list)
    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader_list = train_data_maker(train_feature_path_list, train_click_path_list,
                                         train_true_propensity_path_list,
                                         batch_size=batch_size, how_many_task=scenario, use_cuda=use_cuda)

    test_loader_list = test_data_maker(test_feature_path_list, test_true_propensity_path_list,
                                       topK_position=topK_position, how_many_task=scenario, use_cuda=use_cuda)
    #
    valid_loader_list = test_data_maker(valid_feature_path_list, valid_true_propensity_path_list,
                                        topK_position=topK_position, how_many_task=scenario, use_cuda=use_cuda)
    #
    eval_train_loader_list = test_data_maker(train_feature_path_list, train_true_propensity_path_list,
                                             topK_position=topK_position, how_many_task=scenario, use_cuda=use_cuda)

    predict_propensity = None
    for i in range(epoch):
        train(model, train_loader_list, optimizer)

        print("Epoch %d: " % i)

        eval(model, valid_loader_list, "valid")
        eval(model, test_loader_list, "test")
        scheduler.step()
        print()
    # np.savetxt(model_para_path, predict_propensity, fmt='%.18f')
    # torch.save(model, model_para_path)

