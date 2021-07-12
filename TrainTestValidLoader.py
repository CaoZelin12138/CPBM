from torch.utils.data import DataLoader
from TrainModel.data_wrapper import TrainDataset
from TrainModel.data_wrapper import TestDataSet
from Utils.LoadData import load_propensity
import numpy as np
import torch
def train_data_maker(feature_path_list, click_path_list, true_propensity_path_list, batch_size, how_many_task, use_cuda):

    data_loader_list = []
    for i in range(how_many_task):
        temp = DataLoader(TrainDataset(feature_path_list[i],
                                click_path_list[i],
                                true_propensity_path_list[i], use_cuda),
                   batch_size=batch_size, shuffle=True, num_workers=4)
        data_loader_list.append(temp)
    return data_loader_list

def test_data_maker(feature_path_list, true_propensity_path_list, topK_position, how_many_task, use_cuda):

    data_loader_list = []
    for i in range(how_many_task):
        a = feature_path_list[i]
        b = true_propensity_path_list[i]
        context_feature = np.load(feature_path_list[i])
        how_many_query = context_feature.shape[0]
        click = np.zeros(shape=(how_many_query, topK_position, topK_position))
        non_click = np.zeros(shape=(how_many_query, topK_position, topK_position))
        y_label = load_propensity(true_propensity_path_list[i])
        if use_cuda:
            context_feature = torch.tensor(context_feature, dtype=torch.float32).cuda()
            click = torch.tensor(click, dtype=torch.float32).cuda()
            non_click = torch.tensor(non_click, dtype=torch.float32).cuda()
            y_label = torch.tensor(y_label, dtype=torch.float32).cuda()
        else:
            context_feature = torch.tensor(context_feature, dtype=torch.float32)
            click = torch.tensor(click, dtype=torch.float32)
            non_click = torch.tensor(non_click, dtype=torch.float32)
            y_label = torch.tensor(y_label, dtype=torch.float32)
        data_loader_list.append((context_feature, y_label, click, non_click))
    return data_loader_list


if __name__ == "__main__":

    pass