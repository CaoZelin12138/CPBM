from torch.utils.data import Dataset
from Utils.LoadData import load_propensity
import numpy as np
import torch
class TrainDataset(Dataset):
    def __init__(self, feature_path, click_path, propsensity_path, use_cuda=False):
        super(TrainDataset, self).__init__()
        self.feature = np.load(feature_path)
        self.click_log, self.not_click_log = np.load(click_path)
        self.propensity = load_propensity(propsensity_path)
        if use_cuda:
            self.feature = torch.tensor(self.feature, dtype=torch.float32).cuda()
            self.click_log = torch.tensor(self.click_log, dtype=torch.float32).cuda()
            self.propensity = torch.tensor(self.propensity, dtype=torch.float32).cuda()
            self.not_click_log = torch.tensor(self.not_click_log, dtype=torch.float32).cuda()
        else:
            self.feature = torch.tensor(self.feature, dtype=torch.float32).cpu()
            self.click_log = torch.tensor(self.click_log, dtype=torch.float32).cpu()
            self.propensity = torch.tensor(self.propensity, dtype=torch.float32).cpu()
            self.not_click_log = torch.tensor(self.not_click_log, dtype=torch.float32).cpu()

    def __getitem__(self, index):
        return (self.feature[index], self.propensity[index], self.click_log[index], self.not_click_log[index])
    def __len__(self):
        return self.feature.size(0)


class TestDataSet(Dataset):
    def __init__(self, feature_path, click_path, propsensity_path, topK_position, use_cuda=False):
        super(TestDataSet, self).__init__()
        self.feature = np.load(feature_path)
        self.propensity = load_propensity(propsensity_path)
        self.how_many_query = self.read_file(click_path)
        self.click_log = np.zeros(shape=(self.how_many_query, topK_position, topK_position))
        self.not_click_log = np.zeros(shape=(self.how_many_query, topK_position, topK_position))
        if use_cuda:
            self.feature = torch.tensor(self.feature, dtype=torch.float32).cuda()
            self.click_log = torch.tensor(self.click_log, dtype=torch.float32).cuda()
            self.propensity = torch.tensor(self.propensity, dtype=torch.float32).cuda()
            self.not_click_log = torch.tensor(self.not_click_log, dtype=torch.float32).cuda()
        else:
            self.feature = torch.tensor(self.feature, dtype=torch.float32).cpu()
            self.click_log = torch.tensor(self.click_log, dtype=torch.float32).cpu()
            self.propensity = torch.tensor(self.propensity, dtype=torch.float32).cpu()
            self.not_click_log = torch.tensor(self.not_click_log, dtype=torch.float32).cpu()

    def __getitem__(self, index):
        return (self.feature[index], self.propensity[index], self.click_log[index], self.not_click_log[index])

    def __len__(self):
        return self.feature.size(0)

    def read_file(self, input_file_path):
        with open(input_file_path) as f:
            for line in f:
                line = line.strip()
                return int(line)