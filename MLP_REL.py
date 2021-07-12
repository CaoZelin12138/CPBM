import torchvision
import torch
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
from torch.nn import init
class Net(Module):
    def __init__(self, deminsion=10, topK_Position=10, propensity_hidden_layer=12, relevance_hidden_layer=11):
        super(Net, self).__init__()
        self.topK_position = topK_Position

        self.propensity_layer1 = torch.nn.Linear(deminsion, propensity_hidden_layer, bias=True)
        self.propensity_layer2 = torch.nn.Linear(propensity_hidden_layer, topK_Position, bias=True)

        self.relevance_layer1 = torch.nn.Linear(deminsion, relevance_hidden_layer, bias=True)
        self.relevance_layer2 = torch.nn.Linear(relevance_hidden_layer, topK_Position * topK_Position)
        self.init_paramater()
    def init_paramater(self):
        init.kaiming_normal_(self.propensity_layer1.weight)
        init.kaiming_normal_(self.propensity_layer2.weight)
        init.kaiming_normal_(self.relevance_layer1.weight)
        init.kaiming_normal_(self.relevance_layer2.weight)
    def forward(self, x, y, click, not_click):
        propensity_x = self.propensity_layer1(x)
        propensity_layer1 = torch.sigmoid(propensity_x)

        propensity_layer2 = self.propensity_layer2(propensity_layer1)
        propensity_layer2 = torch.sigmoid(propensity_layer2)

        relevance_x = self.relevance_layer1(x)
        relevance_layer1 = torch.sigmoid(relevance_x)

        relevance_layer2 = self.relevance_layer2(relevance_layer1)
        relevance_layer2 = torch.sigmoid(relevance_layer2)
        relevance_layer2 = torch.reshape(relevance_layer2, (-1, self.topK_position, self.topK_position))

        symmetric_relevance_layer2 = torch.transpose(relevance_layer2, 1, 2)

        symmetric_relevance = torch.div(torch.add(relevance_layer2, symmetric_relevance_layer2), 2.0)
        a = torch.reshape(propensity_layer2, (-1, self.topK_position, 1))

        examination_relevance = torch.mul(torch.reshape(propensity_layer2, (-1, self.topK_position, 1)), symmetric_relevance)
        examination_relevance = torch.clamp(examination_relevance, min=1e-6, max=1 - 1e-6)

        click_loss = torch.mul(torch.log(examination_relevance), click)
        not_click_loss = torch.mul(torch.log(1 - examination_relevance), not_click)
        loss = -torch.sum(torch.add(click_loss, not_click_loss))

        b = torch.reshape(propensity_layer2[:, 0], (-1, 1))
        predict_prop = torch.div(propensity_layer2, torch.reshape(propensity_layer2[:, 0], (-1, 1)))
        # d = torch.abs((y - norm_p_) / y)

        err = torch.sum(torch.abs((y - predict_prop) / y))
        return loss, err, predict_prop





