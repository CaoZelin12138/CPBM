import torchvision
import torch
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
from torch.nn import init
class Net(Module):
    def __init__(self, deminsion=10, topK_Position=10, propensity_hidden_layer=14, relevance_hidden_layer=12):
        super(Net, self).__init__()
        self.topK_position = topK_Position

        self.propensity_layer1 = torch.nn.Linear(deminsion, propensity_hidden_layer, bias=True)
        self.propensity_layer2 = torch.nn.Linear(propensity_hidden_layer, propensity_hidden_layer, bias=True)
        self.propensity_layer3 = torch.nn.Linear(propensity_hidden_layer, topK_Position, bias=True)
        # self.propensity_layer4 = torch.nn.Linear(propensity_hidden_layer, topK_Position, bias=True)

        self.relevance_layer1 = torch.nn.Linear(deminsion, relevance_hidden_layer, bias=True)
        self.relevance_layer2 = torch.nn.Linear(relevance_hidden_layer, relevance_hidden_layer, bias=True)
        self.relevance_layer3 = torch.nn.Linear(relevance_hidden_layer, relevance_hidden_layer, bias=True)
        self.relevance_layer4 = torch.nn.Linear(relevance_hidden_layer, topK_Position * topK_Position)

    def forward(self, x, y, click, not_click):
        propensity_layer1 = self.propensity_layer1(x)
        propensity_layer1 = torch.nn.functional.leaky_relu(propensity_layer1)

        propensity_layer2 = self.propensity_layer2(propensity_layer1)
        propensity_layer2 = torch.nn.functional.leaky_relu(propensity_layer2)

        propensity_layer3 = self.propensity_layer3(propensity_layer2)
        propensity_layer3 = torch.nn.functional.leaky_relu(propensity_layer3)

        propensity_layer4 = self.propensity_layer4(propensity_layer3)
        propensity_layer4 = torch.sigmoid(propensity_layer4)

        relevance_layer1 = self.relevance_layer1(x)
        relevance_layer1 = torch.nn.functional.leaky_relu(relevance_layer1)

        relevance_layer2 = self.relevance_layer2(relevance_layer1)
        relevance_layer2 = torch.nn.functional.leaky_relu(relevance_layer2)

        relevance_layer3 = self.relevance_layer3(relevance_layer2)
        relevance_layer3 = torch.nn.functional.leaky_relu(relevance_layer3)

        relevance_layer4 = self.relevance_layer4(relevance_layer3)
        relevance_layer4 = torch.sigmoid(relevance_layer4)

        relevance_layer4 = torch.reshape(relevance_layer4, (-1, self.topK_position, self.topK_position))

        symmetric_relevance_layer4 = torch.transpose(relevance_layer4, 1, 2)

        symmetric_relevance = torch.div(torch.add(relevance_layer4, symmetric_relevance_layer4), 2.0)
        #
        # relevance_layer3 = torch.reshape(relevance_layer3, (-1, self.topK_position, self.topK_position))
        #
        # symmetric_relevance_layer3 = torch.transpose(relevance_layer3, 1, 2)
        #
        # symmetric_relevance = torch.div(torch.add(relevance_layer3, symmetric_relevance_layer3), 2.0)


        examination_relevance = torch.mul(torch.reshape(propensity_layer4, (-1, self.topK_position, 1)), symmetric_relevance)
        examination_relevance = torch.clamp(examination_relevance, min=1e-6, max=1 - 1e-6)

        click_loss = torch.mul(torch.log(examination_relevance), click)
        not_click_loss = torch.mul(torch.log(1 - examination_relevance), not_click)
        loss = -torch.sum(torch.add(click_loss, not_click_loss))


        predict_prop = torch.div(propensity_layer4, torch.reshape(propensity_layer4[:, 0], (-1, 1)))

        # predict_prop = torch.div(propensity_layer, torch.reshape(propensity_layer3[:, 0], (-1, 1)))

        err = torch.sum(torch.abs((y - predict_prop) / y))
        return (loss, err, predict_prop)
        # return loss, err, predict_prop, symmetric_relevance, propensity_layer4, examination_relevance





