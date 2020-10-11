
from logger import coil_logger
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F


class Gaussian_FC(nn.Module):

    def __init__(self, params=None, module_name='Default'
                 ):
        # TODO: Make an auto naming function for this.

        super(Gaussian_FC, self).__init__()


        """" ---------------------- FC ----------------------- """
        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'neurons' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'dropouts' not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if 'end_layer' not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params['dropouts']) != len(params['neurons'])-1:
            raise ValueError("Dropouts should be from the len of kernels minus 1")

        self.log_vars = []
        self.means = []


        for i in range(0, len(params['neurons']) -1):

            fc_means = nn.Linear(params['neurons'][i], params['neurons'][i+1])
            dropout_means = nn.Dropout2d(p=params['dropouts'][i])
            relu_means = nn.ReLU(inplace=True)

            fc_log_vars = nn.Linear(params['neurons'][i], params['neurons'][i + 1])
            dropout_log_vars = nn.Dropout2d(p=params['dropouts'][i])
            relu_log_vars = nn.ReLU(inplace=True)

            if i == len(params['neurons'])-2 and params['end_layer']:
                self.means.append(nn.Sequential(*[fc_means, dropout_means]))
                self.log_vars.append(nn.Sequential(*[fc_log_vars, dropout_log_vars]))
            else:
                self.means.append(nn.Sequential(*[fc_means, dropout_means, relu_means]))
                self.log_vars.append(nn.Sequential(*[fc_log_vars, dropout_log_vars, relu_log_vars]))


        self.log_vars = nn.Sequential(*self.log_vars)
        self.means = nn.Sequential(*self.means)


    def forward(self, x):
        # if X is a tuple, just return the other elements, the idea is to re pass
        # the intermediate layers for future attention plotting
        if type(x) is tuple:
            return self.layers(x[0]), x[1]
        else:
            means = self.means(x)
            log_vars = self.log_vars(x)

            return means, log_vars


class FC(nn.Module):

    def __init__(self, params=None, module_name='Default'
                 ):
        # TODO: Make an auto naming function for this.

        super(FC, self).__init__()


        """" ---------------------- FC ----------------------- """
        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'neurons' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'dropouts' not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if 'end_layer' not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params['dropouts']) != len(params['neurons'])-1:
            raise ValueError("Dropouts should be from the len of kernels minus 1")


        self.layers = []


        for i in range(0, len(params['neurons']) -1):

            fc = nn.Linear(params['neurons'][i], params['neurons'][i+1])
            dropout = nn.Dropout2d(p=params['dropouts'][i])
            relu = nn.ReLU(inplace=True)

            if i == len(params['neurons'])-2 and params['end_layer']:
                self.layers.append(nn.Sequential(*[fc, dropout]))
            else:
                self.layers.append(nn.Sequential(*[fc, dropout, relu]))


        self.layers = nn.Sequential(*self.layers)



    def forward(self, x):
        # if X is a tuple, just return the other elements, the idea is to re pass
        # the intermediate layers for future attention plotting
        if type(x) is tuple:
            return self.layers(x[0]), x[1]
        else:
            return self.layers(x)