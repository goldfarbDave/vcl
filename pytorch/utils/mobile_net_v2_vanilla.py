"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math
from scipy.stats import truncnorm
import torch
import itertools
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

__all__ = ['mobilenetv2']


# variable initialization functions
def truncated_normal(size, stddev=1, variable = False, mean=0, device="cpu"):
    mu, sigma = mean, stddev
    lower, upper= -2 * sigma, 2 * sigma
    X = truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    X_tensor = torch.Tensor(data = X.rvs(size)).to(device = device)
    X_tensor.requires_grad = variable
    return X_tensor

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, device="cpu", num_classes=1000, width_mult=1., learning_rate=0.006):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.device = device
        self.optimizer =  None
        self.lr = learning_rate

        # first build the entire network using nn components

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        seq_layers = conv_3x3_bn(3, input_channel, 2)
        layers = [seq_layers]

        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)


        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self.kernel_filters = []
        self.linear = None

        self.weights = []

        # next build the entire network using functional components and this will be used for weight creation of our network
        self.create_weights()

    def init_layer(self, layer):
        if(isinstance(layer, nn.Conv2d)):
            self.kernel_filters.append(self.create_kern_weight_bias(layer))
        
        elif(isinstance(layer, nn.BatchNorm2d)):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        
        elif(isinstance(layer, nn.Linear)):
            self.linear = self.create_linear_weight_bias(layer)

        elif(isinstance(layer, nn.ReLU6)):
            pass

        else:
            assert False, 'Unknown layer'

    def create_weights(self):
        # create all convs in features first

        for feat in self.features:

            if(isinstance(feat, nn.Sequential)):
                # iterate through all layers in sequential
                for layer in feat:
                    self.init_layer(layer)

            elif(isinstance(feat, InvertedResidual)):
                # iterate through all layers in InvertedResidual.conv
                for layer in feat.conv:
                    self.init_layer(layer)

        # create all layers in depthwise channel conv 1x1 bn
        for layer in self.conv:
            self.init_layer(layer)

        # create the linear layer for final output
        self.init_layer(self.classifier)

        # Collect the batchnorm weights and bias for adding to weights to optimise: For now we won't be bayesian about those
        src_nm = dict(itertools.chain(self.features.named_modules(), self.conv.named_modules()))
        ks = [k for k,v in src_nm.items() if isinstance(v, torch.nn.BatchNorm2d)]
        for k in ks:
            self.weights.append(src_nm[k].weight)
            self.weights.append(src_nm[k].bias)    

        # Collect conv filter weights and bias for adding to model weights to optimise
        for param in self.kernel_filters:
            self.weights.append(param['weight'])
            if(param['bias'] is not None):
                self.weights.append(param['bias'])

        # Collect linear weight and bias for adding to model weights to optimise
        self.weights.append(self.linear['weight'])
        self.weights.append(self.linear['bias'])

        self.optimizer = optim.Adam(self.weights, lr=self.lr)        

    def create_kern_weight_bias(self, layer):
        filter_dict = {}

        filter_dict['groups'] = layer.groups

        # initialising both H and W of the filter to be the same as kernel_size[0] to ensure square filters
        n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        # filter_dict['weight'] = torch.normal(torch.zeros((layer.out_channels, layer.in_channels//filter_dict['groups'], layer.kernel_size[0], layer.kernel_size[0]))
        # , math.sqrt(2. / n)).to(device=self.device)
        # filter_dict['weight'].requires_grad = True 
        filter_dict['weight'] = truncated_normal([layer.out_channels, layer.in_channels//filter_dict['groups'], layer.kernel_size[0], layer.kernel_size[0]], stddev=math.sqrt(2. / n), variable=True, device=self.device)
        if(layer.bias is not None):
            # filter_dict['bias'] = torch.normal(torch.zeros(layer.out_channels), 0.1).to(device=self.device)
            # filter_dict['bias'].requires_grad = True
            filter_dict['bias'] = truncated_normal([layer.out_channels], stddev=0.1, variable=True, device=self.device)
        else:
            filter_dict['bias'] = None

        filter_dict['padding'] = layer.padding
        filter_dict['dilation'] = layer.dilation
        filter_dict['stride'] = layer.stride

        return filter_dict

    def create_linear_weight_bias(self, layer):

        # initialising weight and bias for linear layer
        # weight = torch.normal(torch.zeros((layer.in_features, layer.out_features)), 
        # 0.01).to(device=self.device)
        # weight.requires_grad = True
        weight = truncated_normal([layer.in_features, layer.out_features], stddev=0.01, variable = True, device=self.device)
        # bias = torch.normal(torch.zeros(layer.out_features), 0.01).to(device=self.device)
        # bias.requires_grad = True
        bias = truncated_normal([layer.out_features], stddev=0.01, variable = True, device=self.device)

        return {'weight': weight, 'bias': bias}

    def run_layer(self, layer, act, count):

        if(isinstance(layer, nn.Conv2d)):
            if(self.kernel_filters[count]['bias'] is None):
                return F.conv2d(input=act.to(device=self.device), weight=self.kernel_filters[count]['weight'].to(device=self.device), stride=self.kernel_filters[count]['stride'], padding=self.kernel_filters[count]['padding'], dilation=self.kernel_filters[count]['dilation'], groups=self.kernel_filters[count]['groups']), count+1
            else:
                return F.conv2d(input=act.to(device=self.device), weight=self.kernel_filters[count]['weight'].to(device=self.device), bias=self.kernel_filters[count]['bias'], stride=self.kernel_filters[count]['stride'], padding=self.kernel_filters[count]['padding'], dilation=self.kernel_filters[count]['dilation'], groups=self.kernel_filters[count]['groups']), count+1

        elif(isinstance(layer, nn.BatchNorm2d)):
            return layer(act.to(device=self.device)), count

        elif(isinstance(layer, nn.Linear)):
            return torch.add(torch.matmul(act.to(device=self.device), self.linear['weight']).to(device=self.device), self.linear['bias']).to(device=self.device), count

        elif(isinstance(layer, nn.ReLU6)):
            return layer(act.to(device=self.device)), count

        else:
            assert False, 'Unknown layer'

    def forward(self, x):

        act = x

        # index to access the specific conv layer in the list of conv weights and biases
        conv_layers_count = 0

        # first run through the features block
        for feat in self.features:
            if(isinstance(feat, nn.Sequential)):
                # iterate through all layers in sequential
                for layer in feat:
                    act, conv_layers_count = self.run_layer(layer, act, conv_layers_count)

            elif(isinstance(feat, InvertedResidual)):
                # iterate through all layers in InvertedResidual.conv
                for layer in feat.conv:
                    act, conv_layers_count = self.run_layer(layer, act, conv_layers_count)         

        # run the depthwise channel conv 1x1 bn block
        for layer in self.conv:
            act, conv_layers_count = self.run_layer(layer, act, conv_layers_count)

        act = self.avgpool(act)
        act = act.view(act.size(0), -1)

        # get final classification layer output
        act, _ = self.run_layer(self.classifier, act, conv_layers_count)
        return act

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_weights_for_bayesian(self):
        return {'KernFilters': self.kernel_filters, 'Linear': self.linear}

    def get_loss(self, x, y, task_idx):
        return -self._logpred(x, y, task_idx)

    def _logpred(self, x, y, task_idx):
        # expected log likelihood of data - first term in eqn 4 of paper
        loss = torch.nn.CrossEntropyLoss()
        pred = self.forward(x)
        log_lik = - loss(pred, y.type(torch.long).to(device=self.device))
        return log_lik

    def train(self, x_train, y_train, task_idx, no_epochs=1000, batch_size=100, display_epoch=5, device="cpu"):
        N = x_train.shape[0]
        self.training_size = N
        if batch_size > N:
            batch_size = N

        costs = []
        # Training cycle
        for epoch in range(no_epochs):
            perm_inds = np.arange(x_train.shape[0])
            np.random.shuffle(perm_inds)
            # randomly shuffle datapoints in batch for each epoch
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]
            #import ipdb; ipdb.set_trace()
            avg_cost = 0.
            # number of batches
            total_batch = int(np.ceil(N * 1.0 / batch_size))
            # Loop over all batches
            for i in range(total_batch):
                start_ind = i*batch_size
                end_ind = np.min([(i+1)*batch_size, N])
                # retrieve current batch data for SGD
                batch_x = torch.Tensor(cur_x_train[start_ind:end_ind, :]).to(device = device)
                batch_y = torch.Tensor(cur_y_train[start_ind:end_ind]).to(device = device)

                ##TODO: check if we need to lock the gradient somewhere
                self.optimizer.zero_grad()
                # hard code reshaping of tensor for cifar 3 channel images
                batch_x = batch_x.view((-1, 3, 32, 32))
                cost = self.get_loss(batch_x, batch_y, task_idx)
                cost.backward()
                self.optimizer.step()

                # Compute average loss
                avg_cost += cost / total_batch
            # Display logs per epoch step
            #if epoch % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
            costs.append(avg_cost.item())
        print("Optimization Finished!")
        return costs

def mobilenetv2_vanilla(device, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(device, **kwargs)

if __name__ == '__main__':
    mobile_net = mobilenetv2_vanilla(num_classes=10, device='cpu')
    print(mobile_net.forward(torch.randn(10, 3, 32, 32)).shape)