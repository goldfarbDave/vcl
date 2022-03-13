"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

from os import terminal_size
import torch.nn as nn
import math
from scipy.stats import truncnorm
import torch
import pickle
import itertools
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

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
    def __init__(self, device="cpu", num_classes=1000, width_mult=1., prev_means=None, learning_rate=0.006):
        super(MobileNetV2, self).__init__()
        self.no_samps = 1
        print ("Setting no samples for MC sampling as "+str(self.no_samps))
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
        self.prior_kernel_filters = []
        self.linears = []
        self.prior_linears = []

        self.weights = []

        # next build the entire network using functional components and this will be used for weight creation of our network
        self.create_weights(prev_means=prev_means)
        # creat priors on conv and linear layers
        self.create_priors()

    def create_head(self):
        # create new task head linear
        self.init_layer(self.classifier, None, None, 0)

        # create new prior for task head
        self.init_prior_layer(self.classifier)

        # add new heads to weights 
        self.weights.append(self.linears[-1]['weight_mean'])
        self.weights.append(self.linears[-1]['bias_mean'])
        self.weights.append(self.linears[-1]['weight_logsigma'])
        self.weights.append(self.linears[-1]['bias_logsigma'])
        self.optimizer = optim.Adam(self.weights, lr=self.lr)

    def init_layer(self, layer, prev_conv_means, prev_linear_mean, count):
        if(isinstance(layer, nn.Conv2d)):
            if(prev_conv_means is not None):
                self.kernel_filters.append(self.create_kern_weight_bias(layer, prev_conv_means[count]['weight'], prev_conv_means[count]['bias']))
            else:
                self.kernel_filters.append(self.create_kern_weight_bias(layer, None, None))
            count += 1

        elif(isinstance(layer, nn.BatchNorm2d)):
            # treating batch norm params as point estimates
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        
        elif(isinstance(layer, nn.Linear)):
            if(prev_linear_mean is not None):
                self.linears.append(self.create_linear_weight_bias(layer, prev_linear_mean['weight'], prev_linear_mean['bias']))
            else:
                self.linears.append(self.create_linear_weight_bias(layer, None, None))

        elif(isinstance(layer, nn.ReLU6)):
            pass

        else:
            assert False, 'Unknown layer'

        return count

    def init_prior_layer(self, layer):
        if(isinstance(layer, nn.Conv2d)):
            self.prior_kernel_filters.append(self.create_prior_kern_weight_bias(layer))
        
        elif(isinstance(layer, nn.BatchNorm2d)):
            pass
        
        elif(isinstance(layer, nn.Linear)):
            self.prior_linears.append(self.create_prior_linear_weight_bias(layer))
            assert len(self.prior_linears) == len(self.linears), 'Wrong lengths of priors and actual weights'

        elif(isinstance(layer, nn.ReLU6)):
            pass

        else:
            assert False, 'Unknown layer'            

    def create_priors(self):
        # create all prior convs in features first

        for feat in self.features:

            if(isinstance(feat, nn.Sequential)):
                # iterate through all layers in sequential
                for layer in feat:
                    self.init_prior_layer(layer)

            elif(isinstance(feat, InvertedResidual)):
                # iterate through all layers in InvertedResidual.conv
                for layer in feat.conv:
                    self.init_prior_layer(layer)

        # create all layers in depthwise channel conv 1x1 bn
        for layer in self.conv:
            self.init_prior_layer(layer)

        # create the linear layer for final output
        self.init_prior_layer(self.classifier)

    def create_weights(self, prev_means):
        # prev means for the conv kernels
        prev_conv_means = prev_means['KernFilters']
        prev_linear_mean = prev_means['Linear']

        conv_layer_count = 0

        # create all convs in features first

        for feat in self.features:

            if(isinstance(feat, nn.Sequential)):
                # iterate through all layers in sequential
                for layer in feat:
                    conv_layer_count = self.init_layer(layer, prev_conv_means, prev_linear_mean, conv_layer_count)

            elif(isinstance(feat, InvertedResidual)):
                # iterate through all layers in InvertedResidual.conv
                for layer in feat.conv:
                    conv_layer_count = self.init_layer(layer, prev_conv_means, prev_linear_mean, conv_layer_count)

        # create all layers in depthwise channel conv 1x1 bn
        for layer in self.conv:
            conv_layer_count = self.init_layer(layer, prev_conv_means, prev_linear_mean, conv_layer_count)

        # create the linear layer for final output
        self.init_layer(self.classifier, prev_conv_means, prev_linear_mean, conv_layer_count)

        # if affine=False for batchnorm turn off weight tracking 
        # Collect the batchnorm weights and bias for adding to weights to optimise: For now we won't be bayesian about those
        src_nm = dict(itertools.chain(self.features.named_modules(), self.conv.named_modules()))
        ks = [k for k,v in src_nm.items() if isinstance(v, torch.nn.BatchNorm2d)]
        for k in ks:
            self.weights.append(src_nm[k].weight)
            self.weights.append(src_nm[k].bias)    

        # Collect conv filter weights and bias for adding to model weights to optimise
        for param in self.kernel_filters:
            self.weights.append(param['weight_mean'])
            self.weights.append(param['weight_logsigma'])
            if(param['bias_mean'] is not None):
                self.weights.append(param['bias_mean'])
                self.weights.append(param['bias_logsigma'])

        # Collect linear weight and bias for adding to model weights to optimise
        # Only add first task head to weights when initialising weights from prev means the first time
        self.weights.append(self.linears[0]['weight_mean'])
        self.weights.append(self.linears[0]['bias_mean'])
        self.weights.append(self.linears[0]['weight_logsigma'])
        self.weights.append(self.linears[0]['bias_logsigma'])
        self.optimizer = optim.Adam(self.weights, lr=self.lr)

    def create_kern_weight_bias(self, layer, prev_mean_weight, prev_mean_bias):
        filter_dict = {}

        filter_dict['groups'] = layer.groups

        # initialising both H and W of the filter to be the same as kernel_size[0] to ensure square filters
        n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        # filter_dict['weight'] = torch.normal(torch.zeros((layer.out_channels, layer.in_channels//filter_dict['groups'], layer.kernel_size[0], layer.kernel_size[0]))
        # , math.sqrt(2. / n)).to(device=self.device)
        # filter_dict['weight'].requires_grad = True 
        if(prev_mean_weight is None):
            filter_dict['weight_mean'] = truncated_normal([layer.out_channels, layer.in_channels//filter_dict['groups'], layer.kernel_size[0], layer.kernel_size[0]], stddev=math.sqrt(2. / n), variable=True, device=self.device)
        else:
            filter_dict['weight_mean'] = prev_mean_weight.detach().data.to(device=self.device)

        filter_dict['weight_logsigma'] = -6.0*torch.ones_like(filter_dict['weight_mean']).to(device=self.device)

        if(layer.bias is not None):
            # filter_dict['bias'] = torch.normal(torch.zeros(layer.out_channels), 0.1).to(device=self.device)
            # filter_dict['bias'].requires_grad = True
            if(prev_mean_bias is None):
                filter_dict['bias_mean'] = truncated_normal([layer.out_channels], stddev=0.1, variable=True, device=self.device)
            else:
                filter_dict['bias_mean'] = prev_mean_bias.detach().data.to(device=self.device)
        else:
            filter_dict['bias_mean'] = None

        if(filter_dict['bias_mean'] is not None):
            filter_dict['bias_logsigma'] = -6.0*torch.ones_like(filter_dict['bias_mean']).to(device=self.device)
        else:
            filter_dict['bias_logsigma'] = None

        filter_dict['padding'] = layer.padding
        filter_dict['dilation'] = layer.dilation
        filter_dict['stride'] = layer.stride

        return filter_dict

    def create_prior_kern_weight_bias(self, layer):
        filter_dict = {}

        filter_dict['weight_mean'] = torch.zeros((layer.out_channels, layer.in_channels//layer.groups, layer.kernel_size[0], layer.kernel_size[0])).to(device=self.device)
        filter_dict['weight_logsigma'] = 0.01*torch.ones_like(filter_dict['weight_mean']).to(device=self.device)

        if(layer.bias is not None):
            filter_dict['bias_mean'] = torch.zeros((layer.out_channels,)).to(device=self.device)
            filter_dict['bias_logsigma'] = 0.01*torch.ones_like(filter_dict['bias_mean']).to(device=self.device)

        return filter_dict

    def create_linear_weight_bias(self, layer, prev_mean_weight, prev_mean_bias):

        # initialising weight and bias for linear layer
        # weight = torch.normal(torch.zeros((layer.in_features, layer.out_features)), 
        # 0.01).to(device=self.device)
        # weight.requires_grad = True
        if(prev_mean_weight is None):
            weight_mean = truncated_normal([layer.in_features, layer.out_features], stddev=0.01, variable = True, device=self.device)
        else:
            weight_mean = prev_mean_weight.detach().data.to(device=self.device)

        # bias = torch.normal(torch.zeros(layer.out_features), 0.01).to(device=self.device)
        # bias.requires_grad = True
        if(prev_mean_bias is None):
            bias_mean = truncated_normal([layer.out_features], stddev=0.01, variable = True, device=self.device)
        else:
            bias_mean = prev_mean_bias.detach().data.to(device=self.device)

        weight_logsigma = -6.0*torch.ones_like(weight_mean).to(device=self.device)
        bias_logsigma = -6.0*torch.ones_like(bias_mean).to(device=self.device)

        return {'weight_mean': weight_mean, 'bias_mean': bias_mean,
        'weight_logsigma': weight_logsigma, 'bias_logsigma': bias_logsigma}

    def create_prior_linear_weight_bias(self, layer):
        
        weight_mean = torch.zeros((layer.in_features, layer.out_features)).to(device=self.device)
        bias_mean = torch.zeros((layer.out_features,)).to(device=self.device)

        weight_logsigma = torch.ones_like(weight_mean).to(device=self.device)
        bias_logsigma = torch.ones_like(bias_mean).to(device=self.device)

        return {'weight_mean': weight_mean, 'bias_mean': bias_mean,
        'weight_logsigma': weight_logsigma, 'bias_logsigma': bias_logsigma}

    def run_layer(self, layer, act, count, task_idx):

        if(isinstance(layer, nn.Conv2d)):
            kw_m = self.kernel_filters[count]['weight_mean']
            kb_m = self.kernel_filters[count]['bias_mean']
            kw_v = self.kernel_filters[count]['weight_logsigma']
            kb_v = self.kernel_filters[count]['bias_logsigma']
            res = []
            for i in range(self.no_samps):
                get_eps = lambda ten: torch.normal(torch.zeros(ten.shape)).to(device=self.device) if ten is not None else None
                eps_w = get_eps(kw_m)
                eps_b = get_eps(kb_m)
                weights = torch.add(eps_w *torch.exp(0.5*kw_v), kw_m)
                bias = torch.add(eps_b *torch.exp(0.5*kb_v), kb_m) if eps_b else None
                res.append(F.conv2d(input=act[i].to(device=self.device), 
                    weight=weights, 
                    bias=bias, 
                    stride=self.kernel_filters[count]['stride'], 
                    padding=self.kernel_filters[count]['padding'], 
                    dilation=self.kernel_filters[count]['dilation'], 
                    groups=self.kernel_filters[count]['groups']).unsqueeze(0))
            res_tensor = torch.cat(res)
            return res_tensor.to(device=self.device), count+1

        elif(isinstance(layer, nn.BatchNorm2d)):
            res = []
            for i in range(self.no_samps):
                res.append(layer(act[i].to(device=self.device)).unsqueeze(0))
            res_tensor = torch.cat(res)
            return res_tensor.to(device=self.device), count

        elif(isinstance(layer, nn.Linear)):
            K = self.no_samps
            eps_w = torch.normal(torch.zeros((K, layer.in_features, layer.out_features)), torch.ones((K, layer.in_features, layer.out_features))).to(device = self.device)
            eps_b = torch.normal(torch.zeros((K, 1, layer.out_features)), torch.ones((K, 1, layer.out_features))).to(device = self.device)
            # random sample weight from distribution -- reparameterisation trick
            weights = torch.add(eps_w * torch.exp(0.5*self.linears[task_idx]['weight_logsigma']), self.linears[task_idx]['weight_mean'])
            biases = torch.add(eps_b * torch.exp(0.5*self.linears[task_idx]['bias_logsigma']), self.linears[task_idx]['bias_mean'])
            pre = torch.add(torch.einsum('mni,mio->mno', act, weights), biases)
            return pre, count

        elif(isinstance(layer, nn.ReLU6)):
            return layer(act), count

        else:
            assert False, 'Unknown layer'

    def forward(self, x, task_idx):
        act = x
        K = self.no_samps
        # repeat acts for supporting MC sampling over weights
        act = torch.unsqueeze(act, 0).repeat([K, 1, 1, 1, 1])
        # index to access the specific conv layer in the list of conv weights and biases
        conv_layers_count = 0

        # first run through the features block
        for feat in self.features:
            if(isinstance(feat, nn.Sequential)):
                # iterate through all layers in sequential
                for layer in feat:
                    act, conv_layers_count = self.run_layer(layer, act, conv_layers_count, task_idx)

            elif(isinstance(feat, InvertedResidual)):
                # iterate through all layers in InvertedResidual.conv
                for layer in feat.conv:
                    act, conv_layers_count = self.run_layer(layer, act, conv_layers_count, task_idx)         

        # run the depthwise channel conv 1x1 bn block
        for layer in self.conv:
            act, conv_layers_count = self.run_layer(layer, act, conv_layers_count, task_idx)

        act = self.avgpool(act)
        # Kxbsx...
        act = act.view(act.size(0), act.size(1), -1)

        # get final classification layer output
        act, _ = self.run_layer(self.classifier, act, conv_layers_count, task_idx)
        return act

    def get_loss(self, x, y, task_idx):
        # scale both terms to similar values 500 magic no
        kl_term = torch.div(self._KL_term(), self.training_size*500)
        lik_term = self._logpred(x, y, task_idx)
        return kl_term - lik_term

    def _KL_term(self):
        kl = 0

        # implement kl divergence between 2 gaussians:
        # log(sig_2/sig_1) + (sig_1^2 + (mu_1 - mu_2)^2)/(2.sig_2^2) - 0.5

        # conv
        for i, (kernel_weight, prior_kernel_weight) in enumerate(zip(self.kernel_filters, self.prior_kernel_filters)):
            # weight from current posterior - q_t(theta)
            m, v = kernel_weight['weight_mean'], kernel_weight['weight_logsigma']
            # weight from prev posteriors - q_t-1(theta)
            m0, v0 = prior_kernel_weight['weight_mean'], prior_kernel_weight['weight_logsigma']

            const_term = -0.5 * np.prod(m.shape)
            log_std_diff = 0.5 * torch.sum(v0 - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / torch.exp(v0))
            kl += const_term + log_std_diff + mu_diff_term

            if(kernel_weight['bias_mean'] is not None):
                # weight from current posterior - q_t(theta)
                m, v = kernel_weight['bias_mean'], kernel_weight['bias_logsigma']
                # weight from prev posteriors - q_t-1(theta)
                m0, v0 = prior_kernel_weight['bias_mean'], prior_kernel_weight['bias_logsigma']

                const_term = -0.5 * np.prod(m.shape)
                log_std_diff = 0.5 * torch.sum(v0 - v)
                mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / torch.exp(v0))
                kl += const_term + log_std_diff + mu_diff_term

        # linear
        for i, (linear_weight, prior_linear_weight) in enumerate(zip(self.linears, self.prior_linears)):
            # weight from current posterior - q_t(theta)
            m, v = linear_weight['weight_mean'], linear_weight['weight_logsigma']
            # weight from prev posteriors - q_t-1(theta)
            m0, v0 = prior_linear_weight['weight_mean'], prior_linear_weight['weight_logsigma']

            const_term = -0.5 * np.prod(m.shape)
            log_std_diff = 0.5 * torch.sum(v0 - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / torch.exp(v0))
            kl += const_term + log_std_diff + mu_diff_term

            # bias from current posterior - q_t(theta)
            m, v = linear_weight['bias_mean'], linear_weight['bias_logsigma']
            # bias from prev posteriors - q_t-1(theta)
            m0, v0 = prior_linear_weight['bias_mean'], prior_linear_weight['bias_logsigma']

            const_term = -0.5 * np.prod(m.shape)
            log_std_diff = 0.5 * torch.sum(v0 - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / torch.exp(v0))
            kl += const_term + log_std_diff + mu_diff_term

        return kl

    def _logpred(self, inputs, targets, task_idx):
        loss = torch.nn.CrossEntropyLoss()
        #Kxbsx...
        pred = self.forward(inputs, task_idx)
        # fixing output dims to 2 as all tasks have same num classes
        pred = pred.view(-1, 2)
        targets = targets.repeat([self.no_samps, 1]).view(-1).to(device=self.device)
        log_liks = -loss(pred, targets.type(torch.long))
        log_lik = log_liks.mean()
        return log_lik

    def prediction_prob(self, x_test, task_idx):
        # hard code reshaping of tensor for cifar 3 channel images
        x_test = x_test.view((-1, 3, 32, 32))
        prob = F.softmax(self.forward(x_test, task_idx), dim=-1)
        return prob

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

    def update_prior(self):
        for i, kernel_weight in enumerate(self.kernel_filters):
            self.prior_kernel_filters[i]['weight_mean'].data.copy_(kernel_weight['weight_mean'].clone().detach().data)
            self.prior_kernel_filters[i]['weight_logsigma'].data.copy_(kernel_weight['weight_mean'].clone().detach().data)
            if(kernel_weight['bias_mean'] is not None):
                self.prior_kernel_filters[i]['bias_mean'].data.copy_(kernel_weight['bias_mean'].clone().detach().data)
                self.prior_kernel_filters[i]['bias_logsigma'].data.copy_(kernel_weight['bias_mean'].clone().detach().data)

        for i, linear_weight in enumerate(self.linears):
            self.prior_linears[i]['weight_mean'].data.copy_(linear_weight['weight_mean'].clone().detach().data)
            self.prior_linears[i]['weight_logsigma'].data.copy_(linear_weight['weight_mean'].clone().detach().data)
            self.prior_linears[i]['bias_mean'].data.copy_(linear_weight['bias_mean'].clone().detach().data)
            self.prior_linears[i]['bias_logsigma'].data.copy_(linear_weight['bias_mean'].clone().detach().data)

    def save_weights(self):
        with open('save_weight.pickle', 'wb') as f:
            pickle.dump({'kernel_filters': self.kernel_filters, 
            'linears': self.linears, 
            'prior_kernel_filters': self.prior_kernel_filters, 
            'prior_linears': self.prior_linears}, f)

    def load_weights(self):
        with open('save_weight.pickle', 'rb') as f:
            weight_dict = pickle.load(f)
        self.kernel_filters = weight_dict['kernel_filters']
        self.prior_kernel_filters = weight_dict['prior_kernel_filters']
        self.linears = weight_dict['linears']
        self.prior_linears = weight_dict['prior_linears']

        self.weights = []

        # if affine=False for batchnorm turn off weight tracking 
        # Collect the batchnorm weights and bias for adding to weights to optimise: For now we won't be bayesian about those
        src_nm = dict(itertools.chain(self.features.named_modules(), self.conv.named_modules()))
        ks = [k for k,v in src_nm.items() if isinstance(v, torch.nn.BatchNorm2d)]
        for k in ks:
            self.weights.append(src_nm[k].weight)
            self.weights.append(src_nm[k].bias)    

        # Collect conv filter weights and bias for adding to model weights to optimise
        for param in self.kernel_filters:
            self.weights.append(param['weight_mean'])
            self.weights.append(param['weight_logsigma'])
            if(param['bias_mean'] is not None):
                self.weights.append(param['bias_mean'])
                self.weights.append(param['bias_logsigma'])

        # Collect linear weight and bias for adding to model weights to optimise
        for linear_ in self.linears:
            self.weights.append(linear_['weight_mean'])
            self.weights.append(linear_['bias_mean'])
            self.weights.append(linear_['weight_logsigma'])
            self.weights.append(linear_['bias_logsigma'])
        self.optimizer = optim.Adam(self.weights, lr=self.lr)

def mobilenetv2_bayesian(device, prev_means, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(device=device, prev_means=prev_means, **kwargs)

if __name__ == '__main__':
    mobile_net = mobilenetv2_bayesian(num_classes=10, device='cpu')
    print(mobile_net.forward(torch.randn(10, 3, 32, 32)).shape)