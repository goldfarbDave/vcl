import numpy as np
from utils.DataGenerator import SplitCifarGenerator, SplitMnistGenerator
import torch
from utils.multihead_models import Vanilla_CNN, Vanilla_NN, MFVI_NN, MFVI_CNN, Vanilla_IID_CNN
import utils.coreset as coreset
import utils.test as test
from utils.vcl import run_vcl_cnn, run_vcl
import torchvision.models as models
from utils.mobile_net_v2 import mobilenetv2
from utils.mobile_net_v2_bayesian import mobilenetv2_bayesian
import torch.optim as optim
import torch.nn as nn

data_gen = SplitCifarGenerator()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

in_dim_cnn, out_dim_cnn = 1600, 10
in_dim_fc, out_dim_fc = data_gen.get_dims()
hidden_size_cnn = [512,128]
hidden_size_fc = [256, 256]

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.net = torch.nn.Sequential(*list(mobilenetv2_bayesian().children())[:-1])      
        self.fc = torch.nn.Linear(1280, 10)  

    def forward(self, x):
        return self.fc(self.net(x).view(-1, 1280))

in_dim, out_dim = data_gen.get_dims()
x_testsets, y_testsets = [], []
x_trainsets, y_trainsets = [], []
gans = []
all_acc = np.array([])

mobilenet = MobileNet()
mobilenet.to(device)

x_train, y_train, x_test, y_test = data_gen.get_non_split()
head = 0
bsize = 256
no_epochs = 100
optimizer = optim.Adam(list(mobilenet.parameters()), lr=0.006)
loss = torch.nn.CrossEntropyLoss()

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
    N = x_train.shape[0]
    total_batch = int(np.ceil(N * 1.0 / bsize))
    # Loop over all batches
    for i in range(total_batch):
        start_ind = i*bsize
        end_ind = np.min([(i+1)*bsize, N])
        # retrieve current batch data for SGD
        batch_x = torch.Tensor(cur_x_train[start_ind:end_ind, :]).to(device = device)
        batch_y = torch.Tensor(cur_y_train[start_ind:end_ind]).to(device = device)

        optimizer.zero_grad()
        batch_x = batch_x.view((-1, 3, 32, 32))
        preds = mobilenet(batch_x)
        cost = loss(preds, batch_y.long())
        cost.backward()
        optimizer.step()        

        # Compute average loss
        avg_cost += cost / total_batch
    # Display logs per epoch step
    print("Epoch:", '%04d' % (epoch+1), "cost=", \
            "{:.9f}".format(avg_cost))

    N = x_test.shape[0]
    total_batch = int(np.ceil(N * 1.0 / bsize))
    np_preds = np.zeros_like(y_test)

    for i in range(total_batch):
        # val
        preds = mobilenet(torch.Tensor(x_test[i*bsize : np.min([(i+1)*bsize, N])]).view((-1, 3, 32, 32)).to(device=device))
        preds = np.argmax(preds.detach().cpu().numpy(), axis=1)
        np_preds[i*bsize : np.min([(i+1)*bsize, N])] = preds
    print(list(np_preds == y_test).count(True)/N)

print("Optimization Finished!")
