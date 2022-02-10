import numpy as np
from utils.DataGenerator import SplitMnistGenerator
import torch
from utils.multihead_models import Vanilla_CNN, Vanilla_NN, MFVI_NN, MFVI_CNN
data_gen = SplitMnistGenerator()
x_train, y_train, x_test, y_test = data_gen.next_task()
# in_dim_cnn, out_dim_cnn = 12*12, 2#4*4, 2
# in_dim_fc, out_dim_fc = data_gen.get_dims()
# hidden_size_cnn = [48,48]#[8,8]
# hidden_size_fc = [256, 256]
# model_cnn = Vanilla_CNN(in_dim_cnn, hidden_size_cnn,out_dim_cnn, x_train.shape[0])
# model_cnn.train(x_train, y_train, 0, 50, x_train.shape[0])
# model_fc = Vanilla_NN(in_dim_fc, hidden_size_fc,out_dim_fc, x_train.shape[0])
# model_fc.train(x_train, y_train, 0, 50, x_train.shape[0])
# x_test_ten = torch.Tensor(x_test).to(device=torch.device("cuda:0"))
# probs_cnn = model_cnn.prediction_prob(x_test_ten, 0)
# probs_fc = model_fc.prediction_prob(x_test_ten, 0)
# preds_cnn = torch.argmax(probs_cnn, dim=1)
# preds_fc = torch.argmax(probs_fc, dim=1)

# fc_weights = model_fc.get_weights()
# cnn_weights = model_cnn.get_weights()
# import pickle
# with open("cnn_weights_init.pkl", 'wb') as f:
#     pickle.dump(cnn_weights, f)
# with open("fc_weights_init.pkl", 'wb') as f:
#     pickle.dump(fc_weights, f)
# print(sum(preds_cnn == preds_fc))


in_dim_cnn, out_dim_cnn = 12*12, 2#4*4, 2
in_dim_fc, out_dim_fc = data_gen.get_dims()
hidden_size_cnn = [48,48]#[8,8]
hidden_size_fc = [256, 256]

import pickle
with open("cnn_weights_init.pkl", 'rb') as f:
    cnn_weights = pickle.load(f)
with open("fc_weights_init.pkl", 'rb') as f:
    fc_weights = pickle.load(f)
# model_fc = MFVI_NN(in_dim_fc, hidden_size_fc, out_dim_fc, x_train.shape[0], single_head=False, prev_means=fc_weights)
model_cnn = MFVI_CNN(in_dim_cnn, hidden_size_cnn, out_dim_cnn, x_train.shape[0], single_head=False, prev_means=cnn_weights)

import pdb; pdb.set_trace()
