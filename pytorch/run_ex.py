import numpy as np
import random
from utils.DataGenerator import SplitMnistGenerator
import torch
from utils.multihead_models import Vanilla_CNN, Vanilla_NN, MFVI_NN, MFVI_CNN
import utils.coreset as coreset
import utils.test as test
from utils.vcl import run_vcl_cnn, run_vcl


N_SEEDS = 3

for i in range(1, N_SEEDS+1):
    # seed everything
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)

    data_gen = SplitMnistGenerator()

    in_dim_cnn, out_dim_cnn = 12*12, 2 #4*4, 2
    in_dim_fc, out_dim_fc = data_gen.get_dims()
    hidden_size_cnn = [48,48]#[8,8]
    hidden_size_fc = [256, 256]
    no_epochs = 50

    # CNN VCL
    vcl_result = run_vcl_cnn(in_dim_cnn, hidden_size_cnn, out_dim_cnn, no_epochs, data_gen,
        coreset.rand_from_batch, coreset_size=0, batch_size=None, single_head=False, use_lrt=True)
    print(vcl_result)
    np.save(f"./results/3_seeds_exp/VCL-split-CNN-seed{i}", vcl_result)

    # CNN VCL with random Coreset
    coreset_size = 40
    data_gen = SplitMnistGenerator()
    rand_vcl_result = run_vcl_cnn(in_dim_cnn, hidden_size_cnn, out_dim_cnn, no_epochs, data_gen,
        coreset.rand_from_batch, coreset_size=coreset_size, batch_size=None, single_head=False, use_lrt=True)
    print(rand_vcl_result)
    np.save(f"./results/3_seeds_exp/VCL-split-CNN-randCoreset-seed{i}", rand_vcl_result)

    # CNN VCL with K-center Coreset
    coreset_size = 40
    data_gen = SplitMnistGenerator()
    kcen_vcl_result = run_vcl_cnn(in_dim_cnn, hidden_size_cnn, out_dim_cnn, no_epochs, data_gen,
        coreset.k_center, coreset_size=coreset_size, batch_size=None, single_head=False, use_lrt=True)
    print(kcen_vcl_result)
    np.save(f"./results/3_seeds_exp/VCL-split-CNN-kCenterCoreset-seed{i}", kcen_vcl_result)

