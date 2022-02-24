import numpy as np
import torch
import random
import utils.vcl as vcl
import utils.coreset as coreset
from utils.DataGenerator import PermutedMnistGenerator
from utils.flags import FLAGS
from utils.seed import seed_everything

hidden_size = [100, 100]
batch_size = 256
no_epochs = 100
single_head = True
num_tasks = 10

seed_everything()


N_SEEDS = 3

for i in range(1, N_SEEDS+1):
    # seed everything
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)

    #Just VCL
    coreset_size = 0
    data_gen = PermutedMnistGenerator(num_tasks)
    vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
                    coreset.rand_from_batch, coreset_size, batch_size, single_head)
    print(vcl_result)
    np.save("./results/new_exp/VCL{}-seed{}".format("", i), vcl_result)

    #VCL + Random Coreset
    for coreset_size in [200,400,1000,2500,5000]:
        data_gen = PermutedMnistGenerator(num_tasks)
        rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
                            coreset.rand_from_batch, coreset_size, batch_size, single_head)
        print(rand_vcl_result)
        np.save("./results/new_exp/rand-VCL-nonVGR-{}-seed{}".format(coreset_size, i), rand_vcl_result)

    #VCL + Random Coreset + VGR
    for coreset_size in [200,400,1000,2500,5000]:
        data_gen = PermutedMnistGenerator(num_tasks)
        rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
                            coreset.rand_from_batch, coreset_size, batch_size, single_head, gan_bol=True)
        print(rand_vcl_result)
        np.save("./results/new_exp/rand-VCL-VGR-{}-seed{}".format(coreset_size, i), rand_vcl_result)

    #VCL + k-center coreset
    coreset_size = 200
    data_gen = PermutedMnistGenerator(num_tasks)
    kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
                        coreset.k_center, coreset_size, batch_size, single_head)
    print(kcen_vcl_result)
    np.save("./results/new_exp/kcen-VCL{}-seed{}".format(coreset_size, i), kcen_vcl_result)

