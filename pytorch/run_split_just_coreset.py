import numpy as np
import torch
import random
import utils.vcl as vcl
import utils.coreset as coreset
from utils.DataGenerator import SplitMnistGenerator
from utils.seed import seed_everything

seed_everything()

hidden_size = [256, 256]
batch_size = None
no_epochs = 120
single_head = False
coreset_size = 40


N_SEEDS = 3

for i in range(1, N_SEEDS+1):
    # seed everything
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)

    data_gen = SplitMnistGenerator()
    rand_vcl_result = vcl.run_coreset_only(hidden_size, no_epochs, data_gen,
                        coreset.rand_from_batch, coreset_size, batch_size, single_head)
    print(rand_vcl_result)
    np.save("./results/new_exp/rand-coreset-only-split{}-seed{}".format("", i), rand_vcl_result)

    data_gen = SplitMnistGenerator()
    kcen_vcl_result = vcl.run_coreset_only(hidden_size, no_epochs, data_gen,
                        coreset.k_center, coreset_size, batch_size, single_head)
    print(kcen_vcl_result)
    np.save("./results/new_exp/kcen-coreset-only-split{}-seed{}".format("", i), kcen_vcl_result)
