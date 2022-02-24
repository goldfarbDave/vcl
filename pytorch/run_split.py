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
run_coreset_only = False

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
   data_gen = SplitMnistGenerator()
   vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
      coreset.rand_from_batch, coreset_size, batch_size, single_head)
   np.save(f"./results/new_exp/VCL-split-seed{i}", vcl_result)

   #Just VCL + VGR
   coreset_size = 0
   data_gen = SplitMnistGenerator()
   vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
      coreset.rand_from_batch, coreset_size, batch_size, single_head, gan_bol=True)
   np.save(f"./results/new_exp/VCL-split-with-VGR-seed{i}", vcl_result)

   #VCL + Random Coreset
   coreset_size = 40
   data_gen = SplitMnistGenerator()
   rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
      coreset.rand_from_batch, coreset_size, batch_size, single_head,gan_bol= False)
   print(rand_vcl_result)
   np.save(f"./results/new_exp/VCL-all-split-randcoreset-seed{i}", rand_vcl_result)

   #VCL + Random Coreset with VGR
   coreset_size = 40
   data_gen = SplitMnistGenerator()
   rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
      coreset.rand_from_batch, coreset_size, batch_size, single_head,gan_bol= True)
   print(rand_vcl_result)
   np.save(f"./results/new_exp/VCL-all-split-randCoreset-VGR-seed{i}", rand_vcl_result)

   #VCL + k-center coreset
   coreset_size = 40
   data_gen = SplitMnistGenerator()
   kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
      coreset.k_center, coreset_size, batch_size, single_head)
   print(kcen_vcl_result)
   np.save(f"./results/new_exp/VCL-all-split-kCenterCoreset-seed{i}", kcen_vcl_result)

   #VCL + k-center coreset with VGR
   coreset_size = 40
   data_gen = SplitMnistGenerator()
   kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
      coreset.k_center, coreset_size, batch_size, single_head, gan_bol=True)
   print(kcen_vcl_result)
   np.save(f"./results/new_exp/VCL-all-split-kCenterCoreset-VGR-seed{i}", kcen_vcl_result)