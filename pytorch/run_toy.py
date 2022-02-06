import numpy as np
import utils.vcl as vcl
import utils.coreset as coreset
from utils.DataGenerator import ToyDatasetGenerator

hidden_size = [10]
batch_size = None
no_epochs = 122
single_head = False
run_coreset_only = False
np.random.seed(0)

#Just VCL
coreset_size = 0
data_gen = ToyDatasetGenerator(num_samples_per_class=100)
vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
   coreset.rand_from_batch, coreset_size, batch_size, single_head, is_toy=True)
np.save("./results/VCL-toy{}".format(""), vcl_result)
