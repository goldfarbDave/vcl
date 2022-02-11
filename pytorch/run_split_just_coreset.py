import numpy as np
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

data_gen = SplitMnistGenerator()
rand_vcl_result = vcl.run_coreset_only(hidden_size, no_epochs, data_gen,
   coreset.rand_from_batch, coreset_size, batch_size, single_head)
print(rand_vcl_result)
np.save("./results/rand-coreset-only-split{}".format(""), rand_vcl_result)

data_gen = SplitMnistGenerator()
kcen_vcl_result = vcl.run_coreset_only(hidden_size, no_epochs, data_gen,
    coreset.k_center, coreset_size, batch_size, single_head)
print(kcen_vcl_result)
np.save("./results/kcen-coreset-only-split{}".format(""), kcen_vcl_result)
