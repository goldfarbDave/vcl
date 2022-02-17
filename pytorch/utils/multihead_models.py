from select import select
import torch
from . import flags 
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import truncnorm
from copy import deepcopy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# variable initialization functions
def truncated_normal(size, stddev=1, variable = False, mean=0):
    mu, sigma = mean, stddev
    lower, upper= -2 * sigma, 2 * sigma
    X = truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    X_tensor = torch.Tensor(data = X.rvs(size)).to(device = device)
    X_tensor.requires_grad = variable
    return X_tensor

def init_tensor(value, dout=None, din = 1, variable = False, *, ten_like=None):
    # Sould probably split this up to deal with random data vs random_like data
    if ten_like is None and dout is None:
        assert False
    if ten_like is not None:
        x = value * torch.ones_like(ten_like).to(device=device)
    else:
        if din != 1:
            x = value * torch.ones([din, dout]).to(device = device)
        else:
            x = value * torch.ones([dout]).to(device = device)
    x.requires_grad=variable

    return x

class Cla_NN(object):
    def __init__(self, input_size, hidden_size, output_size, training_size):
        return


    def train(self, x_train, y_train, task_idx, no_epochs=1000, batch_size=100, display_epoch=5):
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
    def get_params(self):
        pass
    def prediction_prob(self, x_test, task_idx):
        prob = F.softmax(self._prediction(x_test, task_idx, self.no_pred_samples), dim=-1)
        return prob



""" Neural Network Model """
class Vanilla_NN(Cla_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size, learning_rate=0.001):
        #
        super(Vanilla_NN, self).__init__(input_size, hidden_size, output_size, training_size)
        # # init weights and biases
        self.W, self.b, self.W_last, self.b_last, self.size = self.create_weights(
                 input_size, hidden_size, output_size)
        # no of hidden + input layers
        self.no_layers = len(hidden_size) + 1
        # list of all parameters [theta in paper]
        self.weights = self.W + self.b + self.W_last + self.b_last
        self.training_size = training_size
        self.optimizer = optim.Adam(self.weights, lr=learning_rate)


    def _prediction(self, inputs, task_idx):
        act = inputs
        # forward pass through network
        for i in range(self.no_layers-1):
             pre = torch.add(torch.matmul(act, self.W[i]), self.b[i])
             act = F.relu(pre)
        pre = torch.add(torch.matmul(act, self.W_last[task_idx]), self.b_last[task_idx])
        return pre

    def _logpred(self, inputs, targets, task_idx):
        # expected log likelihood of data - first term in eqn 4 of paper
        loss = torch.nn.CrossEntropyLoss()
        pred = self._prediction(inputs, task_idx)
        log_lik = - loss(pred, targets.type(torch.long))
        return log_lik

    def prediction_prob(self, x_test, task_idx):
        prob = F.softmax(self._prediction(x_test, task_idx), dim=-1)
        return prob

    def get_loss(self, batch_x, batch_y, task_idx):
        # no kl term for first task since q_t(theta) = q_t-1(theta) = p(theta)
        return -self._logpred(batch_x, batch_y, task_idx)

    def create_weights(self, in_dim, hidden_size, out_dim):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)

        # no of hidden + input layers
        no_layers = len(hidden_size) - 1
        W = []
        b = []
        # output layer weights and biases
        W_last = []
        b_last = []
        # iterating over only hidden layers
        for i in range(no_layers-1):
            din = hidden_size[i]
            dout = hidden_size[i+1]
            
            #Initializiation values of means
            Wi_m = truncated_normal([din, dout], stddev=0.1, variable = True)
            bi_m = truncated_normal([dout], stddev=0.1, variable = True)

            #Append to list weights
            W.append(Wi_m)
            b.append(bi_m)

        # last layer weight matrix and bias distribution initialisation
        Wi = truncated_normal([hidden_size[-2], out_dim], stddev=0.1, variable = True)
        bi = truncated_normal([out_dim], stddev=0.1, variable = True)
        W_last.append(Wi)
        b_last.append(bi)
        return W, b, W_last, b_last, hidden_size

    def get_weights(self):
        weights = [self.weights[:self.no_layers-1], self.weights[self.no_layers-1:2*(self.no_layers-1)], [self.weights[-2]], [self.weights[-1]]]
        return weights
""" Neural Network Model """
class Vanilla_CNN(Cla_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size, learning_rate=0.0001, kern_size=3, is_cifar=False):
        super(Vanilla_CNN, self).__init__(input_size, hidden_size, output_size, training_size)
        self.is_cifar = is_cifar
        # # init weights and biases
        self.W, self.b, self.kern_weights, self.kern_bias, self.W_last, self.b_last, self.size = self.create_weights(
                 input_size, hidden_size, output_size, kern_size)
        # no of hidden + input layers + conv layer
        self.no_layers = len(hidden_size) + 1
        # list of all parameters [theta in paper]
        self.weights = self.W + self.b + self.kern_weights + self.kern_bias + self.W_last + self.b_last
        self.training_size = training_size
        self.optimizer = optim.Adam(self.weights, lr=learning_rate)

    def _prediction(self, inputs, task_idx):
        act = inputs
        # forward pass through network
        d = 32 if self.is_cifar else 28
        in_chan = 3 if self.is_cifar else 1
        act = act.view((-1, in_chan, d, d))
        for idx,(weights,bias) in enumerate(zip(self.kern_weights, self.kern_bias)):
            stride = idx+2 if self.is_cifar else idx+1
            act = F.relu(F.conv2d(input=act, weight=weights, bias=bias, stride=stride))
        #import ipdb; ipdb.set_trace()
        shape_idx = 3 if self.is_cifar else 2
        act = act.view((-1, np.prod(act.shape[-shape_idx:])))

        for i in range(self.no_layers-1):
            act = F.relu(torch.add(torch.matmul(act, self.W[i]), self.b[i]))
        pre = torch.add(torch.matmul(act, self.W_last[task_idx]), self.b_last[task_idx])
        return pre

    def _logpred(self, inputs, targets, task_idx):
        # expected log likelihood of data - first term in eqn 4 of paper
        loss = torch.nn.CrossEntropyLoss()
        pred = self._prediction(inputs, task_idx)
        log_lik = - loss(pred, targets.type(torch.long))
        return log_lik

    def prediction_prob(self, x_test, task_idx):
        prob = F.softmax(self._prediction(x_test, task_idx), dim=-1)
        return prob

    def get_loss(self, batch_x, batch_y, task_idx):
        # no kl term for first task since q_t(theta) = q_t-1(theta) = p(theta)
        return -self._logpred(batch_x, batch_y, task_idx)

    def create_weights(self, in_dim, hidden_size, out_dim, kern_size):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)

        # no of hidden + input layers
        no_layers = len(hidden_size) - 1
        W = []
        b = []
        # output layer weights and biases
        W_last = []
        b_last = []
        # iterating over only hidden layers
        for i in range(no_layers-1):
            din = hidden_size[i]
            dout = hidden_size[i+1]

            #Initializiation values of means
            Wi_m = truncated_normal([din, dout], stddev=0.1, variable = True)
            bi_m = truncated_normal([dout], stddev=0.1, variable = True)

            #Append to list weights
            W.append(Wi_m)
            b.append(bi_m)

        # last layer weight matrix and bias distribution initialisation
        Wi = truncated_normal([hidden_size[-2], out_dim], stddev=0.1, variable = True)
        bi = truncated_normal([out_dim], stddev=0.1, variable = True)
        W_last.append(Wi)
        b_last.append(bi)


        if self.is_cifar:
            # chanseq = [3,128,64,32,16,8]#[3,32,32,64,64,128,128]
            # kern_weights = []
            # kern_bias = []
            # for in_chan, out_chan in zip(chanseq, chanseq[1:]):
            #     kern_weights.append(
            #         truncated_normal([out_chan, in_chan, kern_size,kern_size],
            #                          stddev=0.1,
            #                          variable=True)
            #     )
            #     kern_bias.append(
            #         truncated_normal([out_chan], stddev=0.1, variable=True)
            #     )

            # 4->1 channels
            kern_weights = [truncated_normal([32,3,kern_size, kern_size], stddev=0.1, variable=True),
                            truncated_normal([64,32,kern_size, kern_size], stddev=0.1, variable=True)]
            kern_bias = [truncated_normal([32], stddev=0.1, variable=True),
                         truncated_normal([64], stddev=0.1, variable=True)]                

        else:
            # 4->1 channels
            kern_weights = [truncated_normal([4,1,kern_size, kern_size], stddev=0.1, variable=True),
                            truncated_normal([1,4,kern_size, kern_size], stddev=0.1, variable=True)]
            kern_bias = [truncated_normal([4], stddev=0.1, variable=True),
                         truncated_normal([1], stddev=0.1, variable=True)]

        return W, b, kern_weights, kern_bias, W_last, b_last, hidden_size

    def get_weights(self):
        w = self.weights
        return {"DenseWeights": self.W,
                "DenseBias": self.b,
                "KernWeights": self.kern_weights,
                "KernBias":self.kern_bias,
                "LastWeights":self.W_last,
                "LastBias":self.b_last}

""" Bayesian Neural Network with Mean field VI approximation """
class MFVI_NN(Cla_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size,
        no_train_samples=10, no_pred_samples=100, single_head = False, prev_means=None, learning_rate=0.001, LRT=False):
        ##TODO: handle single head
        super(MFVI_NN, self).__init__(input_size, hidden_size, output_size, training_size)

        m1, v1, hidden_size = self.create_weights(
             input_size, hidden_size, output_size, prev_means)

        self.input_size = input_size
        self.out_size = output_size
        self.size = hidden_size
        self.single_head = single_head

        self.LRT = LRT

        # hidden layer weight and bias means and variances - shared across tasks
        self.W_m, self.b_m = m1[0], m1[1]
        self.W_v, self.b_v = v1[0], v1[1]

        # output layer weight and bias means and variances - independent for each task
        # these will be populated as and when we get new tasks - initially no task heads
        self.W_last_m, self.b_last_m = [], []
        self.W_last_v, self.b_last_v = [], []

        m2, v2 = self.create_prior(input_size, self.size, output_size)

        # prior means and variances on shared weights and biases
        self.prior_W_m, self.prior_b_m, = m2[0], m2[1]
        self.prior_W_v, self.prior_b_v = v2[0], v2[1]

        # prior means and variances on task-specific weights and biases
        # these will be populated as and when we get new tasks - initially no task heads
        self.prior_W_last_m, self.prior_b_last_m = [], []
        self.prior_W_last_v, self.prior_b_last_v = [], []

        self.W_m_copy, self.W_v_copy, self.b_m_copy, self.b_v_copy = None, None, None, None
        self.W_last_m_copy, self.W_last_v_copy, self.b_last_m_copy, self.b_last_v_copy = None, None, None, None
        self.prior_W_m_copy, self.prior_W_v_copy, self.prior_b_m_copy, self.prior_b_v_copy = None, None, None, None
        self.prior_W_last_m_copy, self.prior_W_last_v_copy, self.prior_b_last_m_copy, self.prior_b_last_v_copy = None, None, None, None

        self.no_layers = len(self.size) - 1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.training_size = training_size
        self.learning_rate = learning_rate

        if prev_means is not None:
            # initialise first task head
            self.init_first_head(prev_means)
        else:
            # create new task head -- involves new prior and posterior weights and biases for the new task
            self.create_head()

        m1.append(self.W_last_m)
        m1.append(self.b_last_m)
        v1.append(self.W_last_v)
        v1.append(self.b_last_v)

        r1 = m1 + v1
        self.weights = [item for sublist in r1 for item in sublist]

        self.optimizer = optim.Adam(self.weights, lr=learning_rate)

    def get_loss(self, batch_x, batch_y, task_idx):
        # equation 4
        return torch.div(self._KL_term(), self.training_size) - self._logpred(batch_x, batch_y, task_idx)

    def _prediction(self, inputs, task_idx, no_samples):
        K = no_samples
        size = self.size
        act = torch.unsqueeze(inputs, 0).repeat([K, 1, 1])
        for i in range(self.no_layers-1):
            din = self.size[i]
            dout = self.size[i+1]

            if(self.LRT):
                #local reparameterisation -- compute act_mu and act_sigma and then do act_mu+eps*act_sigma
                # repeat the individual weight mean and biases K times for performing the activation sampling
                repeated_weight_mean = self.W_m[i].repeat([K, 1, 1]).view(-1, *self.W_m[i].shape)
                repeated_weight_logsigma = self.W_v[i].repeat([K, 1, 1]).view(-1, *self.W_v[i].shape)
                repeated_bias_mean = self.b_m[i].repeat([K, 1, 1]).view(-1, *self.b_m[i].shape).unsqueeze(1)
                repeated_bias_logsigma = self.b_v[i].repeat([K, 1, 1]).view(-1, *self.b_v[i].shape).unsqueeze(1)

                # compute activation mu and sigma
                act_mu = torch.add(torch.einsum('mni,mio->mno', act, repeated_weight_mean), repeated_bias_mean)
                act_sigma = torch.sqrt(1e-9 + torch.add(torch.einsum('mni,mio->mno', act**2, torch.exp(0.5*repeated_weight_logsigma)**2), torch.exp(0.5*repeated_bias_logsigma)**2))
                # sample activation from act_mu and act_sigma
                eps_sample = torch.normal(torch.zeros_like(act_mu), torch.ones_like(act_mu)).to(device = device)
                # perform LRT
                pre = torch.add(eps_sample * act_sigma, act_mu)
                act = F.relu(pre)
            else:
                # normal reparameterisation -- sample eps and compute mu+eps*sigma
                eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout))).to(device = device)
                eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout))).to(device = device)
                # random sample weight from distribution -- reparameterisation trick
                weights = torch.add(eps_w * torch.exp(0.5*self.W_v[i]), self.W_m[i])
                biases = torch.add(eps_b * torch.exp(0.5*self.b_v[i]), self.b_m[i])
                pre = torch.add(torch.einsum('mni,mio->mno', act, weights), biases)
                act = F.relu(pre)

        din = self.size[-2]
        dout = self.size[-1]

        Wtask_m = self.W_last_m[task_idx]
        Wtask_v = self.W_last_v[task_idx]
        btask_m = self.b_last_m[task_idx]
        btask_v = self.b_last_v[task_idx]

        if(self.LRT):
            #local reparameterisation -- compute act_mu and act_sigma and then do act_mu+eps*act_sigma
            # repeat the individual weight mean and biases K times for performing the activation sampling
            repeated_last_weight_mean = Wtask_m.repeat([K, 1, 1]).view(-1, *Wtask_m.shape).unsqueeze(1)
            repeated_last_weight_logsigma = Wtask_v.repeat([K, 1, 1]).view(-1, *Wtask_v.shape).unsqueeze(1)
            repeated_last_bias_mean = btask_m.repeat([K, 1, 1]).view(-1, *btask_m.shape).unsqueeze(1)
            repeated_last_bias_logsigma = btask_v.repeat([K, 1, 1]).view(-1, *btask_v.shape).unsqueeze(1)
            act = torch.unsqueeze(act, 3)

            # compute activation mu and sigma
            pre_last_mu = torch.add(torch.sum(act * repeated_last_weight_mean, dim = 2), repeated_last_bias_mean)
            pre_last_sigma = torch.sqrt(1e-9 + torch.add(torch.sum(act**2 * torch.exp(0.5*repeated_last_weight_logsigma)**2, dim = 2), torch.exp(0.5*repeated_last_bias_logsigma)**2))
            # sample activation from act_mu and act_sigma
            eps_last_sample = torch.normal(torch.zeros_like(pre_last_mu), torch.ones_like(pre_last_mu)).to(device = device)
            # perform LRT
            pre = torch.add(eps_last_sample * pre_last_sigma, pre_last_mu)
        else:
            eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout))).to(device = device)
            eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout))).to(device = device)

            # random sample weight from distribution -- reparameterisation trick
            weights = torch.add(eps_w * torch.exp(0.5*Wtask_v),Wtask_m)
            biases = torch.add(eps_b * torch.exp(0.5*btask_v), btask_m)
            act = torch.unsqueeze(act, 3)
            weights = torch.unsqueeze(weights, 1)
            pre = torch.add(torch.sum(act * weights, dim = 2), biases)
        return pre

    def _logpred(self, inputs, targets, task_idx):
        loss = torch.nn.CrossEntropyLoss()
        pred = self._prediction(inputs, task_idx, self.no_train_samples).view(-1,self.out_size)
        targets = targets.repeat([self.no_train_samples, 1]).view(-1)
        log_liks = -loss(pred, targets.type(torch.long))
        log_lik = log_liks.mean()
        return log_lik

    def _KL_term(self):
        kl = 0
        # theta_S
        for i in range(self.no_layers-1):
            din = self.size[i]
            dout = self.size[i+1]
            # weight from current posterior - q_t(theta)
            m, v = self.W_m[i], self.W_v[i]
            # weight from prev posterior - q_t-1(theta)
            m0, v0 = self.prior_W_m[i], self.prior_W_v[i]

            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            # bias from current posterior - q_t(theta)
            m, v = self.b_m[i], self.b_v[i]
            # bias from prev posterior - q_t-1(theta)
            m0, v0 = self.prior_b_m[i], self.prior_b_v[i]

            const_term = -0.5 * dout
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl +=  log_std_diff + mu_diff_term + const_term

        no_tasks = len(self.W_last_m)
        din = self.size[-2]
        dout = self.size[-1]

        # theta_Hs
        for i in range(no_tasks):
            m, v = self.W_last_m[i], self.W_last_v[i]
            m0, v0 = self.prior_W_last_m[i], self.prior_W_last_v[i]

            const_term = - 0.5 * dout * din
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            m, v = self.b_last_m[i], self.b_last_v[i]
            m0, v0 = self.prior_b_last_m[i], self.prior_b_last_v[i]

            const_term = -0.5 * dout
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term
        return kl

    def save_weights(self):
        ''' Save weights before training on the coreset before getting the test accuracy '''

        print("Saving weights before core set training")
        self.W_m_copy = [self.W_m[i].clone().detach().data for i in range(len(self.W_m))]
        self.W_v_copy = [self.W_v[i].clone().detach().data for i in range(len(self.W_v))]
        self.b_m_copy = [self.b_m[i].clone().detach().data for i in range(len(self.b_m))]
        self.b_v_copy = [self.b_v[i].clone().detach().data for i in range(len(self.b_v))]

        self.W_last_m_copy = [self.W_last_m[i].clone().detach().data for i in range(len(self.W_last_m))]
        self.W_last_v_copy = [self.W_last_v[i].clone().detach().data for i in range(len(self.W_last_v))]
        self.b_last_m_copy = [self.b_last_m[i].clone().detach().data for i in range(len(self.b_last_m))]
        self.b_last_v_copy = [self.b_last_v[i].clone().detach().data for i in range(len(self.b_last_v))]

        self.prior_W_m_copy = [self.prior_W_m[i].data for i in range(len(self.prior_W_m))]
        self.prior_W_v_copy = [self.prior_W_v[i].data for i in range(len(self.prior_W_v))]
        self.prior_b_m_copy = [self.prior_b_m[i].data for i in range(len(self.prior_b_m))]
        self.prior_b_v_copy = [self.prior_b_v[i].data for i in range(len(self.prior_b_v))]

        self.prior_W_last_m_copy = [self.prior_W_last_m[i].data for i in range(len(self.prior_W_last_m))]
        self.prior_W_last_v_copy = [self.prior_W_last_v[i].data for i in range(len(self.prior_W_last_v))]
        self.prior_b_last_m_copy = [self.prior_b_last_m[i].data for i in range(len(self.prior_b_last_m))]
        self.prior_b_last_v_copy = [self.prior_b_last_v[i].data for i in range(len(self.prior_b_last_v))]

        return

    def load_weights(self):
        ''' Re-load weights after getting the test accuracy '''

        print("Reloading previous weights after core set training")
        self.weights = []
        self.W_m = [self.W_m_copy[i].clone().detach().data for i in range(len(self.W_m))]
        self.W_v = [self.W_v_copy[i].clone().detach().data for i in range(len(self.W_v))]
        self.b_m = [self.b_m_copy[i].clone().detach().data for i in range(len(self.b_m))]
        self.b_v = [self.b_v_copy[i].clone().detach().data for i in range(len(self.b_v))]

        for i in range(len(self.W_m)):
            self.W_m[i].requires_grad = True
            self.W_v[i].requires_grad = True
            self.b_m[i].requires_grad = True
            self.b_v[i].requires_grad = True

        self.weights += self.W_m
        self.weights += self.W_v
        self.weights += self.b_m
        self.weights += self.b_v


        self.W_last_m = [self.W_last_m_copy[i].clone().detach().data for i in range(len(self.W_last_m))]
        self.W_last_v = [self.W_last_v_copy[i].clone().detach().data for i in range(len(self.W_last_v))]
        self.b_last_m = [self.b_last_m_copy[i].clone().detach().data for i in range(len(self.b_last_m))]
        self.b_last_v = [self.b_last_v_copy[i].clone().detach().data for i in range(len(self.b_last_v))]

        for i in range(len(self.W_last_m)):
            self.W_last_m[i].requires_grad = True
            self.W_last_v[i].requires_grad = True
            self.b_last_m[i].requires_grad = True
            self.b_last_v[i].requires_grad = True

        self.weights += self.W_last_m
        self.weights += self.W_last_v
        self.weights += self.b_last_m
        self.weights += self.b_last_v

        self.optimizer = optim.Adam(self.weights, lr=self.learning_rate)
        self.prior_W_m = [self.prior_W_m_copy[i].data for i in range(len(self.prior_W_m))]
        self.prior_W_v = [self.prior_W_v_copy[i].data for i in range(len(self.prior_W_v))]
        self.prior_b_m = [self.prior_b_m_copy[i].data for i in range(len(self.prior_b_m))]
        self.prior_b_v = [self.prior_b_v_copy[i].data for i in range(len(self.prior_b_v))]

        self.prior_W_last_m = [self.prior_W_last_m_copy[i].data for i in range(len(self.prior_W_last_m))]
        self.prior_W_last_v = [self.prior_W_last_v_copy[i].data for i in range(len(self.prior_W_last_v))]
        self.prior_b_last_m = [self.prior_b_last_m_copy[i].data for i in range(len(self.prior_b_last_m))]
        self.prior_b_last_v = [self.prior_b_last_v_copy[i].data for i in range(len(self.prior_b_last_v))]

        return

    def clean_copy_weights(self):
        self.W_m_copy, self.W_v_copy, self.b_m_copy, self.b_v_copy = None, None, None, None
        self.W_last_m_copy, self.W_last_v_copy, self.b_last_m_copy, self.b_last_v_copy = None, None, None, None
        self.prior_W_m_copy, self.prior_W_v_copy, self.prior_b_m_copy, self.prior_b_v_copy = None, None, None, None
        self.prior_W_last_m_copy, self.prior_W_last_v_copy, self.prior_b_last_m_copy, self.prior_b_last_v_copy = None, None, None, None

    def create_head(self):
        ''''Create new head when a new task is detected'''
        print("creating a new head")
        din = self.size[-2]
        dout = self.size[-1]

        W_m= truncated_normal([din, dout], stddev=0.1, variable=True)
        b_m= truncated_normal([dout], stddev=0.1, variable=True)
        W_v = init_tensor(-6.0,  dout = dout, din = din, variable= True)
        b_v = init_tensor(-6.0,  dout = dout, variable= True)

        self.W_last_m.append(W_m)
        self.W_last_v.append(W_v)
        self.b_last_m.append(b_m)
        self.b_last_v.append(b_v)


        W_m_p = torch.zeros([din, dout]).to(device = device)
        b_m_p = torch.zeros([dout]).to(device = device)
        W_v_p =  init_tensor(1,  dout = dout, din = din)
        b_v_p = init_tensor(1, dout = dout)

        self.prior_W_last_m.append(W_m_p)
        self.prior_W_last_v.append(W_v_p)
        self.prior_b_last_m.append(b_m_p)
        self.prior_b_last_v.append(b_v_p)
        # update weights list to contain shared plus task-specific weights and biases
        self.weights = []
        self.weights += self.W_m
        self.weights += self.W_v
        self.weights += self.b_m
        self.weights += self.b_v
        self.weights += self.W_last_m
        self.weights += self.W_last_v
        self.weights += self.b_last_m
        self.weights += self.b_last_v
        self.optimizer = optim.Adam(self.weights, lr=self.learning_rate)

        return


    def init_first_head(self, prev_means):
        ''''When the MFVI_NN is instantiated, we initialize weights with those of the Vanilla NN'''
        print("initializing first head")
        din = self.size[-2]
        dout = self.size[-1]
        self.prior_W_last_m = [torch.zeros([din, dout]).to(device = device)]
        self.prior_b_last_m = [torch.zeros([dout]).to(device = device)]
        self.prior_W_last_v =  [init_tensor(1,  dout = dout, din = din)]
        self.prior_b_last_v = [init_tensor(1, dout = dout)]

        # initialising the first task head
        W_last_m = prev_means[2][0].detach().data
        W_last_m.requires_grad = True
        self.W_last_m = [W_last_m]
        self.W_last_v = [init_tensor(-6.0,  dout = dout, din = din, variable= True)]


        b_last_m = prev_means[3][0].detach().data
        b_last_m.requires_grad = True
        self.b_last_m = [b_last_m]
        self.b_last_v = [init_tensor(-6.0, dout = dout, variable= True)]

        return

    def create_weights(self, in_dim, hidden_size, out_dim, prev_means):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)

        no_layers = len(hidden_size) - 1
        W_m = []
        b_m = []
        W_v = []
        b_v = []

        for i in range(no_layers-1):
            din = hidden_size[i]
            dout = hidden_size[i+1]
            if prev_means is not None:
                # new prior means are old posteriors means from prev task
                W_m_i = prev_means[0][i].detach().data
                W_m_i.requires_grad = True
                bi_m_i = prev_means[1][i].detach().data
                bi_m_i.requires_grad = True
            else:
            # This is same as prior weights - initialise with truncated normals
            #Initializiation values of means
                W_m_i= truncated_normal([din, dout], stddev=0.1, variable=True)
                bi_m_i= truncated_normal([dout], stddev=0.1, variable=True)
           
            #Initializiation values of variances
            # how are variances negative?
            W_v_i = init_tensor(-6.0,  dout = dout, din = din, variable = True)
            bi_v_i = init_tensor(-6.0,  dout = dout, variable = True)

            #Append to list weights
            W_m.append(W_m_i)
            b_m.append(bi_m_i)
            W_v.append(W_v_i)
            b_v.append(bi_v_i)

        # return means and variances
        return [W_m, b_m], [W_v, b_v], hidden_size

    def create_prior(self, in_dim, hidden_size, out_dim, initial_mean = 0, initial_variance = 1):

        no_layers = len(hidden_size) - 1
        W_m = []
        b_m = []

        W_v = []
        b_v = []

        for i in range(no_layers - 1):
            din = hidden_size[i]
            dout = hidden_size[i + 1]

            # Initializiation values of means
            W_m_val = initial_mean * torch.zeros([din, dout]).to(device = device)
            bi_m_val = initial_mean * torch.zeros([dout]).to(device = device)

            # Initializiation values of variances
            W_v_val = initial_variance * init_tensor(1,  dout = dout, din = din )
            bi_v_val =  initial_variance * init_tensor(1,  dout = dout)

            # Append to list weights
            W_m.append(W_m_val)
            b_m.append(bi_m_val)
            W_v.append(W_v_val)
            b_v.append(bi_v_val)

        return [W_m, b_m], [W_v, b_v]

    def update_prior(self):
        print("updating prior...")
        # update new prior to be old posterior weights and biases means and vars
        for i in range(len(self.W_m)):
            self.prior_W_m[i].data.copy_(self.W_m[i].clone().detach().data)
            self.prior_b_m[i].data.copy_(self.b_m[i].clone().detach().data)
            self.prior_W_v[i].data.copy_(torch.exp(self.W_v[i].clone().detach().data))
            self.prior_b_v[i].data.copy_(torch.exp(self.b_v[i].clone().detach().data))

        length = len(self.W_last_m)

        for i in range(length):
            self.prior_W_last_m[i].data.copy_(self.W_last_m[i].clone().detach().data)
            self.prior_b_last_m[i].data.copy_(self.b_last_m[i].clone().detach().data)
            self.prior_W_last_v[i].data.copy_(torch.exp(self.W_last_v[i].clone().detach().data))
            self.prior_b_last_v[i].data.copy_(torch.exp(self.b_last_v[i].clone().detach().data))

        return


""" Bayesian Neural Network with Mean field VI approximation """
class MFVI_CNN(Cla_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size,
                 no_train_samples=10, no_pred_samples=100, single_head = False, prev_means=None, kern_size=3, learning_rate=0.0023, LRT=False,is_cifar=False):
        ##TODO: handle single head
        super(MFVI_CNN, self).__init__(input_size, hidden_size, output_size, training_size)
        self.is_cifar=is_cifar
        m1, v1, kern_m, kern_v, hidden_size = self.create_weights(
             input_size, hidden_size, output_size, prev_means)

        self.input_size = input_size
        self.out_size = output_size
        self.size = hidden_size
        self.single_head = single_head

        self.LRT = LRT

        # hidden layer weight and bias means and variances - shared across tasks
        self.W_m, self.b_m = m1[0], m1[1]
        self.W_v, self.b_v = v1[0], v1[1]
        self.kern_weights_m, self.kern_bias_m = kern_m
        self.kern_weights_v, self.kern_bias_v = kern_v
        # output layer weight and bias means and variances - independent for each task
        # these will be populated as and when we get new tasks - initially no task heads
        self.W_last_m, self.b_last_m = [], []
        self.W_last_v, self.b_last_v = [], []

        m2, v2, kern_m_prior, kern_v_prior = self.create_prior(input_size, self.size, output_size)


        # prior means and variances on shared weights and biases
        self.prior_W_m, self.prior_b_m, = m2[0], m2[1]
        self.prior_W_v, self.prior_b_v = v2[0], v2[1]
        self.prior_kern_weights_m, self.prior_kern_bias_m = kern_m_prior
        self.prior_kern_weights_v, self.prior_kern_bias_v = kern_v_prior
        # prior means and variances on task-specific weights and biases
        # these will be populated as and when we get new tasks - initially no task heads
        self.prior_W_last_m, self.prior_b_last_m = [], []
        self.prior_W_last_v, self.prior_b_last_v = [], []

        self.W_m_copy, self.W_v_copy, self.b_m_copy, self.b_v_copy = None, None, None, None
        self.W_last_m_copy, self.W_last_v_copy, self.b_last_m_copy, self.b_last_v_copy = None, None, None, None
        self.prior_W_m_copy, self.prior_W_v_copy, self.prior_b_m_copy, self.prior_b_v_copy = None, None, None, None
        self.prior_W_last_m_copy, self.prior_W_last_v_copy, self.prior_b_last_m_copy, self.prior_b_last_v_copy = None, None, None, None
        self.kern_weights_m_copy, self.kern_weights_v_copy, self.kern_bias_m_copy, self.kern_bias_v_copy = None, None, None, None
        self.prior_kern_weights_m_copy, self.prior_kern_weights_v_copy, self.prior_kern_bias_m_copy, self.prior_kern_bias_v_copy = None, None, None, None

        self.no_layers = len(self.size) - 1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.training_size = training_size
        self.learning_rate = learning_rate

        if prev_means is not None:
            # initialise first task head
            self.init_first_head(prev_means)
        else:
            # create new task head -- involves new prior and posterior weights and biases for the new task
            self.create_head()


        m1.append(self.W_last_m)
        m1.append(self.b_last_m)
        v1.append(self.W_last_v)
        v1.append(self.b_last_v)

        r1 = m1 + v1 + kern_m + kern_v
        self.weights = [item for sublist in r1 for item in sublist]


        self.optimizer = optim.Adam(self.weights, lr=learning_rate)

    def get_loss(self, batch_x, batch_y, task_idx):
        # equation 4
        # if(task_idx>0):
        #     import ipdb; ipdb.set_trace()

        return torch.div(self._KL_term(), self.training_size) - self._logpred(batch_x, batch_y, task_idx)

    def _prediction(self, inputs, task_idx, no_samples):
        K = no_samples
        size = self.size
        d = 32 if self.is_cifar else 28
        in_chan = 3 if self.is_cifar else 1
        out_chan = 64 if self.is_cifar else 1
        act = torch.unsqueeze(inputs, 0).repeat([K, 1, 1])
        act = act.view((K, -1, in_chan, d, d))
        #final_conv_shape
        fcs = 5 if self.is_cifar else 12
        conv_out = torch.zeros((K, inputs.shape[0], out_chan, fcs, fcs)).to(device=device)
        for samp_ind in range(K):     
            pre = act[samp_ind]               
            for i, (kw_m, kb_m, kw_v, kb_v) in enumerate(zip(self.kern_weights_m, self.kern_bias_m, self.kern_weights_v, self.kern_bias_v)):
                
                # TODO: discuss and verify if LRT for convolutions is poor due to filter sampling
                # being more efficient as compared to activation sampling due to shapes. For now,
                # defaulting to using normal reparameterisation for conv layers and using LRT if 
                # required for FC layers

                # if(self.LRT):
                #     # perform local reparameterisation of kernel weights and biases
                #     # compute activation using only kernel weight means
                #     act_mu = F.conv2d(input=pre, weight=kw_m, bias=kb_m, stride=[i+1])
                #     # compute activation using only kernel weight variances
                #     act_sigma = torch.sqrt(1e-9 + F.conv2d(input=pre, weight=torch.exp(0.5*kw_v)**2, bias=torch.exp(0.5*kb_v)**2, stride=[i+1]))
                #     # sample activation from std normal
                #     eps_sample = torch.normal(torch.zeros_like(act_mu), torch.ones_like(act_mu)).to(device=device)
                #     # perform LRT
                #     pre = F.relu(torch.add(eps_sample * act_sigma, act_mu))
                # else:

                get_eps = lambda ten: torch.normal(torch.zeros(ten.shape)).to(device=device)
                eps_w = get_eps(kw_m)
                eps_b = get_eps(kb_m)
                weights = torch.add(eps_w *torch.exp(0.5*kw_v), kw_m)
                bias = torch.add(eps_b *torch.exp(0.5*kb_v), kb_m)
                stride = i+2 if self.is_cifar else i+1
                pre = F.relu(F.conv2d(input=pre, weight=weights, bias=bias, stride=[stride]))

            conv_out[samp_ind, :] = pre

            #TODO
            # pre = F.conv2d(input=act, weight=weights.unsqueeze(1), bias=bias.unsqueeze, stride=[i+1])
            # act = F.relu(pre)
        act = conv_out.view((K, inputs.shape[0], -1))

        for i in range(self.no_layers-1):
            din = self.size[i]
            dout = self.size[i+1]
            if(self.LRT):
                #local reparameterisation -- compute act_mu and act_sigma and then do act_mu+eps*act_sigma
                # repeat the individual weight mean and biases K times for performing the activation sampling
                repeated_weight_mean = self.W_m[i].repeat([K, 1, 1]).view(-1, *self.W_m[i].shape)
                repeated_weight_logsigma = self.W_v[i].repeat([K, 1, 1]).view(-1, *self.W_v[i].shape)
                repeated_bias_mean = self.b_m[i].repeat([K, 1, 1]).view(-1, *self.b_m[i].shape).unsqueeze(1)
                repeated_bias_logsigma = self.b_v[i].repeat([K, 1, 1]).view(-1, *self.b_v[i].shape).unsqueeze(1)

                # compute activation mu and sigma
                act_mu = torch.add(torch.einsum('mni,mio->mno', act, repeated_weight_mean), repeated_bias_mean)
                act_sigma = torch.sqrt(1e-9 + torch.add(torch.einsum('mni,mio->mno', act**2, torch.exp(0.5*repeated_weight_logsigma)**2), torch.exp(0.5*repeated_bias_logsigma)**2))
                # sample activation from act_mu and act_sigma
                eps_sample = torch.normal(torch.zeros_like(act_mu), torch.ones_like(act_mu)).to(device = device)
                # perform LRT
                pre = torch.add(eps_sample * act_sigma, act_mu)
                act = F.relu(pre)                
            else:
                eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout))).to(device = device)
                eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout))).to(device = device)
                # random sample weight from distribution -- reparameterisation trick
                weights = torch.add(eps_w * torch.exp(0.5*self.W_v[i]), self.W_m[i])
                biases = torch.add(eps_b * torch.exp(0.5*self.b_v[i]), self.b_m[i])
                pre = torch.add(torch.einsum('mni,mio->mno', act, weights), biases)
                act = F.relu(pre)

        din = self.size[-2]
        dout = self.size[-1]

        Wtask_m = self.W_last_m[task_idx]
        Wtask_v = self.W_last_v[task_idx]
        btask_m = self.b_last_m[task_idx]
        btask_v = self.b_last_v[task_idx]

        if(self.LRT):
            #local reparameterisation -- compute act_mu and act_sigma and then do act_mu+eps*act_sigma
            # repeat the individual weight mean and biases K times for performing the activation sampling
            repeated_last_weight_mean = Wtask_m.repeat([K, 1, 1]).view(-1, *Wtask_m.shape).unsqueeze(1)
            repeated_last_weight_logsigma = Wtask_v.repeat([K, 1, 1]).view(-1, *Wtask_v.shape).unsqueeze(1)
            repeated_last_bias_mean = btask_m.repeat([K, 1, 1]).view(-1, *btask_m.shape).unsqueeze(1)
            repeated_last_bias_logsigma = btask_v.repeat([K, 1, 1]).view(-1, *btask_v.shape).unsqueeze(1)
            act = torch.unsqueeze(act, 3)

            # compute activation mu and sigma
            pre_last_mu = torch.add(torch.sum(act * repeated_last_weight_mean, dim = 2), repeated_last_bias_mean)
            pre_last_sigma = torch.sqrt(1e-9 + torch.add(torch.sum(act**2 * torch.exp(0.5*repeated_last_weight_logsigma)**2, dim = 2), torch.exp(0.5*repeated_last_bias_logsigma)**2))
            # sample activation from act_mu and act_sigma
            eps_last_sample = torch.normal(torch.zeros_like(pre_last_mu), torch.ones_like(pre_last_mu)).to(device = device)
            # perform LRT
            pre = torch.add(eps_last_sample * pre_last_sigma, pre_last_mu)
        else:
            eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout))).to(device = device)
            eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout))).to(device = device)

            # random sample weight from distribution -- reparameterisation trick
            weights = torch.add(eps_w * torch.exp(0.5*Wtask_v),Wtask_m)
            biases = torch.add(eps_b * torch.exp(0.5*btask_v), btask_m)
            act = torch.unsqueeze(act, 3)
            weights = torch.unsqueeze(weights, 1)
            pre = torch.add(torch.sum(act * weights, dim = 2), biases)
        return pre

    def _logpred(self, inputs, targets, task_idx):
        loss = torch.nn.CrossEntropyLoss()
        pred = self._prediction(inputs, task_idx, self.no_train_samples).view(-1,self.out_size)
        targets = targets.repeat([self.no_train_samples, 1]).view(-1)
        log_liks = -loss(pred, targets.type(torch.long))
        log_lik = log_liks.mean()
        return log_lik

    def _KL_term(self):
        kl = 0

        # conv
        for i, (kw_m, kb_m, kw_v, kb_v) in enumerate(zip(self.kern_weights_m, self.kern_bias_m, self.kern_weights_v, self.kern_bias_v)):
            # weight from current posterior - q_t(theta)
            m, v = kw_m, kw_v
            # weight from prev posteriors - q_t-1(theta)
            m0, v0 = self.prior_kern_weights_m[i], self.prior_kern_weights_v[i]

            const_term = -0.5 * np.prod(m.shape)
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            # weight from current posterior - q_t(theta)
            m, v = kb_m, kb_v
            # weight from prev posteriors - q_t-1(theta)
            m0, v0 = self.prior_kern_bias_m[i], self.prior_kern_bias_v[i]

            const_term = -0.5 * np.prod(m.shape)
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

        # theta_S
        for i in range(self.no_layers-1):
            din = self.size[i]
            dout = self.size[i+1]
            # weight from current posterior - q_t(theta)
            m, v = self.W_m[i], self.W_v[i]
            # weight from prev posterior - q_t-1(theta)
            m0, v0 = self.prior_W_m[i], self.prior_W_v[i]

            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            # bias from current posterior - q_t(theta)
            m, v = self.b_m[i], self.b_v[i]
            # bias from prev posterior - q_t-1(theta)
            m0, v0 = self.prior_b_m[i], self.prior_b_v[i]

            const_term = -0.5 * dout
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl +=  log_std_diff + mu_diff_term + const_term

        no_tasks = len(self.W_last_m)
        din = self.size[-2]
        dout = self.size[-1]

        # theta_Hs
        for i in range(no_tasks):
            m, v = self.W_last_m[i], self.W_last_v[i]
            m0, v0 = self.prior_W_last_m[i], self.prior_W_last_v[i]

            const_term = - 0.5 * dout * din
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            m, v = self.b_last_m[i], self.b_last_v[i]
            m0, v0 = self.prior_b_last_m[i], self.prior_b_last_v[i]

            const_term = -0.5 * dout
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term
        return kl

    def save_weights(self):
        ''' Save weights before training on the coreset before getting the test accuracy '''

        print("Saving weights before core set training")
        self.W_m_copy = [self.W_m[i].clone().detach().data for i in range(len(self.W_m))]
        self.W_v_copy = [self.W_v[i].clone().detach().data for i in range(len(self.W_v))]
        self.b_m_copy = [self.b_m[i].clone().detach().data for i in range(len(self.b_m))]
        self.b_v_copy = [self.b_v[i].clone().detach().data for i in range(len(self.b_v))]

        self.kern_weights_m_copy = [self.kern_weights_m[i].clone().detach().data for i in range(len(self.kern_weights_m))]
        self.kern_weights_v_copy = [self.kern_weights_v[i].clone().detach().data for i in range(len(self.kern_weights_v))]
        self.kern_bias_m_copy = [self.kern_bias_m[i].clone().detach().data for i in range(len(self.kern_bias_m))]
        self.kern_bias_v_copy = [self.kern_bias_v[i].clone().detach().data for i in range(len(self.kern_bias_v))]

        self.W_last_m_copy = [self.W_last_m[i].clone().detach().data for i in range(len(self.W_last_m))]
        self.W_last_v_copy = [self.W_last_v[i].clone().detach().data for i in range(len(self.W_last_v))]
        self.b_last_m_copy = [self.b_last_m[i].clone().detach().data for i in range(len(self.b_last_m))]
        self.b_last_v_copy = [self.b_last_v[i].clone().detach().data for i in range(len(self.b_last_v))]

        self.prior_W_m_copy = [self.prior_W_m[i].data for i in range(len(self.prior_W_m))]
        self.prior_W_v_copy = [self.prior_W_v[i].data for i in range(len(self.prior_W_v))]
        self.prior_b_m_copy = [self.prior_b_m[i].data for i in range(len(self.prior_b_m))]
        self.prior_b_v_copy = [self.prior_b_v[i].data for i in range(len(self.prior_b_v))]

        self.prior_kern_weights_m_copy = [self.prior_kern_weights_m[i].clone().detach().data for i in range(len(self.prior_kern_weights_m))]
        self.prior_kern_weights_v_copy = [self.prior_kern_weights_v[i].clone().detach().data for i in range(len(self.prior_kern_weights_v))]
        self.prior_kern_bias_m_copy = [self.prior_kern_bias_m[i].clone().detach().data for i in range(len(self.prior_kern_bias_m))]
        self.prior_kern_bias_v_copy = [self.prior_kern_bias_v[i].clone().detach().data for i in range(len(self.prior_kern_bias_v))]

        self.prior_W_last_m_copy = [self.prior_W_last_m[i].data for i in range(len(self.prior_W_last_m))]
        self.prior_W_last_v_copy = [self.prior_W_last_v[i].data for i in range(len(self.prior_W_last_v))]
        self.prior_b_last_m_copy = [self.prior_b_last_m[i].data for i in range(len(self.prior_b_last_m))]
        self.prior_b_last_v_copy = [self.prior_b_last_v[i].data for i in range(len(self.prior_b_last_v))]

        return

    def load_weights(self):
        ''' Re-load weights after getting the test accuracy '''

        print("Reloading previous weights after core set training")
        self.weights = []
        self.W_m = [self.W_m_copy[i].clone().detach().data for i in range(len(self.W_m))]
        self.W_v = [self.W_v_copy[i].clone().detach().data for i in range(len(self.W_v))]
        self.b_m = [self.b_m_copy[i].clone().detach().data for i in range(len(self.b_m))]
        self.b_v = [self.b_v_copy[i].clone().detach().data for i in range(len(self.b_v))]

        self.kern_weights_m = [self.kern_weights_m_copy[i].clone().detach().data for i in range(len(self.kern_weights_m))]
        self.kern_weights_v = [self.kern_weights_v_copy[i].clone().detach().data for i in range(len(self.kern_weights_v))]
        self.kern_bias_m = [self.kern_bias_m_copy[i].clone().detach().data for i in range(len(self.kern_bias_m))]
        self.kern_bias_v = [self.kern_bias_v_copy[i].clone().detach().data for i in range(len(self.kern_bias_v))]

        for i in range(len(self.W_m)):
            self.W_m[i].requires_grad = True
            self.W_v[i].requires_grad = True
            self.b_m[i].requires_grad = True
            self.b_v[i].requires_grad = True

        for i in range(len(self.kern_weights_m)):
            self.kern_weights_m[i].requires_grad = True
            self.kern_weights_v[i].requires_grad = True
            self.kern_bias_m[i].requires_grad = True
            self.kern_bias_v[i].requires_grad = True

        self.weights += self.W_m
        self.weights += self.W_v
        self.weights += self.b_m
        self.weights += self.b_v
        self.weights += self.kern_weights_m + self.kern_weights_v + self.kern_bias_m + self.kern_bias_v


        self.W_last_m = [self.W_last_m_copy[i].clone().detach().data for i in range(len(self.W_last_m))]
        self.W_last_v = [self.W_last_v_copy[i].clone().detach().data for i in range(len(self.W_last_v))]
        self.b_last_m = [self.b_last_m_copy[i].clone().detach().data for i in range(len(self.b_last_m))]
        self.b_last_v = [self.b_last_v_copy[i].clone().detach().data for i in range(len(self.b_last_v))]

        for i in range(len(self.W_last_m)):
            self.W_last_m[i].requires_grad = True
            self.W_last_v[i].requires_grad = True
            self.b_last_m[i].requires_grad = True
            self.b_last_v[i].requires_grad = True

        self.weights += self.W_last_m
        self.weights += self.W_last_v
        self.weights += self.b_last_m
        self.weights += self.b_last_v

        self.optimizer = optim.Adam(self.weights, lr=self.learning_rate)
        self.prior_W_m = [self.prior_W_m_copy[i].data for i in range(len(self.prior_W_m))]
        self.prior_W_v = [self.prior_W_v_copy[i].data for i in range(len(self.prior_W_v))]
        self.prior_b_m = [self.prior_b_m_copy[i].data for i in range(len(self.prior_b_m))]
        self.prior_b_v = [self.prior_b_v_copy[i].data for i in range(len(self.prior_b_v))]

        self.prior_kern_weights_m = [self.prior_kern_weights_m_copy[i].data for i in range(len(self.prior_kern_weights_m))]
        self.prior_kern_weights_v = [self.prior_kern_weights_v_copy[i].data for i in range(len(self.prior_kern_weights_v))]
        self.prior_kern_bias_m = [self.prior_kern_bias_m_copy[i].data for i in range(len(self.prior_kern_bias_m))]
        self.prior_kern_bias_v = [self.prior_kern_bias_v_copy[i].data for i in range(len(self.prior_kern_bias_v))]

        self.prior_W_last_m = [self.prior_W_last_m_copy[i].data for i in range(len(self.prior_W_last_m))]
        self.prior_W_last_v = [self.prior_W_last_v_copy[i].data for i in range(len(self.prior_W_last_v))]
        self.prior_b_last_m = [self.prior_b_last_m_copy[i].data for i in range(len(self.prior_b_last_m))]
        self.prior_b_last_v = [self.prior_b_last_v_copy[i].data for i in range(len(self.prior_b_last_v))]

        return

    def clean_copy_weights(self):
        self.W_m_copy, self.W_v_copy, self.b_m_copy, self.b_v_copy = None, None, None, None
        self.W_last_m_copy, self.W_last_v_copy, self.b_last_m_copy, self.b_last_v_copy = None, None, None, None
        self.prior_W_m_copy, self.prior_W_v_copy, self.prior_b_m_copy, self.prior_b_v_copy = None, None, None, None
        self.prior_W_last_m_copy, self.prior_W_last_v_copy, self.prior_b_last_m_copy, self.prior_b_last_v_copy = None, None, None, None
        self.kern_weights_m_copy, self.kern_weights_v_copy, self.kern_bias_m_copy, self.kern_bias_v_copy = None, None, None, None
        self.prior_kern_weights_m_copy, self.prior_kern_weights_v_copy, self.prior_kern_bias_m_copy, self.prior_kern_bias_v_copy = None, None, None, None

    def create_head(self):
        ''''Create new head when a new task is detected'''
        print("creating a new head")
        din = self.size[-2]
        dout = self.size[-1]

        W_m= truncated_normal([din, dout], stddev=0.1, variable=True)
        b_m= truncated_normal([dout], stddev=0.1, variable=True)
        W_v = init_tensor(-6.0,  dout = dout, din = din, variable= True)
        b_v = init_tensor(-6.0,  dout = dout, variable= True)

        self.W_last_m.append(W_m)
        self.W_last_v.append(W_v)
        self.b_last_m.append(b_m)
        self.b_last_v.append(b_v)


        W_m_p = torch.zeros([din, dout]).to(device = device)
        b_m_p = torch.zeros([dout]).to(device = device)
        W_v_p =  init_tensor(1,  dout = dout, din = din)
        b_v_p = init_tensor(1, dout = dout)

        self.prior_W_last_m.append(W_m_p)
        self.prior_W_last_v.append(W_v_p)
        self.prior_b_last_m.append(b_m_p)
        self.prior_b_last_v.append(b_v_p)
        # update weights list to contain shared plus task-specific weights and biases
        self.weights = []
        self.weights += self.W_m
        self.weights += self.W_v
        self.weights += self.b_m
        self.weights += self.b_v
        self.weights += self.W_last_m
        self.weights += self.W_last_v
        self.weights += self.b_last_m
        self.weights += self.b_last_v
        self.weights += self.kern_weights_m
        self.weights += self.kern_weights_v
        self.weights += self.kern_bias_m
        self.weights += self.kern_bias_v        
        self.optimizer = optim.Adam(self.weights, lr=self.learning_rate)

        return


    def init_first_head(self, prev_means):
        ''''When the MFVI_NN is instantiated, we initialize weights with those of the Vanilla NN'''
        print("initializing first head")
        din = self.size[-2]
        dout = self.size[-1]
        self.prior_W_last_m = [torch.zeros([din, dout]).to(device = device)]
        self.prior_b_last_m = [torch.zeros([dout]).to(device = device)]
        self.prior_W_last_v =  [init_tensor(1,  dout = dout, din = din)]
        self.prior_b_last_v = [init_tensor(1, dout = dout)]

        # initialising the first task head
        W_last_m = prev_means["LastWeights"][0].detach().data
        W_last_m.requires_grad = True
        self.W_last_m = [W_last_m]
        self.W_last_v = [init_tensor(-6.0,  dout = dout, din = din, variable= True)]


        b_last_m = prev_means["LastBias"][0].detach().data
        b_last_m.requires_grad = True
        self.b_last_m = [b_last_m]
        self.b_last_v = [init_tensor(-6.0, dout = dout, variable= True)]

        return

    def create_weights(self, in_dim, hidden_size, out_dim, prev_means):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)

        no_layers = len(hidden_size) - 1
        W_m = []
        b_m = []
        W_v = []
        b_v = []

        for i in range(no_layers-1):
            din = hidden_size[i]
            dout = hidden_size[i+1]
            if prev_means is not None:
                # new prior means are old posteriors means from prev task
                W_m_i = prev_means["DenseWeights"][i].detach().data
                W_m_i.requires_grad = True
                bi_m_i = prev_means["DenseBias"][i].detach().data
                bi_m_i.requires_grad = True
            else:
            # This is same as prior weights - initialise with truncated normals
            #Initializiation values of means
                assert False
                W_m_i= truncated_normal([din, dout], stddev=0.1, variable=True)
                bi_m_i= truncated_normal([dout], stddev=0.1, variable=True)

            #Initializiation values of variances
            # how are variances negative?
            W_v_i = init_tensor(-6.0,  dout = dout, din = din, variable = True)
            bi_v_i = init_tensor(-6.0,  dout = dout, variable = True)

            #Append to list weights
            W_m.append(W_m_i)
            b_m.append(bi_m_i)
            W_v.append(W_v_i)
            b_v.append(bi_v_i)
        kern_w_m = prev_means["KernWeights"]
        kern_b_m = prev_means["KernBias"]
        make_ten = lambda ref:[init_tensor(-6.0, ten_like=ten, variable=True) for ten in ref]
        kern_w_v = make_ten(kern_w_m)
        kern_b_v = make_ten(kern_b_m)
        # return means and variances
        return [W_m, b_m], [W_v, b_v], [kern_w_m, kern_b_m], [kern_w_v, kern_b_v], hidden_size

    def create_prior(self, in_dim, hidden_size, out_dim, initial_mean = 0, initial_variance = 1):

        no_layers = len(hidden_size) - 1
        W_m = []
        b_m = []

        W_v = []
        b_v = []

        for i in range(no_layers - 1):
            din = hidden_size[i]
            dout = hidden_size[i + 1]

            # Initializiation values of means
            W_m_val = initial_mean + torch.zeros([din, dout]).to(device = device)
            bi_m_val = initial_mean + torch.zeros([dout]).to(device = device)

            # Initializiation values of variances
            W_v_val = initial_variance * init_tensor(1,  dout = dout, din = din )
            bi_v_val =  initial_variance * init_tensor(1,  dout = dout)

            # Append to list weights
            W_m.append(W_m_val)
            b_m.append(bi_m_val)
            W_v.append(W_v_val)
            b_v.append(bi_v_val)

        kern_w_m = [init_tensor(0,  ten_like=ten) for ten in self.kern_weights_m]
        kern_b_m = [init_tensor(0,  ten_like=ten) for ten in self.kern_bias_m]
        kern_w_v = [init_tensor(.01,  ten_like=ten) for ten in self.kern_weights_v]
        kern_b_v = [init_tensor(.01,  ten_like=ten) for ten in self.kern_bias_v]
        return [W_m, b_m], [W_v, b_v], [kern_w_m, kern_b_m], [kern_w_v, kern_b_v]


    def update_prior(self):
        print("updating prior...")
        # update new prior to be old posterior weights and biases means and vars
        for i in range(len(self.W_m)):
            self.prior_W_m[i].data.copy_(self.W_m[i].clone().detach().data)
            self.prior_b_m[i].data.copy_(self.b_m[i].clone().detach().data)
            self.prior_W_v[i].data.copy_(torch.exp(self.W_v[i].clone().detach().data))
            self.prior_b_v[i].data.copy_(torch.exp(self.b_v[i].clone().detach().data))

        for i in range(len(self.kern_weights_m)):
            self.prior_kern_weights_m[i].data.copy_(self.kern_weights_m[i].clone().detach().data)
            self.prior_kern_weights_v[i].data.copy_(torch.exp(self.kern_weights_v[i].clone().detach().data))
            self.prior_kern_bias_m[i].data.copy_(self.kern_bias_m[i].clone().detach().data)
            self.prior_kern_bias_v[i].data.copy_(torch.exp(self.kern_bias_v[i].clone().detach().data))

        length = len(self.W_last_m)

        for i in range(length):
            self.prior_W_last_m[i].data.copy_(self.W_last_m[i].clone().detach().data)
            self.prior_b_last_m[i].data.copy_(self.b_last_m[i].clone().detach().data)
            self.prior_W_last_v[i].data.copy_(torch.exp(self.W_last_v[i].clone().detach().data))
            self.prior_b_last_v[i].data.copy_(torch.exp(self.b_last_v[i].clone().detach().data))

        return
