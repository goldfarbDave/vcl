import numpy as np
from utils.mobile_net_v2_vanilla import mobilenetv2_vanilla
from utils.mobile_net_v2_bayesian import mobilenetv2_bayesian
import utils.test  as test
from utils.multihead_models import Vanilla_NN, Vanilla_CNN, MFVI_NN, MFVI_CNN
from . import flags 
import utils.GAN as GAN
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try:
    from torchviz import make_dot, make_dot_from_trace
except ImportError:
    print("Torchviz was not found.")

def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True, gan_bol = False, is_toy=False, use_lrt=False):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    x_trainsets, y_trainsets = [], []
    gans = []
    all_acc = np.array([])

    for task_id in range(data_gen.max_iter):
        print('Current task: '+str(task_id))
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)
        x_trainsets.append(x_train)
        y_trainsets.append(y_train)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        if task_id == 0:
            print_graph_bol = False #set to True if you want to see the graph
            if(is_toy):
                ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0], learning_rate=0.005)
            else:
                ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
            # train for first task
            ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
            # updated weights of network after SGD on task 1 -- these are means of posterior distribution of weights after task 1 ==> new prior for task 2
            mf_weights = ml_model.get_weights()
            # use these weights to initialise weights of new task model
            mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], single_head = single_head, prev_means=mf_weights, LRT=use_lrt)

        if not gan_bol:
            if coreset_size > 0:
                x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)
            gans = None
        if print_graph_bol:
            #Just if you want to see the computational graph
            output_tensor = mf_model._KL_term() #mf_model.get_loss(torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device), task_id), params=params)
            print_graph(mf_model, output_tensor)
            print_graph_bol = False

        if gan_bol:
            gan_i = GAN.VGR(task_id)
            gan_i.train(x_train, y_train)
            gans.append(gan_i)
        mf_model.train(x_train, y_train, head, no_epochs, bsize)

        mf_model.update_prior()
        # Save weights before test (and last-minute training on coreset
        mf_model.save_weights()

        acc = test.get_scores(mf_model, x_trainsets, y_trainsets, x_testsets, y_testsets, no_epochs, single_head, x_coresets, y_coresets, batch_size, False,gans, is_toy=is_toy)
        
        all_acc = test.concatenate_results(acc, all_acc)

        mf_model.load_weights()
        mf_model.clean_copy_weights()


        if not single_head:
            mf_model.create_head()

    return all_acc

def run_vcl_cnn(input_dims, hidden_size, output_dims, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True, gan_bol = False, is_toy=False, use_lrt=False, is_cifar=False):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    x_trainsets, y_trainsets = [], []
    gans = []
    all_acc = np.array([])

    for task_id in range(data_gen.max_iter):
        print('Current task: '+str(task_id))
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)
        x_trainsets.append(x_train)
        y_trainsets.append(y_train)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        if task_id == 0:
            print_graph_bol = False #set to True if you want to see the graph
            if(is_toy):
                ml_model = Vanilla_CNN(input_dims, hidden_size, output_dims, x_train.shape[0], learning_rate=0.005, is_cifar=is_cifar)
            else:
                ml_model = Vanilla_CNN(input_dims, hidden_size, output_dims, x_train.shape[0],is_cifar=is_cifar)
            # train for first task
            ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
            # updated weights of network after SGD on task 1 -- these are means of posterior distribution of weights after task 1 ==> new prior for task 2
            mf_weights = ml_model.get_weights()
            # use these weights to initialise weights of new task model
            if(is_cifar):
                mf_model = MFVI_CNN(input_dims, hidden_size, output_dims, x_train.shape[0], single_head = single_head, prev_means=mf_weights, LRT=use_lrt, is_cifar=is_cifar, learning_rate=0.01)
            else:
                mf_model = MFVI_CNN(input_dims, hidden_size, output_dims, x_train.shape[0], single_head = single_head, prev_means=mf_weights, LRT=use_lrt, is_cifar=is_cifar)

        if not gan_bol:
            if coreset_size > 0:
                x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)
            gans = None
        if print_graph_bol:
            #Just if you want to see the computational graph
            output_tensor = mf_model._KL_term() #mf_model.get_loss(torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device), task_id), params=params)
            print_graph(mf_model, output_tensor)
            print_graph_bol = False

        if gan_bol:
            gan_i = GAN.VGR(task_id)
            gan_i.train(x_train, y_train)
            gans.append(gan_i)
        mf_model.train(x_train, y_train, head, no_epochs, bsize)

        mf_model.update_prior()
        # Save weights before test (and last-minute training on coreset
        mf_model.save_weights()

        acc = test.get_scores(mf_model, x_trainsets, y_trainsets, x_testsets, y_testsets, no_epochs, single_head, x_coresets, y_coresets, batch_size, False,gans, is_toy=is_toy)
        
        all_acc = test.concatenate_results(acc, all_acc)

        mf_model.load_weights()
        mf_model.clean_copy_weights()


        if not single_head:
            mf_model.create_head()

    return all_acc

def run_coreset_only(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    x_trainsets, y_trainsets = [], []
    all_acc = np.array([])

    for task_id in range(data_gen.max_iter):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)
        x_trainsets.append(x_train)
        y_trainsets.append(y_train)

        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        if task_id == 0:
            mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], single_head = single_head, prev_means=None)

        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)


        mf_model.save_weights()

        acc = test.get_scores(mf_model, x_trainsets, y_trainsets, x_testsets, y_testsets, no_epochs, single_head, x_coresets, y_coresets, batch_size, just_vanilla =False)

        all_acc = test.concatenate_results(acc, all_acc)

        mf_model.load_weights()
        mf_model.clean_copy_weights()

        if not single_head:
            mf_model.create_head()

    return all_acc

def run_vcl_cifar(no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True, gan_bol = False, use_lrt=False, device="cpu"):
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    x_trainsets, y_trainsets = [], []
    gans = []
    all_acc = np.array([])

    in_dim, out_dim = data_gen.get_dims()

    for task_id in range(data_gen.max_iter):
        print('Current task: '+str(task_id))
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)
        x_trainsets.append(x_train)
        y_trainsets.append(y_train)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        cur_acc = 0

        # Train network with maximum likelihood to initialize first model
        if task_id == 0:
            print_graph_bol = False #set to True if you want to see the graph
            ml_model = mobilenetv2_vanilla(device=device, num_classes=out_dim)
            ml_model.to(device=device)
            # train for first task
            ml_model.train(x_train, y_train, task_id, no_epochs, bsize)

            pred_means = []

            pred=torch.argmax(ml_model.prediction_prob(torch.Tensor(x_test).to(device=device), None), dim=1)
            y_labs = torch.Tensor(y_test).type(torch.LongTensor).to(device=device)
            print(pred.shape, y_labs.shape)

            # # # Loop over all batches
            # for i in range(len(x_test)//bsize):
            #     start_ind = i*bsize
            #     end_ind = np.min([(i+1)*bsize, len(x_test)])
            #     batch_x_test = torch.Tensor(x_test[start_ind:end_ind, :]).to(device = device)
            #     batch_y_test = torch.Tensor(y_test[start_ind:end_ind]).type(torch.LongTensor).to(device = device)
            #     pred = ml_model.prediction_prob(batch_x_test, head)
            #     # pred_mean = pred.mean(0)
            #     pred_means.extend(list(pred.detach().cpu().numpy()))
            #     # pred_y = torch.argmax(pred_mean, dim=0)
            #     # cur_acc += end_ind - start_ind-(pred_y - batch_y_test).nonzero().shape[0]

            print(sum(pred==y_labs)/len(y_labs))

            # cur_acc = float(cur_acc)
            # cur_acc /= len(x_test)
            # print(cur_acc)
            # acc.append(cur_acc)
            # print("Accuracy is {}".format(cur_acc))

            # updated weights of network after SGD on task 1 -- these are means of posterior distribution of weights after task 1 ==> new prior for task 2
            mf_weights = ml_model.get_weights_for_bayesian()
            # use these weights to initialise weights of new task model
            mf_model = mobilenetv2_bayesian(device=device, num_classes=out_dim, prev_means=mf_weights)
            mf_model.to(device=device)

        if not gan_bol:
            if coreset_size > 0:
                x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)
            gans = None
        if print_graph_bol:
            #Just if you want to see the computational graph
            output_tensor = mf_model._KL_term() #mf_model.get_loss(torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device), task_id), params=params)
            print_graph(mf_model, output_tensor)
            print_graph_bol = False

        if gan_bol:
            gan_i = GAN.VGR(task_id)
            gan_i.train(x_train, y_train)
            gans.append(gan_i)
        mf_model.train(x_train, y_train, head, no_epochs, bsize)

        for ind_test, x_test_ in enumerate(x_testsets):
            print('Task:'+str(ind_test))
            pred=torch.argmax(mf_model.prediction_prob(torch.Tensor(x_test_).to(device=device), head).squeeze(0), dim=1)
            y_labs = torch.Tensor(y_testsets[ind_test]).type(torch.LongTensor).to(device=device)
            print(pred.shape, y_labs.shape)

        # # # Loop over all batches
        # for i in range(len(x_test)//bsize):
        #     start_ind = i*bsize
        #     end_ind = np.min([(i+1)*bsize, len(x_test)])
        #     batch_x_test = torch.Tensor(x_test[start_ind:end_ind, :]).to(device = device)
        #     batch_y_test = torch.Tensor(y_test[start_ind:end_ind]).type(torch.LongTensor).to(device = device)
        #     pred = ml_model.prediction_prob(batch_x_test, head)
        #     # pred_mean = pred.mean(0)
        #     pred_means.extend(list(pred.detach().cpu().numpy()))
        #     # pred_y = torch.argmax(pred_mean, dim=0)
        #     # cur_acc += end_ind - start_ind-(pred_y - batch_y_test).nonzero().shape[0]

            print(sum(pred==y_labs)/len(y_labs))
            print(len(x_testsets), len(y_testsets))

        mf_model.update_prior()
        # Save weights before test (and last-minute training on coreset)
        mf_model.save_weights()

        # acc = test.get_scores(mf_model, x_trainsets, y_trainsets, x_testsets, y_testsets, no_epochs, single_head, x_coresets, y_coresets, batch_size, False,gans)
        
        # all_acc = test.concatenate_results(acc, all_acc)

        mf_model.load_weights()

        if not single_head:
            mf_model.create_head()

    return None
    # return all_acc

def print_graph(model, output):
    params = dict()
    for i in range(len(model.W_m)):
        params["W_m{}".format(i)] = model.W_m[i]
        params["W_v{}".format(i)] = model.W_v[i]
        params["b_m{}".format(i)] = model.b_m[i]
        params["b_v{}".format(i)] = model.b_v[i]
        params["prior_W_m".format(i)] = model.prior_W_m[i]
        params["prior_W_v".format(i)] = model.prior_W_v[i]
        params["prior_b_m".format(i)] = model.prior_b_m[i]
        params["prior_b_v".format(i)] = model.prior_b_v[i]

    for i in range(len(model.W_last_m)):
         params["W_last_m".format(i)] = model.W_last_m[i]
         params["W_last_v".format(i)] = model.W_last_v[i]
         params["b_last_m".format(i)] = model.b_last_m[i]
         params["b_last_v".format(i)] = model.b_last_v[i]
         params["prior_W_last_m".format(i)] = model.prior_W_last_m[i]
         params["prior_W_last_v".format(i)] = model.prior_W_last_v[i]
         params["prior_b_last_m".format(i)] = model.prior_b_last_m[i]
         params["prior_b_last_v".format(i)] = model.prior_b_last_v[i]
    dot = make_dot(output, params=params)
    dot.view()

    return
