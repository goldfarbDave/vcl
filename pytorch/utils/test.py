import numpy as np
import matplotlib
matplotlib.use('agg')
from . import flags 
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def merge_coresets(x_coresets, y_coresets):
    merged_x, merged_y = x_coresets[0], y_coresets[0]
    for i in range(1, len(x_coresets)):
        merged_x = np.vstack((merged_x, x_coresets[i]))
        merged_y = np.hstack((merged_y, y_coresets[i]))
    return merged_x, merged_y


def get_coreset(x_coresets, y_coresets, single_head, coreset_size = 5000, gans = None, task_id=0):
    if gans is not None:
        if single_head:
            merged_x, merged_y = gans[0].generate_samples(coreset_size, task_id)
            for i in range(1, len(gans)):
                new_x, new_y = gans[i].generate_samples(coreset_size, task_id)
                merged_x = np.vstack((merged_x,new_x))
                merged_y = np.hstack((merged_y,new_y))
            return merged_x, merged_y
        else:
            return gans.generate_samples(coreset_size, task_id)[:coreset_size]
    else:
        if single_head:
            return merge_coresets(x_coresets, y_coresets)
        else:
            return x_coresets, y_coresets


def get_scores(model, x_trainsets, y_trainsets, x_testsets, y_testsets, no_epochs, single_head,  x_coresets, y_coresets, batch_size=None, just_vanilla = False, gans = None, is_toy=False):

    task_num = len(x_trainsets)
    acc = []
    if single_head:
        if len(x_coresets) > 0 or gans is not None:
            x_train, y_train = get_coreset(x_coresets, y_coresets, single_head, coreset_size = 6000, gans = gans, task_id=0)

            bsize = x_train.shape[0] if (batch_size is None) else batch_size
            x_train = torch.Tensor(x_train)
            y_train = torch.Tensor(y_train)
            model.train(x_train, y_train, 0, no_epochs, bsize)

    # this is only for the toy dataset visualisation -- probability contour plots
    if(is_toy):
        for i in range(len(x_trainsets)):
            head = 0 if single_head else i

            x_train, y_train = x_trainsets[i], y_trainsets[i]

            x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
            y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

            whole_space_data = np.stack((xx.ravel(),yy.ravel()), axis=-1)

            N = whole_space_data.shape[0]
            bsize = N if (batch_size is None) else batch_size
            total_batch = int(np.ceil(N * 1.0 / bsize))
            # Loop over all batches
            for j in range(total_batch):
                start_ind = j*bsize
                end_ind = np.min([(j+1)*bsize, N])
                batch_x_train = torch.Tensor(whole_space_data[start_ind:end_ind, :]).to(device = device)
                # these are model probabilities over different samples of weights from the posterior distribution
                pred = model.prediction_prob(batch_x_train, head)
                # this simply takes the mean over all the different outputs with respect to the weight samples
                if not just_vanilla:
                    pred_mean = pred.mean(0)
                else:
                    pred_mean = pred

                prob_ones = pred_mean[:, 0]

                half_curve = []
                for ind in range(len(prob_ones)):
                    if(prob_ones[ind]<=0.51 and prob_ones[ind]>=0.5):
                        half_curve.append(whole_space_data[ind])
                half_curve = np.asarray(half_curve)

                onethird_curve = []
                for ind in range(len(prob_ones)):
                    if(prob_ones[ind]<=0.34 and prob_ones[ind]>=0.3):
                        onethird_curve.append(whole_space_data[ind])
                onethird_curve = np.asarray(onethird_curve)

                ninety_curve = []
                for ind in range(len(prob_ones)):
                    if(prob_ones[ind]>=0.9):
                        ninety_curve.append(whole_space_data[ind])
                ninety_curve = np.asarray(ninety_curve)        

                fig1, ax2 = plt.subplots(constrained_layout=True)                
                cb = ax2.scatter(whole_space_data[:, 0], whole_space_data[:, 1], c=prob_ones.detach().cpu().numpy(), cmap='inferno')
                ax2.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
                ax2.set_xlabel('x1')
                ax2.set_ylabel('x2')

                custom_lines = [Line2D([0], [0], color='indigo', lw=4), Line2D([0], [0], color='yellow', lw=4)]
                ax2.legend(custom_lines, ['Class 0', 'Class 1'])
                fig1.colorbar(cb)
                plt.savefig('toy-vis-task-'+str(task_num)+str(i)+str(j)+'.png')

                plt.figure()
                plt.scatter(whole_space_data[:, 0], whole_space_data[:, 1], c=prob_ones.detach().cpu().numpy(), cmap='inferno')
                plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
                if(half_curve.shape[0]!=0):
                    plt.plot(half_curve[:, 0], half_curve[:, 1])
                if(onethird_curve.shape[0]!=0):
                    plt.plot(onethird_curve[:, 0], onethird_curve[:, 1])
                if(ninety_curve.shape[0]!=0):
                    plt.plot(ninety_curve[:, 0], ninety_curve[:, 1])
                plt.colorbar()    
                plt.savefig('toy-vis-task-with-contours-'+str(task_num)+str(i)+str(j)+'.png')

    for i in range(len(x_testsets)):
        if not single_head:
            if len(x_coresets)>0 or gans is not None:
                model.load_weights()
                gan_i = None
                if gans is not None:
                    gan_i = gans[i]
                    x_train, y_train = get_coreset(None, None, single_head, coreset_size = 6000, gans= gan_i, task_id=i)
                else:
                    x_train, y_train = get_coreset(x_coresets[i], y_coresets[i], single_head, coreset_size = 6000, gans= None, task_id=i)
                bsize = x_train.shape[0] if (batch_size is None) else batch_size
                x_train = torch.Tensor(x_train)
                y_train = torch.Tensor(y_train)
                model.train(x_train, y_train, i, no_epochs, bsize)

        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]
        N = x_test.shape[0]
        bsize = N if (batch_size is None) else batch_size
        cur_acc = 0
        total_batch = int(np.ceil(N * 1.0 / bsize))
        # Loop over all batches
        for i in range(total_batch):
            start_ind = i*bsize
            end_ind = np.min([(i+1)*bsize, N])
            batch_x_test = torch.Tensor(x_test[start_ind:end_ind, :]).to(device = device)
            batch_y_test = torch.Tensor(y_test[start_ind:end_ind]).type(torch.LongTensor).to(device = device)
            pred = model.prediction_prob(batch_x_test, head)
            if not just_vanilla:
                pred_mean = pred.mean(0)
            else:
                pred_mean = pred
            pred_y = torch.argmax(pred_mean, dim=1)
            cur_acc += end_ind - start_ind-(pred_y - batch_y_test).nonzero().shape[0]

        cur_acc = float(cur_acc)
        cur_acc /= N
        acc.append(cur_acc)
        print("Accuracy is {}".format(cur_acc))
    return acc

def concatenate_results(score, all_score):
    if all_score.size == 0:
        all_score = np.reshape(score, (1,-1))
    else:
        new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
        new_arr[:] = np.nan
        new_arr[:,:-1] = all_score
        all_score = np.vstack((new_arr, score))
    return all_score