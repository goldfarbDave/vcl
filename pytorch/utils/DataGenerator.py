import numpy as np
import gzip
import pickle as cp
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
'''
This class replicates the dataset on page 17 of VCL paper. It has two tasks and each task has 2 classes.
'''
class ToyDatasetGenerator():
    def __init__(self, num_samples_per_class):
        self.cur_task = 0
        # fix number of tasks to be 2
        self.max_iter = 2
        self.num_samples_per_class = num_samples_per_class
        # the below variables contains <num_tasks> arrays
        # self.train_data[i] contains the train data for task i
        # self.train_label[i] contains the train label for task i
        # self.test_data[i] contains the test data for task i
        # self.test_label[i] contains the test label for task i
        self.train_data, self.train_label, self.test_data, self.test_label = self.generate_data()

    def next_task(self):
        if(self.cur_task >= 2):
            raise Exception('Number of tasks exceeded!')
        else:
            self.cur_task += 1
            return np.asarray(self.train_data[self.cur_task-1]), np.asarray(self.train_label[self.cur_task-1]), np.asarray(self.test_data[self.cur_task-1]), np.asarray(self.test_label[self.cur_task-1])

    def get_dims(self):
        # Get data input and output dimensions
        return np.asarray(self.train_data[0]).shape[1], 2

    def generate_data(self):
        # FIX: updated global seed to fixed value from flags
        # fix seed to generate same dataset everytime
        # np.random.seed(1)

        # first task first class data = second task first class data
        # mean = [0, 0]
        # cov = [1 0; 0 1]
        samples_1_1 = np.random.multivariate_normal([0, 0], [[0.15, 0], [0, 0.15]], self.num_samples_per_class)
        samples_2_1 = samples_1_1

        # first task second class data
        # mean = [1.5, 0]
        # cov = [0.25 0; 0 2]
        samples_1_2 = np.random.multivariate_normal([1.5, 0], [[0.12, 0], [0, 1]], self.num_samples_per_class)

        # second task second class data
        # mean = [0, 1.5]
        # cov = [2 0; 0 0.25]
        samples_2_2 = np.random.multivariate_normal([0, 1.5], [[1, 0], [0, 0.12]], self.num_samples_per_class)

        # plot samples to vis
        # plt.figure()
        # plt.scatter(samples_1_1[:, 0], samples_1_1[:, 1], c='g')
        # plt.scatter(samples_1_2[:, 0], samples_1_2[:, 1], c='b')
        # plt.xlim(-3, 3)
        # plt.ylim(-3, 3)
        # plt.savefig('../data/toy/toy-task-1.png')

        # plt.figure()
        # plt.scatter(samples_2_1[:, 0], samples_2_1[:, 1], c='g')
        # plt.scatter(samples_2_2[:, 0], samples_2_2[:, 1], c='b')
        # plt.xlim(-3, 3)
        # plt.ylim(-3, 3)
        # plt.savefig('../data/toy/toy-task-2.png')

        # each task training set contains 160 samples -- 80 samples from each class
        # each task test set contains 40 samples -- 20 samples from each class
        return [list(samples_1_1[:80]) + list(samples_1_2[:80]), list(samples_2_1[:80]) + list(samples_2_2[:80])], [[0]*80 + [1]*80, [0]*80 + [1]*80], [list(samples_1_1[:20]) + list(samples_1_2[:20]), list(samples_2_1[:20]) +list(samples_2_2[:20])], [[0]*20 + [1]*20, [0]*20 + [1]*20]

class PermutedMnistGenerator():
    def __init__(self, max_iter=10):

        with gzip.open('data/mnist.pkl.gz', 'rb') as file:
            u = cp._Unpickler(file)
            u.encoding = 'latin1'
            p = u.load()
            train_set, valid_set, test_set = p

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        # number of tasks
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter)
            perm_inds = np.arange(self.X_train.shape[1])
            np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]
            next_y_train = self.Y_train

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]
            next_y_test = self.Y_test

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test


class SplitMnistGenerator():
    def __init__(self):
        with gzip.open('data/mnist.pkl.gz', 'rb') as file:
            u = cp._Unpickler(file)
            u.encoding = 'latin1'
            p = u.load()
            train_set, valid_set, test_set = p

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0],1 )), np.zeros((train_1_id.shape[0],1 )))).squeeze(-1)

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0],1 )), np.zeros((test_1_id.shape[0], 1)))).squeeze(-1)

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class SplitCifarGenerator():
    def __init__(self):
        with open("data/cifar-10.pkl", 'rb') as f:
            dat = pickle.load(f)
        split = lambda ar, idx: (ar[:idx], ar[idx:])
        tt_split = lambda ar: split(ar, 50000)
        self.data = dat['data']
        self.labels = dat['labels']
        self.train_data, self.test_data = tt_split(self.data)
        self.train_labels, self.test_labels = tt_split(self.labels)
        self.tasks = list(zip(range(0,10,2), range(1,10,2)))
        self.cur_iter = 0
        self.max_iter = len(self.tasks)

    def get_non_split(self):
        return self.train_data, self.train_labels, self.test_data, self.test_labels

    def get_dims(self):
        return self.data.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception("Number of tasks exceeded!")
        task = self.tasks[self.cur_iter]
        get_task_idxs = lambda ref: [ref == t for t in task]
        #get_labels = lambda label_ar: np.hstack([np.ones(int(label_ar.shape[0]/self.max_iter/2))*t for t in task])
        label_len = lambda ar: int(ar.shape[0]/self.max_iter/2)
        get_labels = lambda ar: np.hstack(([0]*label_len(ar), [1]*label_len(ar)))
        get_data = lambda data_ar, label_ar: np.vstack([data_ar[t] for t in get_task_idxs(label_ar)])
        get_data_labels = lambda data_ar, label_ar: (get_data(data_ar, label_ar), get_labels(label_ar))
        x_train, y_train = get_data_labels(self.train_data, self.train_labels)
        x_test, y_test = get_data_labels(self.test_data, self.test_labels)
        self.cur_iter += 1
        return x_train, y_train, x_test, y_test

def cifar_format_to_rgb(cifar_ar):
    # Cifar format is flattened(Channel,Row,column). We want Row, Column, Channel
    return cifar_ar.reshape(3,32,32).transpose(1,2,0)


if __name__ == '__main__':
    data_gen = ToyDatasetGenerator(100)
    for i in range(2):
        train_d, train_l, test_d, test_l = data_gen.next_task()
        print(train_d.shape, train_l.shape, test_d.shape, test_l.shape)
