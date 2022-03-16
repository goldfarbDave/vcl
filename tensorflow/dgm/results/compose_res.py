import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
sns.set_theme()

with open('mnist_onlinevi_K10.pkl', 'rb') as f:
    ll_mnist = pickle.load(f)

with open('mnist_onlinevi_K10_gen_class.pkl', 'rb') as f:
    kl_mnist = pickle.load(f)

with open('notmnist_onlinevi_K10.pkl', 'rb') as f:
    ll_notmnist = pickle.load(f)

with open('notmnist_onlinevi_K10_gen_class.pkl', 'rb') as f:
    kl_notmnist = pickle.load(f)

with open('mnist-res-dict.pkl', 'rb') as f:
    ll_dict_mnist = pickle.load(f)

with open('notmnist-res-dict.pkl', 'rb') as f:
    ll_dict_notmnist = pickle.load(f)

with open('mnist-res-dict-kl.pkl', 'rb') as f:
    kl_dict_mnist = pickle.load(f)

with open('notmnist-res-dict-kl.pkl', 'rb') as f:
    kl_dict_notmnist = pickle.load(f)

# with sns.axes_style(rc={'axes.facecolor': '4bacc620'}):
#     plt.figure()
#     for task_idx in range(10):
#         plots = []
#         for sel_model in range(task_idx+1, 11):
#             plots.append(ll_dict_mnist[sel_model][task_idx][0])
#         plt.plot(np.linspace(10-len(plots)+1, 10, len(plots)), plots, 'o-', label=''+str(task_idx))
# plt.xlabel('Task tested on')
# plt.ylabel('Log-Likelihood')
# plt.legend(loc='best')
# plt.savefig('all-tasks-ll-mnist.png')

# with sns.axes_style(rc={'axes.facecolor': '4bacc620'}):
#     plt.figure()
#     for task_idx in range(10):
#         plots = []
#         for sel_model in range(task_idx+1, 11):
#             plots.append(ll_dict_notmnist[sel_model][task_idx][0])
#         plt.plot(np.linspace(10-len(plots)+1, 10, len(plots)), plots, 'o-', label=''+str(task_idx))
# plt.xlabel('Task tested on')
# plt.ylabel('Log-Likelihood')
# plt.legend(loc='best')
# plt.savefig('all-tasks-ll-notmnist.png')

with sns.axes_style(rc={'axes.facecolor': '4bacc620'}):
    plt.figure()
    for task_idx in range(10):
        plots = []
        for sel_model in range(task_idx+1, 11):
            plots.append(kl_dict_mnist[sel_model][task_idx][0])
        plt.plot(np.linspace(10-len(plots)+1, 10, len(plots)), plots, 'o-', label=''+str(task_idx))
plt.xlabel('Task tested on')
plt.ylabel('Classifier Uncertainty')
plt.legend(loc='best')
plt.savefig('all-tasks-kl-mnist.png')

with sns.axes_style(rc={'axes.facecolor': '4bacc620'}):
    plt.figure()
    for task_idx in range(10):
        plots = []
        for sel_model in range(task_idx+1, 11):
            plots.append(kl_dict_notmnist[sel_model][task_idx][0])
        plt.plot(np.linspace(10-len(plots)+1, 10, len(plots)), plots, 'o-', label=''+str(task_idx))
plt.xlabel('Task tested on')
plt.ylabel('Classifier Uncertainty')
plt.legend(loc='best')
plt.savefig('all-tasks-kl-notmnist.png')



# ll_mnist_means = [ll_mnist_[0][0] for ll_mnist_ in ll_mnist]
# ll_mnist_stds = [ll_mnist_[0][1] for ll_mnist_ in ll_mnist]

# kl_mnist_means = [kl_mnist_[0][0] for kl_mnist_ in kl_mnist]
# kl_mnist_stds = [kl_mnist_[0][1] for kl_mnist_ in kl_mnist]

# ll_notmnist_means = [ll_notmnist_[0][0] for ll_notmnist_ in ll_notmnist]
# ll_notmnist_stds = [ll_notmnist_[0][1] for ll_notmnist_ in ll_notmnist]

# kl_notmnist_means = [kl_notmnist_[0][0] for kl_notmnist_ in kl_notmnist]
# kl_notmnist_stds = [kl_notmnist_[0][1] for kl_notmnist_ in kl_notmnist]

# plt.figure()
# plt.plot(ll_mnist_means, 'o-')
# plt.savefig('test-ll_mnist.png')

# plt.figure()
# plt.plot(kl_mnist_means, 'o-')
# plt.savefig('test-kl_mnist.png')

# plt.figure()
# plt.plot(ll_notmnist_means, 'o-')
# plt.savefig('test-ll_notmnist.png')

# plt.figure()
# plt.plot(kl_notmnist_means, 'o-')
# plt.savefig('test-kl_notmnist.png')