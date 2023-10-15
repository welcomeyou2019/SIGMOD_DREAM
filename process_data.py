import numpy as np
file = ['MNIST_cuda:0.txt','MNIST_cuda:1.txt','MNIST_cuda:2.txt','MNIST_cuda:3.txt']
total = []
for i in file:
    result = []
    with open(i, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            result.append(float(line))
            # print(line)
            # if 'MIX' in line:
            #     contant = line.split(':')
            #     result.append(float(contant[-1][:-1]))
    print(len(result))
    total.append(result)
total = np.stack(total, axis=0)
acc_mean = np.mean(total,axis=0)
best_epoch = acc_mean.argmax().item()
print(f'---------------- Best Epoch: {best_epoch} ----------------')
print('Mean: {:7f}, Std: {:7f}'.format(total[:, best_epoch].mean(), total[:, best_epoch].std()), flush=True)

# print(total)