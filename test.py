import numpy as np

task = np.random.rand(1,2,2,3)
result = np.zeros([2,2,3])
for i in range(task.shape[1]):
    # print("i shape", task[:,i,:].shape)
    # for j in range(2):
    temp = task[:, i, :, :].squeeze()
    print(temp.shape)
    result[0] += temp
    print(result)
