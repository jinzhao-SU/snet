import torch
import torch.nn as nn
import torchvision


class SubNet(nn.Module):
    def __init__(self):
        super(SubNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(10000, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 4096)
        self.fc4 = nn.Linear(4096,10000)

        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(4096)

    def init_parameters(self):
        torch.nn.init.zeros_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc4.weight)

    def _subnet_forward(self, x):
        x = self.fc1(x)
        # x = torch.squeeze(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = x.view(-1, 100, 100)
        x = torch.sigmoid(x)
        return x

    # def _embedding(self, input, time_stamp = None):
    #     task_md = torch.zeros([input.shape[0], 100, 100])
    #     for i in range(input.shape[0]):
    #         x1 = int(input[i][0])
    #         y1 = int(input[i][1])
    #         x2 = int(input[i][2])
    #         y2 = int(input[i][3])
    #         if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
    #             continue
    #         else:
    #             task_md[i][x1][y1] = 1.00
    #             task_md[i][x2][y2] = 1.00
    #     return task_md


    def forward(self, task_embedded, time_stamp = None):
        #task_embedded = self._embedding(x_image, time_stamp)
        #task_embedded = task_embedded.view(-1,10000).float()

        #print("input task_embedded shape", task_embedded.shape)
        #shape = task_embedded.shape
        # print("x shape", x_image.shape)
        # x_image = x_image.permute(0,)
        #x_image = torch.squeeze(x_image)
        x = self._subnet_forward(task_embedded)
        #print("xoutput", x.shape) #32 100 100
        return x

if __name__ == '__main__':
    net = SubNet()
    subx = torch.rand(15, 10000)
    output = net(subx)
    print(output.shape)
    #15*100*100