import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import SGD
import torch.utils.data as Data
import numpy
import pandas as pd

train_data_set = pd.read_csv(r'C:\Users\18834\Desktop\Question 3\train_data.txt', sep='\t')
train_truth_set = pd.read_csv(r'C:\Users\18834\Desktop\Question 3\train_truth.txt', sep='\t')
test_data_set = pd.read_csv(r'C:\Users\18834\Desktop\Question 3\test_data.txt', sep='\t')
train_data_numpy = train_data_set.values
train_truth_numpy = train_truth_set.values
test_data_numpy = test_data_set.values
x = Variable(torch.from_numpy(train_data_numpy).float(), requires_grad=False)
y = Variable(torch.from_numpy(train_truth_numpy).float(), requires_grad=False)
test_x = torch.from_numpy(test_data_numpy).float()
train_data = Data.TensorDataset(x, y)
train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=0)


class MLPPregression(nn.Module):
    def __init__(self):
        super(MLPPregression, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 4)
        self.fc4 = nn.Linear(4, 4)
        self.predict = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        output = self.predict(x)
        return output[:, 0]


mlpreg = MLPPregression()
optimizer = SGD(mlpreg.parameters(), lr=0.001, weight_decay=0.0001)
loss_func = nn.MSELoss()
train_loss_all = []

for epoch in range(30):
    train_loss = 0
    train_num = 0
    for step, (b_x, b_y) in enumerate(train_loader):
        output = mlpreg(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * b_x.size(0)
        train_num += b_x.size(0)
    train_loss_all.append(train_loss / train_num)
print(train_loss_all)

# predict the test set
pre_y = mlpreg(test_x)
pre_y = pre_y.data.numpy()
with open(r"C:\Users\18834\Desktop\Question 3\test_predicted.txt", 'w', encoding='utf8') as f:
    f.write('y\t\n')
    for i in range(len(pre_y)):
        f.write(str(pre_y[i]))
        f.write("\t\n")
