import torch
import torch.nn.functional as F
import torch.nn

class RegressionTripleHidden(torch.nn.Module):
    def __init__(self, input_dim, output_dim, first_hidden_dim = 400, second_hidden_dim = 300, third_hidden_dim = 200, drop_out = 0):
        super(RegressionTripleHidden, self).__init__()
        self.input_dim = input_dim
        self.first_hidden_dim  = first_hidden_dim
        self.second_hidden_dim  = second_hidden_dim
        self.third_hidden_dim  = third_hidden_dim
        self.output_dim = output_dim
        self.dpin = torch.nn.Dropout(drop_out)

        self.fc1 = torch.nn.Linear(self.input_dim, self.first_hidden_dim)
        self.fc1_bn = torch.nn.BatchNorm1d(self.first_hidden_dim)

        self.fc2 = torch.nn.Linear(self.first_hidden_dim, self.second_hidden_dim)
        self.fc2_bn = torch.nn.BatchNorm1d(self.second_hidden_dim)

        self.fc3 = torch.nn.Linear(self.second_hidden_dim, self.third_hidden_dim)
        self.fc3_bn = torch.nn.BatchNorm1d(self.third_hidden_dim)

        self.fc4 = torch.nn.Linear(self.third_hidden_dim, self.output_dim)

    def forward(self, x):
        hidden1 = self.fc1_bn(F.relu((self.fc1(self.dpin(x)))))
        hidden2 = self.fc2_bn(F.relu(self.fc2(hidden1)))
        hidden3 = self.fc3_bn(F.relu(self.fc3(hidden2)))
        output = F.normalize(self.fc4(hidden3), dim = 1)
        return output
