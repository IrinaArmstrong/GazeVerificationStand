import torch
import torch.nn as nn


class Siamese(nn.Module):

    def __init__(self, input_size: int, hidden_size: int=64):
        super(Siamese, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, int(hidden_size // 2)),
            nn.PReLU(),
        )
        self.linear = nn.Sequential(nn.Linear(int(hidden_size // 2), int(hidden_size // 4)),
                                    nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(int(hidden_size // 4), 1))

    def forward_one(self, x):
        x = self.fc(x)
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out