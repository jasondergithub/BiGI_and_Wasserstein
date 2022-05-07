import torch
import torch.nn as nn

class Mixup_discriminator(nn.Module):
    def __init__(self, opt):   
        self.opt = opt     
        self.conv1 = nn.Conv1d(1, 16, 3)
        self.conv2 = nn.Conv1d(1, 8, 3)
        self.relu = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(8 * (opt["hidden_dim"] - (3-1)*2), 1)
    
    def forward(self, mixup_feature):
        mixup_feature = mixup_feature.unsqueeze(1)
        output = self.conv1(mixup_feature)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = output.reshape(self.opt["hidden_dim"], -1)
        output = self.fc(output)
        return output