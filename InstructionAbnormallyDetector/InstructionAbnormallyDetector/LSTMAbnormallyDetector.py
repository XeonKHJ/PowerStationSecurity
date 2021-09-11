import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class LSTMAbnormallyDetector(nn.Module):
    def __init__(self, input_size, hidden_size, feature_size, output_size):
        this.lstm = nn.LSTM()
