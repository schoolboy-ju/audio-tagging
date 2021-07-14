import torch
from torch import nn


class ConvRNN(nn.Module):
    @property
    def num_classes(self):
        return self._num_classes

    def __init__(self,
                 num_classes: int,
                 num_rnn_layers: int = 3, ):
        super(ConvRNN, self).__init__()
        self._num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=5, padding=5 // 2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=5 // 2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1)),
        )

        rnn_input_dim = (256, 4, 401)
        channels, width, length = rnn_input_dim
        self.seq_len = length
        self.width = width
        self.num_rnn_layers = num_rnn_layers

        self.rnn = nn.RNN(input_size=width,
                          hidden_size=width,
                          num_layers=num_rnn_layers,
                          batch_first=True,
                          nonlinearity='tanh')

        self.linear = nn.Linear(width, out_features=num_classes)

    def forward(self, x):
        device = x.device
        x = self.cnn(x)
        x = x.view(-1, self.seq_len, self.width)

        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_rnn_layers, x.size(0), self.width).requires_grad_().to(device)

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        x, _ = self.rnn(x, h0.detach())
        out = self.linear(x)
        return out
