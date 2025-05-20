# ------------------------------------------------------------------------
# DeepConvLSTM model based on architecture suggested by Ordonez and Roggen 
# https://www.mdpi.com/1424-8220/16/1/115
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

from torch import nn


class DeepConvLSTM(nn.Module):
    """
    DeepConvLSTM model as described in "Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition" (https://doi.org/10.1145/3460421.3480419).
    Args:
    
    Args:
        channels: int
            Number of channels in the input data.
        classes: int
            Number of classes for classification.
        window_size: int
            Size of the input window.
        conv_kernels: int
            Number of convolutional kernels.
        conv_kernel_size: int
            Size of the convolutional kernels.
        lstm_units: int
            Number of LSTM units.
        lstm_layers: int
            Number of LSTM layers.
        dropout: float
            Dropout rate.
    """
    def __init__(self, channels, classes, window_size, conv_kernels=64, conv_kernel_size=5, lstm_units=128, lstm_layers=2, dropout=0.5):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.lstm = nn.LSTM(channels * conv_kernels, lstm_units, num_layers=lstm_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, classes)
        self.activation = nn.ReLU()
        self.final_seq_len = window_size - (conv_kernel_size - 1) * 4
        self.lstm_units = lstm_units
        self.classes = classes

    def forward(self, x):
        x = x.unsqueeze(1) # batch, 1, sequence, axes
        x = self.activation(self.conv1(x))# batch, kernels, sequence, axes
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, _ = self.lstm(x)
        x = x[-1, :, :]
        x = x.view(-1, self.lstm_units)
        x = self.dropout(x)
        return self.classifier(x)
        