import pytest
import torch

from models.crnn import ConvolutionalNeuralNetwork


@pytest.fixture()
def sample_data():
    return torch.randn((1, 1, 19, 401))


def test_models(sample_data):
    model = ConvolutionalNeuralNetwork(num_classes=8)
    out = model(sample_data)
