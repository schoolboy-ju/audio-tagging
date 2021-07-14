import os

from datamodules.chime_home import ChimeHomeDatamodule, ChimeDataset
from models.crnn import ConvRNN


def train():
    ...


def valid():
    ...


def run():
    num_epochs = 100

    dataset_root_path = os.path.join(os.sep, 'home', 'joohyun', 'database')
    dm = ChimeHomeDatamodule(data_root_path=dataset_root_path)

    for fold in dm.folds:
        dataset = ChimeDataset()
        model = ConvRNN()

        for epoch in range(num_epochs):
            ...


if __name__ == '__main__':
    run()
