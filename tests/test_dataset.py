import os

import pytest

from datamodules.chime_home import ChimeHomeDatamodule, ChimeDataset


@pytest.fixture
def dataset_root_path():
    return os.path.join(os.sep, 'home', 'joohyun', 'database')


def test_dataset_fetch(dataset_root_path):
    datamodule = ChimeHomeDatamodule(data_root_path=dataset_root_path)
    datamodule.fetch()


def test_dataset(dataset_root_path):
    ds = ChimeDataset(
        data_root_path=dataset_root_path,
        data_split='train',
        fold=1
    )
