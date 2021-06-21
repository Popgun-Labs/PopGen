from popgen import datasets
from torch.utils.data import DataLoader


def setup_loaders(dataset_class: str, data_opts: dict, loader_opts: dict):
    """
    :param dataset_class: name of dataset class. any class exported in data/__init__.py
    :param data_opts: `dataset` config sub-dictionary
    :param loader_opts: `loader` config sub-dictionary
    :return:
    """

    # initialise datasets
    dataset_class = getattr(datasets, dataset_class)
    train_opts = {**data_opts["both"], **data_opts["train"]}
    test_opts = {**data_opts["both"], **data_opts["test"]}
    train_dataset = dataset_class(**train_opts)
    test_dataset = dataset_class(**test_opts)

    # initialise loaders
    train_opts = {**loader_opts["both"], **loader_opts["train"]}
    test_opts = {**loader_opts["both"], **loader_opts["test"]}
    train_loader = DataLoader(train_dataset, **train_opts)
    test_loader = DataLoader(test_dataset, **test_opts)

    return train_loader, test_loader
