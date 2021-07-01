from typing import Optional

from torch.utils.data import DataLoader

from popgen.setup.utils import import_pkg


def setup_loaders(dataset_class: str, data_opts: dict, loader_opts: dict, module: Optional[str] = None):
    """
    :param dataset_class: name of dataset class. any class exported in <module>/data/__init__.py
    :param data_opts: `dataset` config sub-dictionary
    :param loader_opts: `loader` config sub-dictionary
    :param module: string name of the module to import `datasets` from
    :return:
    """
    if module is not None:
        datasets = import_pkg(module, "datasets")
    else:
        from popgen import datasets

        print("Warning: No module supplied in `setup_loaders`. Defaulting to `popgen.datasets`")

    # initialise datasets
    dataset_class = getattr(datasets, dataset_class)
    both = data_opts.get("both", {})
    train = data_opts.get("train", {})
    test = data_opts.get("test", {})
    train_opts = {**both, **train}
    test_opts = {**both, **test}
    train_dataset = dataset_class(**train_opts)
    test_dataset = dataset_class(**test_opts)

    # initialise loaders
    both = loader_opts.get("both", {})
    train = loader_opts.get("train", {})
    test = loader_opts.get("test", {})
    train_opts = {**both, **train}
    test_opts = {**both, **test}
    train_loader = DataLoader(train_dataset, **train_opts)
    test_loader = DataLoader(test_dataset, **test_opts)

    return train_loader, test_loader
