from typing import Optional, Tuple

from torch.utils.data import DataLoader

from popgen.setup.utils import import_pkg


def setup_loaders(
    dataset_class: str, data_opts: dict, loader_opts: dict, module: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
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

        print("Warning: No module name provided to `setup_loaders`. Defaulting to `popgen.datasets`")

    # get dataset options
    both = data_opts.get("both", {})
    train = data_opts.get("train", {})
    test = data_opts.get("test", {})

    # initialise datasets
    dataset_class = getattr(datasets, dataset_class)
    train_dataset = dataset_class(**{**both, **train})
    test_dataset = dataset_class(**{**both, **test})

    # loader settings
    both = loader_opts.get("both", {})
    train = loader_opts.get("train", {})
    test = loader_opts.get("test", {})

    # account for custom `collate_fn`
    if hasattr(train_dataset, "get_collate_fn"):
        train["collate_fn"] = train_dataset.get_collate_fn()
        test["collate_fn"] = test_dataset.get_collate_fn()

    # account for custom batch sampler
    if hasattr(train_dataset, "get_batch_sampler"):
        train["batch_sampler"] = train_dataset.get_batch_sampler()
        test["batch_sampler"] = test_dataset.get_batch_sampler()

        # remove mutually exclusive args
        for opts in (both, train, test):
            opts.pop("batch_size", None)
            opts.pop("shuffle", None)
            opts.pop("sampler", None)
            opts.pop("drop_last", None)

    # initialise loaders
    train_loader = DataLoader(train_dataset, **{**both, **train})
    test_loader = DataLoader(test_dataset, **{**both, **test})

    return train_loader, test_loader
