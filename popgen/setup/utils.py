from importlib import import_module
from types import ModuleType


def import_pkg(base: str, sub: str) -> ModuleType:
    """
    Import the package located at base.str
    :param base:
    :param sub:
    :return:
    """
    pkg_path = f"{base}.{sub}"
    try:
        pkg = import_module(pkg_path)
    except ModuleNotFoundError as _e:
        raise Exception(
            f"Cannot find '{pkg_path}' on the current path. Check that the module '{base}' is on "
            f"'sys.path', and that it contains the sub-module '{sub}'."
        )
    return pkg
