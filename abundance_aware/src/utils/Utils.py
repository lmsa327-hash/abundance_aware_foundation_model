from importlib.resources import files
from pathlib import Path

from abundance_aware.src.utils.Contants import RESOURCES_PACKAGE_NAME


def find_pkg_resource(path):
    resource = files(RESOURCES_PACKAGE_NAME).joinpath(path)

    if not resource.is_file():
        raise FileNotFoundError(
            "Resource {} not found, please check".format(path)
        )

    return resource
def get_pkg_resource_path(path):
    dir_path  = Path(RESOURCES_PACKAGE_NAME).joinpath(path)
    if not dir_path.is_file():
        raise FileNotFoundError(
            "Resource {} not found, please check".format(path)
        )
    return dir_path