from importlib.resources import files


def find_pkg_resource(path):
    resource = files("mgm").joinpath(path)

    if not resource.is_file():
        raise FileNotFoundError(
            "Resource {} not found, please check".format(path)
        )

    return resource