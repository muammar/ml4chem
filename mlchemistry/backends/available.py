import pkg_resources


def available_backends():
    """A function that reports available backends"""
    supported_backends = ['tensorflow', 'torch', 'torchvision', 'numpy']
    available_backends = []

    installed_packages = [d.project_name for d in pkg_resources.working_set]

    for backend in supported_backends:
        if backend in installed_packages:
            available_backends.append(backend)

    return available_backends
