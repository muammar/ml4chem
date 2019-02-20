import pkg_resources


def available_backends():
    """A function that reports available backends"""

    supported_backends = ['tensorflow', 'torch', 'torchvision', 'numpy']
    available_backends = []

    installed_packages = [p.project_name for p in pkg_resources.working_set]

    for backend in supported_backends:
        if backend in installed_packages:
            available_backends.append(backend)

    return available_backends

def cuda_is_available(backend):
    """Check if cuda is avaible for the current backend

    Parameters
    ----------
    backend : obj
        A backend object

    Returns
    -------
    cuda_available : bool
        Whether or not cuda is there.
    """

    cuda_supported_backends = ['torch', 'tensorflow']
    backend_name = backend.__name__

    if backend_name in cuda_supported_backends:
        if backend.name == 'torch' or 'torchvision':
            cuda_available = backend.cuda.is_available()
    else:
        cuda_available = False

    return cuda_available
