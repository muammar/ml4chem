import torch


class SE3Net(torch.nn.Module):
    """Rotational equivariant neural network


    Parameters
    ----------
    num_classes : int
    size : int
    activation : str
    """

    def __init__(self, num_classes, size, activation='relu'):
        super(SE3Net, self).__init__()

        features = [(1,), (2, 2, 2, 1), (4, 4, 4, 4), (6, 4, 4, 0), (64,)]

        self.num_features = len(features)

        num_radial = size // 2 + 1
        max_radius = size // 2

        radii = torch.linspace(0, max_radius, steps=num_radial,
                               dtype=torch.float64)

        activation = activation

        kwargs = {'radii': radii,
                  'activation': (torch.nn.functional.relu, torch.sigmoid)}

        layers = []

        for i in range(len(features) - 1):
            layers.append(PointGatedBlock(features[i], features[i+1],
                                          **kwargs))

        self.layers = torch.nn.ModuleList(layers)

        with torch_default_dtype(torch.float64):
            self.layers.extend([AvgSpacial(), torch.nn.Dropout(p=0.2),
                                torch.nn.Linear(64, num_classes)])

    def forward(self, inputs, difference_mat):
        output = inputs
        for i in range(self.num_features - 1):
            conv = self.layers[i]
            output = conv(output, difference_mat)

        for i in range(self.num_features - 1, len(self.layers)):
            layer = self.layers[i]
            output = layer(output)

        return output


class torch_default_dtype:
    def __init__(self, dtype):
        self.saved_dtype = None
        self.dtype = dtype

    def __enter__(self):
        self.saved_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_default_dtype(self.saved_dtype)


class AvgSpacial(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)
