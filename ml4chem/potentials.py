import ase
import codecs
import copy
import json
import logging
import torch
from ase.calculators.calculator import Calculator
from ml4chem.backends.available import available_backends
from ml4chem.data.handler import Data
from ml4chem.data.serialization import dump, load
from ml4chem.utils import get_header_message, dynamic_import


logger = logging.getLogger()


class Potentials(Calculator, object):
    """Atomistic Machine Learning Potentials

    This class is highly inspired by the Atomistic Machine-Learning package
    (Amp).

    Parameters
    ----------
    features : object
        Atomic feature vectors (local chemical environments) from any of the
        features module.
    model : object
        Machine learning algorithm to build a model.
    path : str
        Path to save files.
    label : str
        Name of files. Default ml4chem.
    preprocessor : str
        Path to load sklearn preprocessor object. Useful when doing inference.
    batch_size : int
        Number of data points per batch to use for training. Default is None.
    """

    # This is needed by ASE
    implemented_properties = ["energy", "forces"]

    # This is a good way to make attributes available to the class. This can be
    # accessed as Potentials.attribute
    svm_models = ["KernelRidge", "GaussianProcess"]
    autoencoders = ["AutoEncoder", "VAE"]
    module_names = {
        "PytorchPotentials": "neuralnetwork",
        "PytorchIonicPotentials": "ionic",
        "RetentionTimes": "rt",
        "KernelRidge": "kernelridge",
        "GaussianProcess": "gaussian_process",
    }

    def __init__(
        self,
        features=None,
        model=None,
        path=None,
        label="ml4chem",
        atoms=None,
        ml4chem_path=None,
        preprocessor=None,
        batch_size=None,
    ):

        Calculator.__init__(self, label=label, atoms=atoms)
        self.features = features
        self.available_backends = available_backends()
        self.path = path
        self.label = label
        self.model = model
        self.ml4chem_path = ml4chem_path
        self.preprocessor = preprocessor
        self.batch_size = batch_size

        logger.info(get_header_message())

        self.reference_space = None

    @classmethod
    def load(Cls, model=None, params=None, preprocessor=None, **kwargs):
        """Load ML4Chem models

        Parameters
        ----------
        model : str
            The path to load the model from the .ml4c file for inference.
        params : srt
            The path to load .params file with users' inputs.
        preprocessor : str
            The path to load the file with the sklearn preprocessor object.
        """
        kwargs["ml4chem_path"] = model
        kwargs["preprocessor"] = preprocessor

        with open(params) as ml4chem_params:
            ml4chem_params = json.load(ml4chem_params)
            model_type = ml4chem_params["model"].get("type")

            model_params = ml4chem_params["model"]
            class_name = model_params["class_name"]
            module_name = Potentials.module_names[model_params["name"]]

            model_class = dynamic_import(
                class_name, "ml4chem.models", alt_name=module_name
            )

            delete = ["name", "type", "class_name"]
            for param in delete:
                # delete unneeded (key, value) pairs.
                del model_params[param]

            if model_type == "svm":

                weights = load(model)
                # TODO remove after de/serialization is fixed.
                weights = {key.decode("utf-8"): value for key, value in weights.items()}
                model_params.update({"weights": weights})
                model = model_class(**model_params)
            else:
                # Instantiate the model class
                model = model_class(**model_params)

        # Instantiation of fingerprint class
        fingerprint_params = ml4chem_params.get("features", None)

        if fingerprint_params is None:
            features = fingerprint_params
        else:
            name = fingerprint_params.get("name")
            del fingerprint_params["name"]

            features = dynamic_import(name, "ml4chem.features")
            features = features(**fingerprint_params)

        calc = Cls(features=features, model=model, **kwargs)

        return calc

    @staticmethod
    def save(model=None, features=None, path=None, label="ml4chem"):
        """Save a model

        Parameters
        ----------
        model : obj
            The model to be saved.
        features : obj
            Features object.
        path : str
            The path where to save the model.
        label : str
            Name of files. Default ml4chem.
        """

        if path is None:
            path = ""

        path += label

        if model is not None:
            model_name = model.name()
            if model_name in Potentials.svm_models:
                params = {"model": model.params}

                # Save model weights to file
                dump(model.weights, path + ".ml4c")
            else:
                # FIXME a global class to save params?
                params = {
                    "model": {
                        "name": model_name,
                        "class_name": model.__class__.__name__,
                        "hiddenlayers": model.hiddenlayers,
                        "activation": model.activation,
                        "type": "nn",
                        "input_dimension": model.input_dimension,
                    }
                }

                torch.save(model.state_dict(), path + ".ml4c")

                if model_name in Potentials.autoencoders:
                    output_dimension = {"output_dimension": model.output_dimension}
                    params["model"].update(output_dimension)
                    variant = {"variant": model.variant}
                    params["model"].update(variant)
                    one_for_all = {"one_for_all": model.one_for_all}
                    params["model"].update(one_for_all)
        else:
            params = {}

        if features is not None:
            # Adding features to .params json file.
            features = {"features": features.params}
            params.update(features)

        # Save parameters to file
        with open(path + ".params", "wb") as json_file:
            json.dump(
                params,
                codecs.getwriter("utf-8")(json_file),
                ensure_ascii=False,
                indent=4,
            )

    def train(
        self,
        training_set,
        epochs=100,
        lr=0.001,
        convergence=None,
        device="cpu",
        optimizer=(None, None),
        lossfxn=None,
        regularization=0.0,
        batch_size=None,
        **kwargs
    ):
        """Method to train models

        Parameters
        ----------
        training_set : object, list
            List containing the training set.
        epochs : int
            Number of full training cycles.
        lr : float
            Learning rate.
        convergence : dict
            Instead of using epochs, users can set a convergence criterion.
        device : str
            Calculation can be run in the cpu or cuda (gpu).
        optimizer : tuple
            The optimizer is a tuple with the structure:

                >>> ('adam', {'lr': float, 'weight_decay'=float})

        lossfxn : object
            A loss function object.
        regularization : float
            This is the L2 regularization. It is not the same as weight decay.
        batch_size : int
            Number of data points per batch to use for training. Default is
            None.
        """

        purpose = "training"
        data_handler = Data(training_set, purpose=purpose)
        # Raw input and targets aka X, y
        training_set, targets = data_handler.get_data(purpose=purpose)

        # Now let's train
        # SVM models
        if self.model.name() in Potentials.svm_models:
            # Mapping raw positions into a feature space aka X
            feature_space, reference_features = self.features.calculate(
                training_set, data=data_handler, purpose=purpose, svm=True
            )
            self.model.prepare_model(
                feature_space, reference_features, data=data_handler
            )

            self.model.train(feature_space, targets)
        else:
            # Mapping raw positions into a feature space aka X

            feature_space = self.features.calculate(
                training_set, data=data_handler, purpose=purpose, svm=False
            )

            # Fixed fingerprint dimension
            input_dimension = len(list(feature_space.values())[0][0][-1])
            self.model.prepare_model(input_dimension, data=data_handler)

            # CUDA stuff
            if device == "cuda":
                logger.info("Checking if CUDA is available...")
                use_cuda = torch.cuda.is_available()
                if use_cuda:
                    count = torch.cuda.device_count()
                    logger.info(
                        "ML4Chem found {} CUDA devices available.".format(count)
                    )

                    for index in range(count):
                        device_name = torch.cuda.get_device_name(index)

                        if index == 0:
                            device_name += " (Default)"

                        logger.info("    - {}.".format(device_name))

                else:
                    logger.warning("No CUDA available. We will use CPU.")
                    device = "cpu"

            device_ = torch.device(device)

            self.model.to(device_)

            # This is something specific of pytorch.
            module = Potentials.module_names[self.model.name()]
            train = dynamic_import("train", "ml4chem.models", alt_name=module)

            train(
                feature_space,
                targets,
                model=self.model,
                data=data_handler,
                optimizer=optimizer,
                regularization=regularization,
                epochs=epochs,
                convergence=convergence,
                lossfxn=lossfxn,
                device=device,
                batch_size=batch_size,
                **kwargs
            )

        self.save(self.model, features=self.features, path=self.path, label=self.label)

    def calculate(self, atoms, properties, system_changes):
        """Calculate things

        Parameters
        ----------
        atoms : object, list
            List if images in ASE format.
        properties :
        """
        purpose = "inference"
        # Calculator.calculate(self, atoms, properties, system_changes)
        model_name = self.model.name()

        # We convert the atoms in atomic features
        if isinstance(atoms, ase.atoms.Atoms):
            atoms = [atoms]

        data_handler = Data(atoms, purpose=purpose)
        atoms = data_handler.get_data(purpose=purpose)

        # We copy the loaded fingerprint class
        features = copy.deepcopy(self.features)
        kwargs = {"data": data_handler, "purpose": purpose}

        if model_name in Potentials.svm_models:
            kwargs.update({"svm": True})

        if features.name() == "LatentFeatures":
            features = features.calculate(atoms, **kwargs)
        else:
            features.batch_size = self.batch_size
            features.preprocessor = self.preprocessor
            features = features.calculate(atoms, **kwargs)

        if "energy" in properties:
            logger.info("Computing energy...")
            if model_name in Potentials.svm_models:

                try:
                    reference_space = load(self.reference_space)
                except:
                    raise ("This is not a database...")

                energy = self.model.get_potential_energy(
                    features, reference_space, purpose=purpose
                )
            else:
                input_dimension = len(list(features.values())[0][0][-1])
                model = copy.deepcopy(self.model)
                model.prepare_model(input_dimension, data=data_handler, purpose=purpose)
                try:
                    model.load_state_dict(torch.load(self.ml4chem_path), strict=True)
                except RuntimeError:
                    logger.warning(
                        "Your image does not have some atoms present in the loaded model.\n"
                    )
                    model.load_state_dict(torch.load(self.ml4chem_path), strict=False)
                model.eval()

                try:
                    # A single-point energy calculation
                    energy = model(features).item()
                except ValueError:
                    # A list of single-point energy calculations.
                    energy = model(features).tolist()

            # Populate ASE's self.results dict
            self.results["energy"] = energy
