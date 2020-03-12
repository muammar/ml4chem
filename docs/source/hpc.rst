===================
Introduction
===================

ML4Chem uses `Dask <https://docs.dask.org/en/latest/>`_ which is a flexible
library for parallel computing in Python. Dask allows easy scaling up and
down without too much effort. 

In this part of the documentation, we will cover how ML4Chem can be run on a
laptop or workstation and how we can scale up to running on HPC clusters.
Dask has a  modern and interesting structure:

#. A scheduler is in charge of registering tasks.
#. Tasks can be registered in a delayed way (registered but not computed) or
   simply submitted as futures (submitted and computed).
#. When the scheduler receives a task, it sends it to workers that carry out
   the computations and keep them in memory. 
#. Results from computations can be subsequently used for more calculations or
   just brought back to memory.


=====================
Scale Down
=====================

Running computations with ML4Chem on a personal workstation or laptop is very
easy thanks to Dask. The :code:`LocalCluster` class uses local resources to
carry out computations. This is useful when prototyping and building your
pipeline withouth wasting time waiting for HPC resources in a crowded cluster
facility.

ML4Chem can run with:code:`LocalCluster` objects, for which the scripts have
to contain the following::

   from dask.distributed import Client, LocalCluster

   cluster = LocalCluster(n_workers=8, threads_per_worker=2)
   client = Client(cluster)

In the snippet above, we imported :code:`Client` that will connect to the
scheduler created by the :code:`LocalCluster` class. The scheduler will have
8 workers with 2 threads. As tasks are required, they are sent by the
:code:`Client` to the :code:`LocalCluster` for being computed and kept in
memory.

A typical script for running training in ML4Chem looks as follows::


    from ase.io import Trajectory
    from dask.distributed import Client, LocalCluster
    from ml4chem.atomistic import Potentials
    from ml4chem.atomistic.features import Gaussian
    from ml4chem.atomistic.models.neuralnetwork import NeuralNetwork
    from ml4chem.utils import logger


    def train():
        # Load the images with ASE
        images = Trajectory("cu_training.traj")

        # Arguments for fingerprinting the images
        normalized = True

        # Arguments for building the model
        n = 10
        activation = "relu"

        # Arguments for training the potential
        convergence = {"energy": 5e-3}
        epochs = 100
        lr = 1.0e-2
        weight_decay = 0.0
        regularization = 0.0

        calc = Potentials(
            features=Gaussian(
                cutoff=6.5, normalized=normalized, save_preprocessor="model.scaler"
            ),
            model=NeuralNetwork(hiddenlayers=(n, n), activation=activation),
            label="cu_training",
        )

        optimizer = ("adam", {"lr": lr, "weight_decay": weight_decay})
        calc.train(
            training_set=images,
            epochs=epochs,
            regularization=regularization,
            convergence=convergence,
            optimizer=optimizer,
        )


    if __name__ == "__main__":
        logger(filename="cu_training.log")
        cluster = LocalCluster()
        client = Client(cluster)
        train()

=====================
Scale Up
=====================

Once you have finished with prototyping and feel ready to scale up, the
snippet above can be trivially expanded to work with high performance
computing (HPC) systems. Dask offers a module called :code:`dask_jobqueue`
that enables sending computations to HPC systems with Batch systems such as
SLURM, LSF, PBS and others (for more information see
`<https://jobqueue.dask.org/en/latest/index.html>`_.

To scale up in ML4Chem with Dask, you only have to slightly change the
snipped above as follows::


    if __name__ == "__main__":
        from dask_jobqueue import SLURMCluster
        logger(filename="cu_training.log")


        cluster = SLURMCluster(
            cores=24, 
            processes=24, 
            memory="100GB", 
            walltime="24:00:00", 
            queue="dirac1", 
        )
        print(cluster)
        print(cluster.job_script())
        cluster.scale(jobs=4)
        client = Client(cluster)
        train()

We removed the :code:`LocalCluster` and instead used the :code:`SLURMCluster`
class to submit our computations to a SLURM batch system. As it can be seen,
the :code:`cluster` is now a :code:`SLURMCluster` requesting a job with 24
cores and 24 processes, 100GB of RAM, a wall time of 1 day, and the queue in
this case is `dirac1`. Then, we scaled this up by requesting to the HPC
cluster 4 jobs with these requirements for a total of 96 processes. This
:code:`cluster` is passed to the :code:`client` and the training is
effectively scaled up.