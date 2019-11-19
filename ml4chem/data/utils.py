from ase.io import Trajectory
import random


def split_Data(
    images,
    training_name="training_images.traj",
    test_name="test_images.traj",
    randomize=True,
    test_set=20,
    logfile="data_split.log",
):
    """Split Data
    
    Parameters
    ----------
    images : str or object
        A path to an ASE trajectory file or a list of Atoms objects.
    training_name : str, optional
        Name of the training set trajectory file, by default 'training_images.traj'
    test_name : str, optional
        Name of the test set file, by default 'test_images.traj'
    randomize : bool, optional
        Randomize indices of images, by default True
    test_set : int, optional
        Percentage of the Data to be used as test set, by default 20
    logfile : str, optional
        Log file name, by default 'data_split.log'
    """
    if isinstance(images, str):
        images = Trajectory(images)

    total_length = len(images)
    test_length = int((test_set * total_length / 100))
    training_leght = int(total_length - test_length)

    _images = list(range(len(images)))

    if randomize:
        random.shuffle(_images)

    training_images = []
    training_traj = Trajectory(training_name, mode="w")

    log = open(logfile, "w")

    log.write("Training set\n")

    for i in _images[0:training_leght]:
        training_images.append(i)
        training_traj.write(images[i])

    log.write(str(training_images))
    log.write("\n")

    log.write("Test set\n")

    if test_set > 0:
        test_images = []
        test_traj = Trajectory(test_name, mode="w")
        for i in _images[-test_length:-1]:
            test_images.append(i)
            test_traj.write(images[i])

        log.write(str(test_images))
    log.close()
