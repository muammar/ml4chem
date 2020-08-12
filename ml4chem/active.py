import numpy as np
import logging
import itertools

# Starting logger object
logger = logging.getLogger()


class ActiveLearning(object):
    """Active Learning

    Parameters
    ----------
    labeled : list
        List of graphs or objects. 
    unlabeled : object
        List of graphs or objects. 
    atomistic : bool, optional
        Atomistic similarities?, by default False.
    """

    def __init__(self, labeled, unlabeled, atomistic=True):
        self.labeled = labeled
        self.unlabeled = unlabeled

        if atomistic == False:
            raise RuntimeError("This is not implemented yet.")

        self.atomistic = atomistic

    def run(self, kernel, max_variance=10, max_iter=None):
        """Run the ActiveLearning class

        Parameters
        ----------
        kernel : object
            A kernel to measure similarity.
        max_variance : float, optional
            Maximum variance allowed, by default 10.
        max_iter : int, optional
            Maximum number of iterations allowed, by default None.
        """

        converged = False
        nodal = self.atomistic

        logger.info("Computing diagonal matrix of both labeled and unlabeled data...")
        self.D = kernel.diag(self.labeled + self.unlabeled, nodal=nodal)
        print(self.D.shape)
        logger.info("Finished...\n")

        _indices = np.cumsum(
            [len(graph.nodes) for graph in self.labeled + self.unlabeled]
        ).tolist()

        l_indices = []
        u_indices = []

        for i, index in enumerate(_indices):
            if i == 0:
                u_indices.append(list(range(0, index)))
            else:
                u_indices.append(list(range(_indices[i - 1], index)))
        del _indices

        iterations = 0

        while not converged:
            iterations += 1
            self.variances = []
            logger.info("Labeled data points   : {}.".format(len(self.labeled)))
            logger.info("Unlabeled data points : {}.".format(len(self.unlabeled)))

            self._rll = kernel(self.labeled, nodal=nodal)
            self._ruu = kernel(self.unlabeled, nodal=nodal)
            self._rlu = kernel(self.labeled, self.unlabeled, nodal=nodal)

            if len(self.labeled) == 1:
                l_indices += u_indices.pop(0)

            print(l_indices)
            print(u_indices)
            Dl = np.take(self.D, l_indices) ** -0.5
            self.kll = Dl[None, :] * self._rll * Dl[:, None]
            _u_indices = list(itertools.chain.from_iterable(u_indices))
            Du = np.take(self.D, _u_indices) ** -0.5
            self.klu = Dl[:, None] * self._rlu * Du[None, :]
            # print("KLL")
            # print(self.kll)
            # print("KLU")
            # print(self.klu.shape)
            # print(_u_indices)
            # print(self.kll)
            k_inv = np.linalg.pinv(self.kll)

            for u in range(len(_u_indices)):
                klu_u = self.klu[None:, u]
                vu = 1.0 - klu_u.T.dot(k_inv).dot(klu_u)
                self.variances.append(vu)

            self.max_var_index = np.argmax(self.variances)
            self.current_var = self.variances[self.max_var_index]

            index_to_search = _u_indices[self.max_var_index]
            print(index_to_search)

            for index, graph in enumerate(u_indices):
                if index_to_search in graph:
                    add_to_labeled = self.unlabeled.pop(index)
                    self.labeled.append(add_to_labeled)
                    l_indices += graph
                    u_indices.pop(index)
                    break

            logger.info(
                "A graph with variance {} has been labeled.".format(self.current_var)
            )

            # TODO add max_variance as a criterion for converging.
            # if self.current_var > max_variance:
            #     add_to_labeled = self.unlabeled.pop(self.pop_index)
            #     self.labeled.append(add_to_labeled)
            # else:
            #     converged = True
            #     logger.info("Convergence reached")

            if len(self.unlabeled) == 0:
                logger.info("There are no more unlabeled data points...")
                break
            elif iterations == max_iter:
                logger.info("Total number of iterations was reached...")
                converged = True
