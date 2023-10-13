import numpy as np


class TargetGenerator:
    def __init__(self, num_dof=1, center_out=False, is_discrete=False, discrete_targs=None, continuous_range=None):
        """
        :param num_dof (int):           Number of degrees of freedom (i.e. how many targets to make)
        :param center_out (bool):       If True, alternates between the center position (0.5) and the other targets
        :param is_discrete (bool):      If True, will choose targets from the discrete_targs list. If False, will
                                        randomly choose targets from the continuous_range.
        :param discrete_targs (list):   List of target positions to choose from in discrete mode. Each target should be
                                        a tuple of length num_dof.
        :param continuous_range (list): List with the upper and lower limits for continuous targets. Defaults to [0, 1]
        """
        self.num_dof = num_dof
        self.center_out = center_out
        self.is_discrete = is_discrete
        self.discrete_targs = np.array(discrete_targs)
        self.cont_range = continuous_range if continuous_range else [0, 1]

        self.at_center = False
        self.target_pos = None

    def reset(self):
        self.at_center = False
        self.target_pos = None

    def generate_targets(self):
        if self.center_out:
            if not self.at_center:
                self.target_pos = 0.5 * np.ones(self.num_dof)
                self.at_center = True
            else:
                # self.target_pos = np.array(np.random.choice(self.discrete_targs))
                self.target_pos = self.discrete_targs[np.random.choice(self.discrete_targs.shape[0])]
                self.at_center = False

        else:
            self.target_pos = np.array([np.random.uniform(self.cont_range[0], self.cont_range[1])
                                        for _ in range(self.num_dof)])
        return self.target_pos
