import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from tqdm import tqdm
import random
import itertools
import time

'''

Common functions that can be imported 

Potential Class definition

'''

class potentials:
    """

    Class for potentials generation

    """
    def __init__(self, n_pots, n_dw, relevant_feat, plot):
        """
        When this class initializes we automatically define the values for the different potentials

        :param n_pots: (int) Total number of potentials to define. n_dw will determine how many of each (DW or SW)
        :param n_dw: (int) Number of Double Well (DW) potentials desired in the dataset, up to the n_pots
            the rest of potentials will be Single Well (SW). This cannot be bigger than n_pots.
        :param relevant_feat: (int) Index of the DW potential which will be our relevant feature for the outcome
            of the simulations. This has to be between 0 and n_pots.

        """
        dw_pot = np.random.randint(low=0, high=n_pots, size=n_dw)
        pots, shape = self.DefinePotentials(n_pots, dw_pot, plot=False)

        self.n_dw = n_dw
        self.n_pots = n_pots
        self.dw_pot_idx = dw_pot
        self.relevant_feature = relevant_feat
        self.relevant_id = self.dw_pot_idx[self.relevant_feature]
        self.potentials = pots
        self.shape = shape

        return

    def gen_potential(self, name="double_well", n_bins=100, RC_range=[0, 1]):
        """
        This function generates the potential requested. Depending on the type and the number of bins as well as the
        range of values that can be used.

        :param name: (str) Type of potential to use "double_well" for SW and "single_well" for SW.
        :param n_bins: (int) Number of bins to define the potential with.
        :param RC_range: (list) Range of values ([first, last]) to work with.
        :return: (tuple) (X, Y) it returns the different bins the potential has been defined with (X) and
            the values over Y of the potentials.

        """
        # Create a potential on the range [0,1], rescaled afterwards to the RC wanted
        X = np.linspace(0, 1, n_bins)
        Y = [0] * n_bins

        if name == "double_well":
            k_1 = 100
            k_2 = 1
            Y = k_1 * pow(X - 1 / 2, 4) + k_2 * scipy.stats.multivariate_normal.pdf(X, 1 / 2, 0.01)
        elif name == "one_well":
            k_1 = 100
            Y = k_1 * pow(X - 1 / 2, 4)
        else:
            print("MISTAKE in [gen_potential()] : potential name '" + name + "' not implemented.")

        X = X * (RC_range[1] - RC_range[0]) + RC_range[0]  # Rescale the RC if not [0,1]
        return X, Y


    def DefinePotentials(self, n_features, double_well_potentials, plot=False):
        """
        This function generates all X and Y values for the number of potentials requested

        :param n_features: (int) Total Number of potentials to define.
        :param double_well_potentials: (int) Number of Double Well (DW) potentials to include in the total n_features.
        :param plot: (bool) Wether to plot (True) or not (False) the shape of the potentials.
        :return: (list) [potentials, shape] Potentials is a list of the values of the coefficients for each potential 
            and Shape is the X/Y shape of the potentials.

        """
        potentials = {}
        shapes = []

        if plot == True:
            plt.figure()
            plt.title("Potentials used")
            print("Generating Potentials")

        for n in tqdm(range(0, n_features), ascii=True, desc="Defining Potentials"):
            if n in double_well_potentials:
                # (A) Generate Potential (discrete)
                X_gen, Y_gen = self.gen_potential(name="double_well")
                shapes.append([X_gen, Y_gen])
                if plot == True:
                    plt.plot(X_gen, Y_gen, label="Double well #{}".format(n))
            else:
                # (A) Generate Potential (discrete)
                X_gen, Y_gen = self.gen_potential(name="one_well")
                shapes.append([X_gen, Y_gen])
                if plot == True:
                    plt.plot(X_gen, Y_gen, label="One well #{}".format(n))

            # (B) Get polyfit of generated potential and derivative (gradient)
            coeffs = np.polyfit(X_gen, Y_gen, deg=12)
            coeffs_derivative = np.polyder(coeffs)
            potentials[n] = coeffs_derivative
        # plt.legend()

        return potentials, shapes

    def gen_traj_langevin(self, coeffs_derivative, start_pos=0.5, n_steps=1000, diffusion=0.01, simul_lagtime=0.0001):
        """
        Function that generates trajectories on a given set of derivative coefficients with different parameters to 
        control the behaviour of the simulation. Crucial to generate data on the 1D potential.  
        
        :param coeffs_derivative: (list) List of coefficients to generate trajectories on. 1
        :param start_pos: (float) starting position for each simulation. We keep this at 0.5 as the transition state. 
        :param n_steps: (int) Number of steps to run the simulation for. Note that increasing this will make longer 
            trajectories, but changing n_steps and simul_lagtime at the same time is not recommended. Bigger n_steps will
            yield longer (time wise) trajectories.
        :param diffusion: (float) Diffusion coefficient for the langevin dynamics equation, it translates on the speed 
            of the transitions across the free energy landscape.
        :param simul_lagtime: (float) Size of the steps recorded on the simulation, the bigger the less resolution the
            coordinates will have.
        :return: (list) A list containing the coordinates of the potential for the desired number of steps.

        """
        traj = [start_pos]
        for step_id in range(1, n_steps):
            traj.append(
                traj[step_id - 1] - simul_lagtime * (np.polyval(coeffs_derivative, traj[step_id - 1])) + np.sqrt(
                    simul_lagtime * diffusion) * np.random.normal())

        return traj

    def GetAnswers(self, data, relevant_feat):
        """
        Small function made to label the outcome of the given simulations from a list containing them. Note that the
        values for classifying are defaulted to above or below 0.5. Change them manually if needed.

        :param data: (list) List of simulations it has to have the shape (n_sims, n_frames, n_potentials).
        :param relevant_feat: (int) Index of the potential that will be used for labelling the simulation.
        :return:

        """
        answers = []
        print("Getting simulation labels for the generated data")
        for sim in tqdm(data, ascii=True, desc="Classifying Simulation Outcomes"):
            if sim.T[-1][relevant_feat] > 0.5:
                answers.append("IN")
            elif sim.T[-1][relevant_feat] < 0.5:
                answers.append("OUT")

        return answers

    def DataGeneration(self, n_samples, potentials, sim_time):
        """
        Function to generate trajectories on the given potentials for the number of samples/simulations and time/steps
        requested.

        :param n_samples: (int) Number of simulations to run for.
        :param potentials: (list) Derivative coefficients for each of the potentials to run for.
        :param sim_time: (int) Number of steps/time to run the simulations for.
        :return: (list) List containing the data generated for every potential with the shape
            of (n_simulations/n_samples, n_frames/sim_time, n_potentials).

        """
        print("Generating dataset")

        data = []
        for sample in tqdm(range(0, n_samples), ascii=True, desc="Running Simulations"):
            seq = []
            for n, feature in enumerate(range(0, len(potentials))):
                coeffs_derivative = potentials[n]
                traj = self.gen_traj_langevin(coeffs_derivative, n_steps=sim_time, diffusion=0.01, simul_lagtime=0.0001)
                seq.append(np.array(traj))
            data.append(np.array(seq))

        return np.array(data)

    def generate_data(self, n_samples, time):
        """
        Function to wrap generating data and labelling it.

        :param n_samples: (int) Number of simulations/samples of the potentials to run for.
        :param time: (int) number of steps/time to run for.
        :return: (list) List containing (data, answers) the first one is the simulation data for each trajectory
            and the second one the corresponding labels for the isimulations.

        """
        data_whole = self.DataGeneration(n_samples, self.potentials, time)

        ans = self.GetAnswers(data_whole, self.relevant_id)

        return np.array(data_whole), np.array(ans)


class dataset:
    """

    Class for the generation of datasets from a given set of potentials

    """
    def __init__(self, potentials, n_feats, degree_of_mixing):
        """
        This class contains all the parameters needed to generate scrambled 1D data to use as an example on ML
        approaches which try to detect important/correlated features.

        :param potentials: (list) List containing the derivative coefficients of all potentials to simulate on.
        :param n_feats: (int) Number of features to create data for, this will set up the coefficients that will be
            needed later on.
        :param degree_of_mixing: (int) Degree of mixing for the newly generated features, 2 means a mix between
            2 different potentials, 3 between 3, and such. Minimum it can be is 2.

        """

        self.potentials = potentials
        self.n_feats = n_feats
        self.degree = degree_of_mixing

        features = np.arange(0, self.potentials.n_pots)
        features = list(itertools.combinations(features, degree_of_mixing))
        random.shuffle(features)
        features = features[:n_feats]

        coefs = []
        for d in range(degree_of_mixing):
            coefs.append(np.random.uniform(0, 1, n_feats))

        self.combinations = features
        self.mixing_coefs = coefs

        return

    def generate_linear(self, n_samples, time, mode="Normal"):
        """
        Wrapper to generate data on demand, for the desired number of samples and time.

        :param n_samples: (int) Number of simulations/samples to generate.
        :param time: (int) Amount of steps/time to run the simulations for.
        :param mode: (str) Mode on how to generate the last potential. Look at the code for more info.
        :return: (list) List containing the data (X/sim_data) and the labels (Y/answers)

        """

        data, ans = self.potentials.generate_data(n_samples, time)

        sim_data = np.zeros((n_samples, self.n_feats, time))

        for n, variables in enumerate(self.combinations):
            for var in range(self.degree):
                X = data[:, variables[var], :] * self.mixing_coefs[var][n]
                sim_data[:, n, :] += X

        if mode == "Normal":
            pass
        elif mode == "Rigged":
            sim_data[:, -1, :] = data[:, self.potentials.relevant_id, :]

        return sim_data, ans

    def PrepareData(self, data, ans, time_frame, mode="Normal"):
        """

        Small wrapper that prepares the data inputted from the dataset object as the correct format to use on the ML
        approach.

        :param data: (list) Data simulation generated from the dataset.
        :param ans: (list) Labels of the outcome from the simulations
        :param time_frame: (list) [start_frame_range, end_frame_range] Values for the range of frames/steps to keep for
            the final data, this allows to select a particular amount from the trajectories.
        :param mode: (str) Wether to use the real value of the relevant potential as a last feature or not. "Normal"
            means using it.
        :return: (list) List containing the data as (X, Y) being X the simulation data as the mixed trajectories and
            Y as the labelled outcomes for each frame.

        """
        X = data[:, :, time_frame[0]: time_frame[1]]
        X = np.concatenate(X, axis=1).T
        Y = np.ones(len(X)).astype(str)

        for n, answer in enumerate(ans):
            frames = time_frame[1] - time_frame[0]
            tmp_ans = np.ones(frames).astype(str)
            tmp_ans[:] = answer
            Y[n * frames:n * frames + frames] = tmp_ans

        if mode == "Normal":
            pass
        elif mode == "Rigged":
            X = X[:, :-1]

        return X, Y