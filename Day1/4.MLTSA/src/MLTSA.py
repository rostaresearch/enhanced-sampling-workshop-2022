""" This is an equivalent to the MLTSA module "pip install MLTSA" """

def read_bin(filename):
    import pickle
    import numpy as np
    data = []
    with open(filename,"rb") as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return data

def create_bin(filename, data):
    import pickle
    import numpy as np
    file = open(filename, "wb+")
    pickle.dump(data, file)
    return file

def append_bin(file, data):
    import pickle
    import numpy as np
    pickle.dump(data, file)
    return



def ADROP(data, ans, model, drop_mode="Average", data_mode="Normal"):
    """

    Function to apply the Machine Learning Transition State Analysis to a given training dataset/answers and trained
    model. It calculates the Gloabl Means and re-calculates accuracy for predicting each outcome.

    :param data: Training data used for training the ML model. Must have shape (samples, features)
    :type data: list
    :param ans: Outcomes for each sample on "data". Shape must be (samples)
    :type ans: list
    :param model:
    :param drop_mode:
    :param data_mode:
    :return:

    """
    import pickle
    import numpy as np
    from tqdm import tqdm
    
    if data_mode == "Rigged":
        data = data[:, :-1, :]

    # Calculating the global means
    means_per_sim = np.mean(data.T, axis=1)
    gmeans = np.mean(means_per_sim, axis=1)
    temp_sim_data = np.copy(data)

    # Swapping the values and predicting for the FR
    FR = []
    for y, data in tqdm(enumerate(temp_sim_data)):
        mean_sim = []
        for n, mean in enumerate(gmeans):
            tmp_dat = np.copy(data)
            tmp_dat.T[n, :] = mean
            yy = np.zeros(len(tmp_dat)).astype(str)
            yy[:] = ans[y]
            res = model.score(tmp_dat, yy)
            mean_sim.append(res)
        FR.append(mean_sim)

    if drop_mode == "Median":
        median = np.median(np.array(FR).T, axis=0)
        median = np.median(median, axis=0)
        dv_from_median = []
        for n, M in enumerate(median):
            dv_from_median.append(abs(M - np.array(FR)[n].T))
        fr = np.mean(np.array(dv_from_median), axis=1)
        fr = np.mean(fr, axis=0)
    return FR



def MLTSA_ADROP_Plot(FR, dataset_og, pots, corr_thresh=30, errorbar=True):
    """
    Wrapper for plotting the results from the Accuracy Drop procedure
    :param FR: Values from the feature reduction
    :param dataset_og: Original dataset object class used for generating the data
    :param pots: Original potentials object class used for generating the data
    :param errorbar: Flag for including or not including the errobars in case of using replicas.
    :return:
    """
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib as mpl
    # Fetching info
    std = np.std(FR, axis=0, ddof=1) * 100
    dat = np.mean(FR, axis=0) * 100
    coefs = dataset_og.mixing_coefs
    coefs = np.array(coefs).T
    combs = dataset_og.combinations
    imp_id = pots.relevant_id

    # Getting the correlated features
    cor_feats = []
    for n, idx in enumerate(combs[:len(dat)]):
        if imp_id in idx:
            cor_feats.append(n)

    # Calculating Correlation relation
    correlations = []
    for n, f in enumerate(cor_feats):
        correlation = coefs[f][np.where(np.array(combs[f]) == imp_id)[0][0]] / np.sum(coefs[f]) * 100
        correlations.append(correlation)

    plt.figure(figsize=(10, 3))
    plt.title("Feature Reduction")
    plt.plot(dat, "-o", color="black", marker="s")
    # print(dat)
    #     if errorbar == True:
    #         plt.errorbar(np.arange(0, len(dat)), dat, yerr=std,
    #                      capsize=5, linestyle="None", marker="^", color="black")

    for correlated_feat, corr in zip(cor_feats, correlations):
        # rgb = colorsys.hsv_to_rgb((200+int(corr))/300., 1.0, 1.0)
        rgb = ((corr * 2.55) / 255, 0, 1 - (corr * 2.55) / 255)
        plt.plot(correlated_feat, dat[correlated_feat], "X", markersize=10, color=rgb)
        if errorbar == True:
            plt.errorbar(correlated_feat, dat[correlated_feat], yerr=std[correlated_feat],
                         capsize=5, linestyle="None", marker="^", color="black")
        #         plt.annotate("{:.2f}".format(corr),
        #                      (correlated_feat, dat[correlated_feat]),
        #                     fontsize=12, fontstyle="italic", fontweight="medium")
        if corr > corr_thresh:
            plt.text(correlated_feat + 6, dat[correlated_feat] - 0.005, "{:.2f}".format(corr),
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
                     fontstyle="italic", fontweight="medium"
                     )
    plt.ylabel("Accuracy (%)")
    plt.xlabel("#Feature")
    # plt.colorbar()
    #"""Code for the colorbar below"""
    #print("Plotting Colorbar")
    rgb1 = (0, 0, 1)
    rgb2 = (1, 0, 0)
    cmap_name = "corr"
    cmap = LinearSegmentedColormap.from_list(cmap_name, [rgb1, rgb2], N=100)
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax, orientation='horizontal', label='Correlation to DW potential (%)')
    #fig.show()

    return

def MLTSA_FeatImp_Plot(FR, dataset_og, pots, corr_thresh=30, errorbar=True):
    """
    Wrapper for plotting the results from the Accuracy Drop procedure
    :param FR: Values from the feature reduction
    :param dataset_og: Original dataset object class used for generating the data
    :param pots: Original potentials object class used for generating the data
    :param errorbar: Flag for including or not including the errobars in case of using replicas.
    :return:
    """
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib as mpl
    # Fetching info
    std = np.std(FR, axis=0, ddof=1)
    dat = np.mean(FR, axis=0) 
    coefs = dataset_og.mixing_coefs
    coefs = np.array(coefs).T
    combs = dataset_og.combinations
    imp_id = pots.relevant_id

    # Getting the correlated features
    cor_feats = []
    for n, idx in enumerate(combs[:len(dat)]):
        if imp_id in idx:
            cor_feats.append(n)

    # Calculating Correlation relation
    correlations = []
    for n, f in enumerate(cor_feats):
        correlation = coefs[f][np.where(np.array(combs[f]) == imp_id)[0][0]] / np.sum(coefs[f]) * 100
        correlations.append(correlation)

    plt.figure(figsize=(15, 3))
    plt.title("Feature Reduction")
    plt.plot(dat, "-o", color="black", marker="s")

    for correlated_feat, corr in zip(cor_feats, correlations):
        rgb = ((corr * 2.55) / 255, 0, 1 - (corr * 2.55) / 255)
        plt.plot(correlated_feat, dat[correlated_feat], "X", markersize=10, color=rgb)
        if errorbar == True:
            plt.errorbar(correlated_feat, dat[correlated_feat], yerr=std[correlated_feat],
                         capsize=5, linestyle="None", marker="^", color="black")
        if corr > corr_thresh:
            plt.text(correlated_feat + 6, dat[correlated_feat] - 0.005, "{:.2f}".format(corr),
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
                     fontstyle="italic", fontweight="medium"
                     )
    plt.ylabel("Gini Feature Importance")
    plt.xlabel("#Feature")

    rgb1 = (0, 0, 1)
    rgb2 = (1, 0, 0)
    cmap_name = "corr"
    cmap = LinearSegmentedColormap.from_list(cmap_name, [rgb1, rgb2], N=100)
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax, orientation='horizontal', label='Correlation to DW potential (%)')
    #fig.show()

    return