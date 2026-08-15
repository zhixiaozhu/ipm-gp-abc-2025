from s0_fun_perturbation import *
from GP_model_fitting import *
resimu_popu_mode = 'OFF' # 'ON'
regene_popu_summary = 'OFF'

# read models

if resimu_popu_mode == 'ON':
    # generating population
    models_true = baseline_model_fit(data_path="control_78.csv", print_mode = 'OFF')

    print("\n\n\n\n" + 'Re-simulating population' + "\n\n\n\n")
    np.random.seed(12345)
    tf.random.set_seed(54321)
    simu = IBM_full(n0=150, popu_max=100000, models=models_true, total_time=70, NB=False)
    simu_sub = simu.loc[(simu['time'] == 65)&(simu['age'] > 0)]

    popu_dataset=IBM_1step(zt=simu_sub["size"], age=simu_sub["age"], models=models_true, NB=False)
    pickle.dump(popu_dataset, open(file = os.getcwd()+"/mcmc_samples" +"/popu_dataset.pkl", mode="wb"))
    pickle.dump(simu_sub, open(file = os.getcwd()+"/mcmc_samples" +"/simu_sub.pkl", mode="wb"))
    pickle.dump(models_true, open(file = os.getcwd()+"/mcmc_samples" +"/models_true.pkl", mode="wb"))
else:
    simu_sub = pd.read_pickle(open(file = os.getcwd()+"/mcmc_samples" +"/simu_sub.pkl", mode="rb"))
    popu_dataset = pickle.load(open(file = os.getcwd()+"/mcmc_samples" +"/popu_dataset.pkl", mode="rb"))
    models_true = pickle.load(open(file = os.getcwd()+"/mcmc_samples" +"/models_true.pkl", mode="rb"))


# generate other populations to obtein varitions of the proposed summaries under 'ture' parameters
if regene_popu_summary == 'ON':
    np.random.seed(123451)
    tf.random.set_seed(543211)
    print("\n\n\n\n" + 'Recalcualting summary_self_real' + "\n\n\n\n")
    summary_self_real = samemodel_compare(models=models_true, input_size=simu_sub["size"], col_names=col_names, 
                                          input_age=simu_sub["age"], popu_dataset=popu_dataset, rep_time=500, NB=False)
    pickle.dump(summary_self_real, open(file = os.getcwd()+"/mcmc_samples" +"/summary_self_real.pkl", mode="wb"))
else:
    summary_self_real = pickle.load(open(file = os.getcwd()+"/mcmc_samples" +"/summary_self_real.pkl", mode="rb"))


# define two plot functions for MCMC samples.
def plot_samples(samples, parameters, y_axis_label, param_to_name):
    plt.figure(figsize=(8, 4))
    for val, param in zip(samples, parameters):
        plt.plot(tf.squeeze(val), label=param_to_name[param])
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xlabel("HMC iteration")
    plt.ylabel(y_axis_label)

# inspect the marginal distribution of samples.
def marginal_samples(samples, parameters, y_axis_label, param_to_name):
    fig, axes = plt.subplots(1, len(param_to_name), figsize=(15, 3), constrained_layout=True)
    for ax, val, param in zip(axes, samples, parameters):
        ax.hist(np.stack(val).flatten(), bins=20)
        ax.set_title(param_to_name[param])
    fig.suptitle(y_axis_label)
    plt.show()


