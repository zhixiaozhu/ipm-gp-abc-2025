from r1_data_generating import *

samples_grw_f = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_grw_f_whole.pkl", mode="rb"))
samples_grw_nf = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_grw_nf_whole.pkl", mode="rb"))
samples_sur = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_sur_whole.pkl", mode="rb"))
samples_fec = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_fec_whole.pkl", mode="rb"))
samples_flow = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_flow_whole.pkl", mode="rb")) 


whole_year = ipmmcmc_whole(popu_dataset_whole=popu_whole, popu_0toT=popu_0toT, 
                            z0=popu_structure_noid_current(popu_0toT[0]), 
                            tmax_list=w.iloc[0], tmin_list=w.iloc[1], precip_list=w.iloc[2],
                            alpha=theta_mle_whole.loc[0, 'alpha'], beta=theta_mle_whole.loc[0, 'beta'], recruit_p=theta_mle_whole.loc[0, 'recruit_p'])

# Loading MCMC samples
# HMC for the other vital rates.
print('\n\n\n' + 'Processing: HMC' + '\n\n\n')

# Grw_nf
whole_year.m_grw_nf_new = gpflow.models.GPR(data=(XY_grw_nf_compu(popu_whole, env=True)), 
                                                kernel=gpflow.kernels.RBF(lengthscales=np.array([1,1,1,1,1])), 
                                                mean_function=None)

# we add priors to the hyperparameters.
whole_year.m_grw_nf_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
whole_year.m_grw_nf_new.likelihood.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
whole_year.m_grw_nf_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
whole_year.hmc_helper_grw_nf = gpflow.optimizers.SamplingHelper(
    whole_year.m_grw_nf_new.log_posterior_density, whole_year.m_grw_nf_new.trainable_parameters
)

whole_year.samples_grw_nf = samples_grw_nf

# Fec
whole_year.m_fec_new = gpflow.models.GPMC(data=(XY_fec_compu(popu_whole, env=True)), 
                            kernel=gpflow.kernels.RBF(lengthscales=np.array([1,1,1,1,1])) , 
                            likelihood=gpflow.likelihoods.Bernoulli())

whole_year.m_fec_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
whole_year.m_fec_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))

whole_year.hmc_helper_fec = gpflow.optimizers.SamplingHelper(
    whole_year.m_fec_new.log_posterior_density, whole_year.m_fec_new.trainable_parameters
)

whole_year.samples_fec = samples_fec

# Flow poi
whole_year.m_flow_poi_new = gpflow.models.GPMC(data=(XY_flow_compu(popu_whole, env=True)), 
                                    kernel=gpflow.kernels.RBF(lengthscales=np.array([1,1,1,1,1])), 
                                    likelihood=gpflow.likelihoods.Poisson())

whole_year.m_flow_poi_new.kernel.lengthscales.prior =  tfd.Normal(loc=f64(0.), scale=f64(100.))
whole_year.m_flow_poi_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))

whole_year.hmc_helper_flow = gpflow.optimizers.SamplingHelper(
    whole_year.m_flow_poi_new.log_posterior_density, whole_year.m_flow_poi_new.trainable_parameters
)

whole_year.samples_flow = samples_flow


# Grw_f
whole_year.m_grw_f_new = gpflow.models.GPR(data=(XY_grw_f_compu(popu_whole, env=True)), 
                                kernel= gpflow.kernels.RBF(lengthscales=np.array([1,1,1,1,1])), mean_function=None)

whole_year.m_grw_f_new.kernel.lengthscales.prior =  tfd.Normal(loc=f64(0.), scale=f64(100.))
whole_year.m_grw_f_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
whole_year.m_grw_f_new.likelihood.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))

whole_year.hmc_helper_grw_f = gpflow.optimizers.SamplingHelper(
    whole_year.m_grw_f_new.log_posterior_density, whole_year.m_grw_f_new.trainable_parameters
)

whole_year.samples_grw_f = samples_grw_f

# Sur
whole_year.m_sur_new = gpflow.models.GPMC(data=(XY_sur_compu(popu_whole, env=True)), 
                            kernel=gpflow.kernels.RBF(lengthscales=np.array([1,1,1,1,1])) , 
                            likelihood=gpflow.likelihoods.Bernoulli())

whole_year.m_sur_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
whole_year.m_sur_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))

whole_year.hmc_helper_sur = gpflow.optimizers.SamplingHelper(
    whole_year.m_sur_new.log_posterior_density, whole_year.m_sur_new.trainable_parameters
)

whole_year.samples_sur = samples_sur

whole_year.rep = 1

whole_year.recru_full = pickle.load(open(file = os.getcwd()+f"/true_popu/mle/theta_mle_full.pkl", mode="rb")) 



for iiiii in range(5000):
    whole_year.GPmodels_givenindex_cache([iiiii, iiiii, iiiii, iiiii, iiiii])
    pickle.dump(GPMC_posterior(whole_year.m_sur_new).cache, open(file = os.getcwd()+f"/true_popu/mcmc/Lm/sur/{iiiii}", mode="wb"))
    pickle.dump(whole_year.m_grw_f_new.posterior().cache, open(file = os.getcwd()+f"/true_popu/mcmc/Lm/grw_f/{iiiii}", mode="wb"))
    pickle.dump(whole_year.m_grw_nf_new.posterior().cache, open(file = os.getcwd()+f"/true_popu/mcmc/Lm/grw_nf/{iiiii}", mode="wb"))
    pickle.dump(GPMC_posterior(whole_year.m_flow_poi_new).cache, open(file = os.getcwd()+f"/true_popu/mcmc/Lm/flow/{iiiii}", mode="wb"))
    pickle.dump(GPMC_posterior(whole_year.m_fec_new).cache, open(file = os.getcwd()+f"/true_popu/mcmc/Lm/fec/{iiiii}", mode="wb"))


for _ in range(50):
    index = np.random.randint(0, 5000, size = 5) 
    whole_year.m_sur_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/sur/{index[0]}", mode="rb"))
    whole_year.m_grw_f_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/grw_f/{index[1]}", mode="rb")) 
    whole_year.m_grw_nf_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/grw_nf/{index[2]}", mode="rb")) 
    whole_year.m_fec_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/fec/{index[3]}", mode="rb")) 
    whole_year.m_flow_poi_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/flow/{index[4]}", mode="rb"))

    for var, var_samples in zip(whole_year.hmc_helper_sur.current_state, whole_year.samples_sur):
        var.assign(var_samples[index[0]])
    # grw_f
    for var, var_samples in zip(whole_year.hmc_helper_grw_f.current_state, whole_year.samples_grw_f):
        var.assign(var_samples[index[1]])
    # grw_nf
    for var, var_samples in zip(whole_year.hmc_helper_grw_nf.current_state, whole_year.samples_grw_nf):
        var.assign(var_samples[index[2]])
    # fec
    for var, var_samples in zip(whole_year.hmc_helper_fec.current_state, whole_year.samples_fec):
        var.assign(var_samples[index[3]])
    # flow
    for var, var_samples in zip(whole_year.hmc_helper_flow.current_state, whole_year.samples_flow):
        var.assign(var_samples[index[4]])


    models_now2 = {
        "m_sur": whole_year.m_sur_new ,
        "m_grw_f": whole_year.m_grw_f_new,
        "m_grw_nf": whole_year.m_grw_nf_new ,
        "m_fec": whole_year.m_fec_new,
        "m_flow_poi": whole_year.m_flow_poi_new,
        "alpha": whole_year.alpha,
        "beta": whole_year.beta,
        "recruit_p": whole_year.recruit_p
    }


    np.random.seed(4567)
    a = IBM_1step(zt=popu_structure_noid_current(popu_0toT[0])[0], age=popu_structure_noid_current(popu_0toT[0])[1], 
            models=models_now2, env=True, tmax=w.iloc[0][0], tmin=w.iloc[1][0], precip=w.iloc[2][0])

    np.random.seed(4567)
    b = IBM_1step_cashe(zt=popu_structure_noid_current(popu_0toT[0])[0], age=popu_structure_noid_current(popu_0toT[0])[1], 
                    models=models_now2, env=True, tmax=w.iloc[0][0], tmin=w.iloc[1][0], precip=w.iloc[2][0])
    np.testing.assert_allclose(a, b)

