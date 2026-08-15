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

Time_all = 9
for i in range(Time_all):
    globals()[f'p_all{i}'] = pickle.load(open(file = os.getcwd()+f"/true_popu/popu_dataset{i}.pkl", mode="rb"))
popu_all = {i: globals()[f'p_all{i}'] for i in range(Time_all)}
for i in range(Time_all):
    del(globals()[f'p_all{i}']) 

whole_year.recru_full = pickle.load(open(file = os.getcwd()+f"/true_popu/mle/theta_mle_full.pkl", mode="rb")) 
whole_year.popu_all = popu_all


# predicting lambdas
ray.init(log_to_driver=False)
print('\n Simulation starts! \n')

whole_year.true_z_allsteps = False
whole_year.abc_index = np.load(open(file = os.getcwd()+f"/ABC_details/p_index_ini.pkl", mode="rb"))

a = np.array(whole_year.prediction_para(total_samples=5000, abc=False), dtype=object)
np.save(open(file = os.getcwd()+"/true_popu/simulations/simu_random.pkl", mode="wb"), a)



