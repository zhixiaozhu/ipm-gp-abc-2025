import pickle
import os
import gpflow
import numpy as np
from tensorflow_probability import distributions as tfd
from s0_fun_base import XY_grw_nf_compu, XY_grw_f_compu, XY_grw_compu, XY_sur_compu, XY_fec_compu, XY_flow_compu
from s0_class_ABCPMC import ipmmcmc_whole
from s0_fun_IBMs import popu_structure
f64 = gpflow.utilities.to_default_float
import ray

truedata_style='glm'
grw_setting='sep'

popu_dataset = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/"+truedata_style+"_mle_true_population.pkl", mode="rb"))
models_true = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/gp_"+truedata_style+"mle_mle_models.pkl", mode="rb"))

samples_grw_f = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/MCMC/gp/sep/samples_grw_f.pkl", mode="rb"))
samples_grw_nf = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/MCMC/gp/sep/samples_grw_nf.pkl", mode="rb"))
samples_sur = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/MCMC/gp/sep/samples_sur.pkl", mode="rb"))
samples_flow = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/MCMC/gp/sep/samples_flow.pkl", mode="rb"))
samples_fec = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/MCMC/gp/sep/samples_fec.pkl", mode="rb"))



whole_year = ipmmcmc_whole(popu_data=popu_dataset, grw_setting=grw_setting, 
                            z0=popu_structure(popu_dataset), 
                            truedata_style=truedata_style, 
                            alpha=models_true['alpha'], beta=models_true['beta'], recruit_p=models_true['recruit_p'])

# Loading MCMC samples
# HMC for the other vital rates.
print('\n\n\n' + 'Processing: HMC' + '\n\n\n')

# Fec
df_fec = XY_fec_compu(popu_dataset)  
whole_year.m_fec_new = gpflow.models.GPMC(data=(df_fec[0], df_fec[1]), 
                               kernel=gpflow.kernels.RBF(), likelihood=gpflow.likelihoods.Bernoulli())

whole_year.m_fec_new.kernel.variance.prior = tfd.HalfNormal(scale=f64(100.))
whole_year.m_fec_new.kernel.lengthscales.prior = tfd.HalfNormal(scale=f64(100.))
whole_year.hmc_helper_fec = gpflow.optimizers.SamplingHelper(
    whole_year.m_fec_new.log_posterior_density, whole_year.m_fec_new.trainable_parameters
)

whole_year.samples_fec = samples_fec

# Flow poi
df_flow = XY_flow_compu(popu_dataset)  
whole_year.m_flow_poi_new =  gpflow.models.GPMC(data=(df_flow[0], df_flow[1]), 
                                kernel=gpflow.kernels.RBF(), likelihood=gpflow.likelihoods.Poisson())
whole_year.m_flow_poi_new.kernel.lengthscales.prior = tfd.HalfNormal(scale=f64(100.))
whole_year.m_flow_poi_new.kernel.variance.prior = tfd.HalfNormal(scale=f64(100.))
whole_year.hmc_helper_flow = gpflow.optimizers.SamplingHelper(
    whole_year.m_flow_poi_new.log_posterior_density, whole_year.m_flow_poi_new.trainable_parameters
)

whole_year.samples_flow = samples_flow


# Grw_f
df_grw_f = XY_grw_f_compu(popu_dataset)    
whole_year.m_grw_f_new = gpflow.models.GPR(data=(df_grw_f[0], df_grw_f[1]), kernel=gpflow.kernels.RBF())
whole_year.m_grw_f_new.kernel.lengthscales.prior = tfd.HalfNormal(scale=f64(100.))
whole_year.m_grw_f_new.kernel.variance.prior = tfd.HalfNormal(scale=f64(100.))
whole_year.m_grw_f_new.likelihood.variance.prior = tfd.HalfNormal(scale=f64(100.))
whole_year.hmc_helper_grw_f = gpflow.optimizers.SamplingHelper(
    whole_year.m_grw_f_new.log_posterior_density, whole_year.m_grw_f_new.trainable_parameters
)

whole_year.samples_grw_f = samples_grw_f


# Grw_nf
df_grw_nf = XY_grw_nf_compu(popu_dataset)    
whole_year.m_grw_nf_new = gpflow.models.GPR(data=(df_grw_nf[0], df_grw_nf[1]), kernel=gpflow.kernels.RBF())
# we add priors to the hyperparameters.
whole_year.m_grw_nf_new.kernel.lengthscales.prior = tfd.HalfNormal(scale=f64(100.))
whole_year.m_grw_nf_new.kernel.variance.prior = tfd.HalfNormal(scale=f64(100.))
whole_year.m_grw_nf_new.likelihood.variance.prior = tfd.HalfNormal(scale=f64(100.))
whole_year.hmc_helper_grw_nf = gpflow.optimizers.SamplingHelper(
    whole_year.m_grw_nf_new.log_posterior_density, whole_year.m_grw_nf_new.trainable_parameters
)

whole_year.samples_grw_nf = samples_grw_nf

# Sur
df_sur = XY_sur_compu(popu_dataset)  
whole_year.m_sur_new = gpflow.models.GPMC(data=(df_sur[0], df_sur[1]), 
                               kernel=gpflow.kernels.RBF(), likelihood=gpflow.likelihoods.Bernoulli())

whole_year.m_sur_new.kernel.lengthscales.prior = tfd.HalfNormal(scale=f64(50.))
whole_year.m_sur_new.kernel.variance.prior = tfd.HalfNormal(scale=f64(50.))
whole_year.hmc_helper_sur = gpflow.optimizers.SamplingHelper(
    whole_year.m_sur_new.log_posterior_density, whole_year.m_sur_new.trainable_parameters
)
whole_year.samples_sur = samples_sur


ray.init(log_to_driver=False)



print('\n Simulation starts! \n')

abcresult = whole_year.ABC_SMC(quantiles=np.array([0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2]),
                                n_particles=np.array([800000, 500000, 250000, 150000, 80000, 30000, 10000]))
