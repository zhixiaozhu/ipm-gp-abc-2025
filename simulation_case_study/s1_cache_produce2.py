import pickle
import os
import gpflow
import numpy as np
from tensorflow_probability import distributions as tfd
from s0_fun_base import XY_grw_nf_compu, XY_grw_f_compu, XY_grw_compu, XY_sur_compu, XY_fec_compu, XY_flow_compu
from s0_class_ABCPMC import ipmmcmc_whole
from s0_fun_IBMs import popu_structure, GPMC_posterior, IBM_1step_gp_cache, IBM_1step_gp_same_cache, IBM_1step_gp, IBM_1step_gp_same
f64 = gpflow.utilities.to_default_float


truedata_style='gp'
popu_dataset = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/"+truedata_style+"_mle_true_population.pkl", mode="rb"))
models_true = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/gp_"+truedata_style+"mle_mle_models.pkl", mode="rb"))

samples_grw_f = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/MCMC/gp/sep/samples_grw_f.pkl", mode="rb"))
samples_grw_nf = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/MCMC/gp/sep/samples_grw_nf.pkl", mode="rb"))
samples_grw = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/MCMC/gp/nonsep/samples_grw.pkl", mode="rb"))
samples_sur = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/MCMC/gp/sep/samples_sur.pkl", mode="rb"))
samples_flow = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/MCMC/gp/sep/samples_flow.pkl", mode="rb"))
samples_fec = pickle.load(open(file = os.getcwd()+f"/true_popu/"+truedata_style+"_mle_true_population/MCMC/gp/sep/samples_fec.pkl", mode="rb"))



whole_year = ipmmcmc_whole(popu_data=popu_dataset, grw_setting='sep', 
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
whole_year.m_fec_new.kernel.variance.prior = tfd.InverseGamma(f64(0.001),f64(0.001))
whole_year.m_fec_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
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

# Grw
df_grw = XY_grw_compu(popu_dataset)    
whole_year.m_grw_new = gpflow.models.GPR(data=(df_grw[0], df_grw[1]), kernel=gpflow.kernels.RBF())
# we add priors to the hyperparameters.
whole_year.m_grw_new.kernel.lengthscales.prior = tfd.HalfNormal(scale=f64(100.))
whole_year.m_grw_new.kernel.variance.prior = tfd.HalfNormal(scale=f64(100.))
whole_year.m_grw_new.likelihood.variance.prior = tfd.HalfNormal(scale=f64(100.))
whole_year.hmc_helper_grw = gpflow.optimizers.SamplingHelper(
    whole_year.m_grw_new.log_posterior_density, whole_year.m_grw_new.trainable_parameters
)

whole_year.samples_grw = samples_grw

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


whole_year.rep = 1




for iiiii in range(5000):
    whole_year.GPmodels_givenindex_cache([iiiii, iiiii, iiiii, iiiii, iiiii, iiiii])
    pickle.dump(GPMC_posterior(whole_year.m_sur_new).cache, open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/sur/{iiiii}", mode="wb"))
    pickle.dump(whole_year.m_grw_f_new.posterior().cache, open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/grw_f/{iiiii}", mode="wb"))
    pickle.dump(whole_year.m_grw_nf_new.posterior().cache, open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/grw_nf/{iiiii}", mode="wb"))
    pickle.dump(GPMC_posterior(whole_year.m_flow_poi_new).cache, open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/flow/{iiiii}", mode="wb"))
    pickle.dump(GPMC_posterior(whole_year.m_fec_new).cache, open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/fec/{iiiii}", mode="wb"))
    pickle.dump(whole_year.m_grw_new.posterior().cache, open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/grw/{iiiii}", mode="wb"))


for _ in range(50):
    index = np.random.randint(0, 5000, size = 5) 
    whole_year.m_sur_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/sur/{index[0]}", mode="rb"))
    whole_year.m_grw_f_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/grw_f/{index[1]}", mode="rb")) 
    whole_year.m_grw_nf_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/grw_nf/{index[2]}", mode="rb")) 
    whole_year.m_fec_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/fec/{index[3]}", mode="rb")) 
    whole_year.m_flow_poi_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/flow/{index[4]}", mode="rb"))

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
    a = IBM_1step_gp(zt=whole_year.z0[0], age=whole_year.z0[1], models=models_now2)

    np.random.seed(4567)
    b = IBM_1step_gp_cache(zt=whole_year.z0[0], age=whole_year.z0[1], models=models_now2)
    np.testing.assert_allclose(a, b)


for _ in range(50):
    index = np.random.randint(0, 5000, size = 4) 
    whole_year.m_sur_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/sur/{index[0]}", mode="rb"))
    whole_year.m_grw_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/grw/{index[1]}", mode="rb"))
    whole_year.m_fec_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/fec/{index[2]}", mode="rb")) 
    whole_year.m_flow_poi_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + whole_year.truedata_style + f"/flow/{index[3]}", mode="rb"))

    for var, var_samples in zip(whole_year.hmc_helper_sur.current_state, whole_year.samples_sur):
        var.assign(var_samples[index[0]])
    # grw
    for var, var_samples in zip(whole_year.hmc_helper_grw.current_state, whole_year.samples_grw):
        var.assign(var_samples[index[1]])
    # fec
    for var, var_samples in zip(whole_year.hmc_helper_fec.current_state, whole_year.samples_fec):
        var.assign(var_samples[index[2]])
    # flow
    for var, var_samples in zip(whole_year.hmc_helper_flow.current_state, whole_year.samples_flow):
        var.assign(var_samples[index[3]])


    models_now2 = {
        "m_sur": whole_year.m_sur_new ,
        "m_grw": whole_year.m_grw_new,
        "m_fec": whole_year.m_fec_new,
        "m_flow_poi": whole_year.m_flow_poi_new,
        "alpha": whole_year.alpha,
        "beta": whole_year.beta,
        "recruit_p": whole_year.recruit_p
    }

    np.random.seed(4567)
    a = IBM_1step_gp_same(zt=whole_year.z0[0], age=whole_year.z0[1], models=models_now2)

    np.random.seed(4567)
    b = IBM_1step_gp_same_cache(zt=whole_year.z0[0], age=whole_year.z0[1], models=models_now2)
    np.testing.assert_allclose(a, b)

