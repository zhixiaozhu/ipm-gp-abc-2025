from re import I
import pandas as pd
import numpy as np
import pickle
import os
import gpflow
import multiprocessing
import ray
import warnings
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
from datetime import datetime
import sys
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.ci_utils import reduce_in_tests as ci_niter
from tensorflow_probability import distributions as tfd
f64 = gpflow.utilities.to_default_float

# Functions for loading data.
# Functions for processing data (used for model fitting and initialization).
# env=False is for seperatly fitting 
def XY_sur_compu(popu_data, env=False):
    index_sur = np.logical_not(np.isnan(popu_data["surv"]))
    size_sur = popu_data["size"][index_sur].to_numpy()
    surv_sur = popu_data["surv"][index_sur].to_numpy()
    age_sur = popu_data["age"][index_sur].to_numpy()
    if env==True:
        tmax_sur = popu_data["tmax"][index_sur].to_numpy()
        tmin_sur = popu_data["tmin"][index_sur].to_numpy() 
        precip_sur = popu_data["precip"][index_sur].to_numpy() 
        X_sur = np.concatenate((size_sur[:, None], age_sur[:, None], tmax_sur[:, None], tmin_sur[:, None], precip_sur[:, None]), axis=1)
    else: 
        X_sur = np.concatenate((size_sur[:, None], age_sur[:, None]), axis=1)

    Y_sur = surv_sur[:, None]

    return (X_sur, Y_sur)

def XY_grw_f_compu(popu_data, env=False):
    index_grw = np.logical_not(np.isnan(popu_data["sizeNext"])) & np.logical_not(np.isnan(popu_data["size"]))
    fec_grw = popu_data["fec"][index_grw]
    
    size_grw_f = popu_data["size"][index_grw][fec_grw == 1].to_numpy()
    sizeNext_grw_f = popu_data["sizeNext"][index_grw][fec_grw == 1].to_numpy()
    age_grw_f = popu_data["age"][index_grw][fec_grw == 1].to_numpy()

    if env==True:
        tmax_grw_f = popu_data["tmax"][index_grw][fec_grw == 1].to_numpy()
        tmin_grw_f = popu_data["tmin"][index_grw][fec_grw == 1].to_numpy()
        precip_grw_f = popu_data["precip"][index_grw][fec_grw == 1].to_numpy()
        X_grw_f = np.concatenate((size_grw_f[:, None], age_grw_f[:, None], tmax_grw_f[:, None], tmin_grw_f[:, None], precip_grw_f[:, None]), axis=1)
    else: 
        X_grw_f = np.concatenate((size_grw_f[:, None], age_grw_f[:, None]), axis=1)
    Y_grw_f = sizeNext_grw_f[:, None]

    return (X_grw_f, Y_grw_f)
    

def XY_grw_nf_compu(popu_data, env=False):
    index_grw = np.logical_not(np.isnan(popu_data["sizeNext"])) & np.logical_not(np.isnan(popu_data["size"]))
    fec_grw = popu_data["fec"][index_grw]
    
    size_grw_nf = popu_data["size"][index_grw][fec_grw == 0].to_numpy()
    sizeNext_grw_nf = popu_data["sizeNext"][index_grw][fec_grw == 0].to_numpy()
    age_grw_nf = popu_data["age"][index_grw][fec_grw == 0].to_numpy()
    
    if env==True:
        tmax_grw_nf = popu_data["tmax"][index_grw][fec_grw == 0].to_numpy()
        tmin_grw_nf = popu_data["tmin"][index_grw][fec_grw == 0].to_numpy()
        precip_grw_nf = popu_data["precip"][index_grw][fec_grw == 0].to_numpy()
        X_grw_nf = np.concatenate((size_grw_nf[:, None], age_grw_nf[:, None], tmax_grw_nf[:, None], tmin_grw_nf[:, None], precip_grw_nf[:, None]), axis=1)
    else: 
        X_grw_nf = np.concatenate((size_grw_nf[:, None], age_grw_nf[:, None]), axis=1)

    Y_grw_nf = sizeNext_grw_nf[:, None]

    return (X_grw_nf, Y_grw_nf)


def XY_fec_compu(popu_data, env=False):
    index_fec = np.logical_not(np.isnan(popu_data["fec"]))
    size_fec = popu_data["size"][index_fec].to_numpy()
    fec_fec = popu_data["fec"][index_fec].to_numpy()
    age_fec = popu_data["age"][index_fec].to_numpy()

    if env==True:
        tmax_fec = popu_data["tmax"][index_fec].to_numpy()
        tmin_fec = popu_data["tmin"][index_fec].to_numpy()
        precip_fec = popu_data["precip"][index_fec].to_numpy()
        X_fec = np.concatenate((size_fec[:, None], age_fec[:, None], tmax_fec[:, None], tmin_fec[:, None], precip_fec[:, None]), axis=1)    
    else: 
        X_fec = np.concatenate((size_fec[:, None], age_fec[:, None]), axis=1)

    Y_fec = fec_fec[:, None]

    return (X_fec, Y_fec)

def XY_flow_compu(popu_data, env=False):
    index_flow = popu_data["fec"] == 1
    size_flow = popu_data["size"][index_flow].to_numpy()
    flow_flow = popu_data["flow"][index_flow].to_numpy()
    age_flow = popu_data["age"][index_flow].to_numpy()

    if env==True:
        tmax_flow = popu_data["tmax"][index_flow].to_numpy()
        tmin_flow = popu_data["tmin"][index_flow].to_numpy()
        precip_flow = popu_data["precip"][index_flow].to_numpy()
        X_flow = np.concatenate((size_flow[:, None], age_flow[:, None], tmax_flow[:, None], tmin_flow[:, None], precip_flow[:, None]), axis=1)
    else: 
        X_flow = np.concatenate((size_flow[:, None], age_flow[:, None]), axis=1)

    Y_flow = flow_flow[:, None] - 1

    return (X_flow, Y_flow)


############################################################################################################################################################# 
# Functions for MCMC sampling (for each single vital rate)

def mcmc_grw_nf(popu_dataset, env, num_burnin_steps = ci_niter(20000), num_samples = ci_niter(5000)):
    # whole = True means that the popu_dataset is for multiple years.

    # Grw_nf
    if env == True:
        m_grw_nf_new = gpflow.models.GPR(data=(XY_grw_nf_compu(popu_dataset, env=True)), 
                                            kernel=gpflow.kernels.RBF(lengthscales=np.array([25,25,25,25,25])), mean_function=None)
                                            #kernel=gpflow.kernels.RBF(lengthscales=np.array([2,20,20,5])), mean_function=None) for 500 samples

    else:
        m_grw_nf_new = gpflow.models.GPR(data=(XY_grw_nf_compu(popu_dataset, env=False)), 
                                            kernel=gpflow.kernels.RBF(lengthscales=np.array([25,25])), mean_function=None)

    # we add priors to the hyperparameters.
    m_grw_nf_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))#tfd.InverseGamma(f64(0.001),f64(0.001))
    m_grw_nf_new.likelihood.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))#tfd.InverseGamma(f64(0.001),f64(0.001))
    m_grw_nf_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))

    optimizer = gpflow.optimizers.Scipy()
    _ = optimizer.minimize(
        m_grw_nf_new.training_loss, m_grw_nf_new.trainable_variables, options=dict(maxiter=3000) #reduce_in_tests(3000))
    )

    hmc_helper_grw_nf = gpflow.optimizers.SamplingHelper(
        m_grw_nf_new.log_posterior_density, m_grw_nf_new.trainable_parameters
    )
    hmc_grw_nf = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper_grw_nf.target_log_prob_fn, num_leapfrog_steps=25, step_size=f64(40)
    )
    adaptive_hmc_grw_nf = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc_grw_nf, num_adaptation_steps=int(0.8*num_burnin_steps), target_accept_prob=f64(0.8), adaptation_rate=f64(0.1)
    )

    @tf.function
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results = 5,
            current_state=hmc_helper_grw_nf.current_state,
            kernel=adaptive_hmc_grw_nf,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        )
    
    print('\n\n\n' + 'Re-generating MCMC samples' + '\n\n\n')
    samples, _ = run_chain_fn()
    parameter_samples = hmc_helper_grw_nf.convert_to_constrained_values(samples)

    return (samples, parameter_samples)



def mcmc_grw_f(popu_dataset, env, num_burnin_steps = ci_niter(20000), num_samples = ci_niter(5000)):

    if env == True:
        m_grw_f_new = gpflow.models.GPR(data=(XY_grw_f_compu(popu_dataset, env=True)), 
                                        kernel=gpflow.kernels.RBF(lengthscales=np.array([25,25,25,25,25])), mean_function=None)
                                        #kernel=gpflow.kernels.RBF(lengthscales=np.array([2,20,20,5])), mean_function=None) for 500 samples
    else:
        m_grw_f_new = gpflow.models.GPR(data=(XY_grw_f_compu(popu_dataset, env=False)), 
                                        kernel=gpflow.kernels.RBF(lengthscales=np.array([25,25])), mean_function=None)


    m_grw_f_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
    m_grw_f_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))#tfd.InverseGamma(f64(0.001),f64(0.001))
    m_grw_f_new.likelihood.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))#tfd.InverseGamma(f64(0.001),f64(0.001))


    optimizer = gpflow.optimizers.Scipy()
    _ = optimizer.minimize(
        m_grw_f_new.training_loss, m_grw_f_new.trainable_variables, options=dict(maxiter=3000) #reduce_in_tests(3000))
    )

    hmc_helper_grw_f = gpflow.optimizers.SamplingHelper(
        m_grw_f_new.log_posterior_density, m_grw_f_new.trainable_parameters
    )

    hmc_grw_f = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper_grw_f.target_log_prob_fn, num_leapfrog_steps=25, step_size=f64(30)
    )
    adaptive_hmc_grw_f = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc_grw_f, num_adaptation_steps=int(0.8*num_burnin_steps), target_accept_prob=f64(0.8), adaptation_rate=f64(0.1)
    )

    @tf.function
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results = 5,
            current_state=hmc_helper_grw_f.current_state,
            kernel=adaptive_hmc_grw_f,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        )
    
    print('\n\n\n' + 'Re-generating MCMC samples' + '\n\n\n')
    samples, _ = run_chain_fn()
    parameter_samples = hmc_helper_grw_f.convert_to_constrained_values(samples)

    return (samples, parameter_samples)


def mcmc_fec(popu_dataset, env, num_burnin_steps = ci_niter(25000), num_samples = ci_niter(5000)):

    # Fec
    if env == True:
        m_fec_new = gpflow.models.GPMC(data=(XY_fec_compu(popu_dataset, env=True)), 
                                        kernel=gpflow.kernels.RBF(lengthscales=np.array([2.8,300,100,100,1.13]), variance=5), 
                                        likelihood=gpflow.likelihoods.Bernoulli())
    else:
        m_fec_new = gpflow.models.GPMC(data=(XY_fec_compu(popu_dataset, env=False)), 
                                        kernel=gpflow.kernels.RBF(lengthscales=np.array([30,30])), 
                                        likelihood=gpflow.likelihoods.Bernoulli())       
                                    
    m_fec_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))#tfd.InverseGamma(f64(0.001),f64(0.001))
    m_fec_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))

    hmc_helper_fec = gpflow.optimizers.SamplingHelper(
        m_fec_new.log_posterior_density, m_fec_new.trainable_parameters
    )

    hmc_fec = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper_fec.target_log_prob_fn, num_leapfrog_steps=15, step_size=f64(5)
    )
    adaptive_hmc_fec = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc_fec, num_adaptation_steps=int(0.8*num_burnin_steps), target_accept_prob=f64(0.8), adaptation_rate=f64(0.1)
    )

    @tf.function
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results = 3,
            current_state=hmc_helper_fec.current_state,
            kernel=adaptive_hmc_fec,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        )
    
    print('\n\n\n' + 'Re-generating MCMC samples' + '\n\n\n')
    samples, _ = run_chain_fn()
    parameter_samples = hmc_helper_fec.convert_to_constrained_values(samples)

    return (samples, parameter_samples)


def mcmc_flow_poi(popu_dataset, env, num_burnin_steps = ci_niter(25000), num_samples = ci_niter(5000)):

    # Flow poi
    if env == True:
        m_flow_poi_new = gpflow.models.GPMC(data=(XY_flow_compu(popu_dataset, env=True)), 
                                            kernel=gpflow.kernels.RBF(lengthscales=np.array([0.79,29.75,13.0,7.995,0.798]), variance=2.644), 
                                            likelihood=gpflow.likelihoods.Poisson())
    else:
        m_flow_poi_new = gpflow.models.GPMC(data=(XY_flow_compu(popu_dataset, env=False)), 
                                            kernel=gpflow.kernels.RBF(lengthscales=np.array([10,30])), 
                                            likelihood=gpflow.likelihoods.Poisson())    

    m_flow_poi_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
    m_flow_poi_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))#tfd.InverseGamma(f64(0.001),f64(0.001))

    optimizer = gpflow.optimizers.Scipy()
    _ = optimizer.minimize(
        m_flow_poi_new.training_loss, m_flow_poi_new.trainable_variables, options=dict(maxiter=3000) #reduce_in_tests(3000))
    )

    hmc_helper_flow = gpflow.optimizers.SamplingHelper(
        # m_flow_poi_new.log_posterior_density, m_flow_poi_new.trainable_parameters
        m_flow_poi_new.log_posterior_density, m_flow_poi_new.trainable_parameters
    )

    hmc_flow = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper_flow.target_log_prob_fn, num_leapfrog_steps=25, step_size=f64(13)
    )
    adaptive_hmc_flow = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc_flow, num_adaptation_steps=int(0.8*num_burnin_steps), target_accept_prob=f64(0.8), adaptation_rate=f64(0.1)
    )

    @tf.function
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results = 7,
            current_state=hmc_helper_flow.current_state,
            kernel=adaptive_hmc_flow,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        )
    
    print('\n\n\n' + 'Re-generating MCMC samples' + '\n\n\n')
    samples, _ = run_chain_fn()
    parameter_samples = hmc_helper_flow.convert_to_constrained_values(samples)

    return (samples, parameter_samples)

def mcmc_sur(popu_dataset, env, num_burnin_steps = ci_niter(25000), num_samples = ci_niter(5000)):

    # Sur
    if env == True:
        m_sur_new = gpflow.models.GPMC(data=(XY_sur_compu(popu_dataset, env=True)), 
                                    kernel=gpflow.kernels.RBF(lengthscales=np.array([1.86, 8.9, 53.6,1.04,1.48]), variance=0.747) , 
                                    likelihood=gpflow.likelihoods.Bernoulli())
    else:
        m_sur_new = gpflow.models.GPMC(data=(XY_sur_compu(popu_dataset, env=False)), 
                                    kernel=gpflow.kernels.RBF(lengthscales=np.array([30,30])) , 
                                    likelihood=gpflow.likelihoods.Bernoulli()) 


    m_sur_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
    m_sur_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))#tfd.InverseGamma(f64(0.001),f64(0.001))

    hmc_helper_sur = gpflow.optimizers.SamplingHelper(
        m_sur_new.log_posterior_density, m_sur_new.trainable_parameters
    )

    hmc_sur = tfp.mcmc.HamiltonianMonteCarlo(
        #target_log_prob_fn=hmc_helper_sur.target_log_prob_fn, num_leapfrog_steps=25, step_size=f64(13)
        target_log_prob_fn=hmc_helper_sur.target_log_prob_fn, num_leapfrog_steps=10, step_size=f64(10)

    )
    adaptive_hmc_sur = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc_sur, num_adaptation_steps=int(0.8*num_burnin_steps), target_accept_prob=f64(0.8), adaptation_rate=f64(0.1)
    )

    @tf.function
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            #num_steps_between_results = 7,
            current_state=hmc_helper_sur.current_state,
            kernel=adaptive_hmc_sur,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        )
    
    print('\n\n\n' + 'Re-generating MCMC samples' + '\n\n\n')
    samples, _ = run_chain_fn()
    parameter_samples = hmc_helper_sur.convert_to_constrained_values(samples)

    return (samples, parameter_samples)




def popu_structure_noid(data):
    # given dataset at t-1, popu_structure_noid would return the starting population structure at time t
    survivor_index = data["surv"] == 1
    new_born_index = np.isnan(data["size"])
    new = data.loc[survivor_index | new_born_index]
    return (new.sizeNext, new.ageNext)

def popu_structure_noid_current(data):
    # given dataset at t, popu_structure_noid_current would return the starting population structure at time t
    index = np.logical_not(np.isnan(data["size"]))
    new = data.loc[index]
    return (new['size'], new['age'])





# IBM simulation

def IBM_1step(zt, age, models, env, tmax=np.nan, tmin=np.nan, precip=np.nan):
    # if (env==True) and (np.isnan(tmax) or np.isnan(precip)):
    #     assert False, '\n Please indicate the precipitation and temperature when considering considering env.'

    # initialize 
    current_n = np.shape(zt)[0]
    rep_stalks = np.repeat(np.nan, current_n)[:, None]
    surv = np.repeat(np.nan, current_n)[:, None]
    zprime = np.repeat(np.nan, current_n)[:, None]
    ageprime = np.repeat(np.nan, current_n)[:, None]
    num_recruits = 0 
    zt = np.array(zt)[:, None]
    age = np.array(age)[:, None]

    if env==True:
        x_tmax = np.repeat(tmax, age.shape[0])
        x_tmin = np.repeat(tmin, age.shape[0])
        x_precip = np.repeat(precip, age.shape[0])
        X_t = np.concatenate((zt, age, x_tmax[:, None], x_tmin[:, None], x_precip[:, None]), axis=1)
    else: 
        X_t = np.concatenate((zt, age), axis=1)

    # we simulate those breeders first
    p_breeding = models["m_fec"].predict_y(X_t)[0]
    #p_breeding = models["m_fec"].predict_y(X_t)[0]
    rep_breeding = np.random.binomial(n = 1, p = p_breeding)
    whether_bre = (rep_breeding == 1).reshape(1, current_n)[0]        
    num_bre = np.sum(whether_bre)
    num_nonbre = current_n - num_bre

    lambda_t = models["m_flow_poi"].predict_y(X_t[whether_bre, :])[0]
    if np.any(np.isnan(lambda_t)):
        warnings.warn('nan lambda_t produced!!')
        print('nan lambda_t produced!!', flush=True)
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)

    if np.any(np.array(lambda_t) >= 10000):
        warnings.warn('lambda_t is too large !!')
        print('lambda_t is too large !!', flush=True) 
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)
    else:
        rep_stalks[whether_bre] = np.random.poisson(lambda_t) + 1
           
    # simulate recruits based on the number of flowering stalks
    if (num_bre != 0):
        num_recruits = np.random.binomial(n = np.nansum(rep_stalks), p = models["recruit_p"])
        # assign size for those recruits
        rep_size = stats.gamma(a = models["alpha"], scale = 1/models["beta"]).rvs(size=num_recruits)[:, None]
            
        
    # now, simulating surviving.
    p_surv = models["m_sur"].predict_y(X_t)[0]
    #p_surv = models["m_sur"].predict_y(X_t)[0]
    surv = np.random.binomial(n = 1, p = p_surv)
    whether_surv = (surv == 1).reshape(1, current_n)[0]
    num_surv = np.sum(whether_surv)

    # let these survivors grow up
    # for breeders
    X_t_f = X_t[whether_surv & whether_bre]
    mean_zprime_f = models["m_grw_f"].predict_y(X_t_f)
    #mean_zprime_f = models["m_grw_f"].predict_y(X_t_f)
    zprime[whether_surv & whether_bre] = np.random.normal(mean_zprime_f[0], 
                                                          np.sqrt(mean_zprime_f[1]))
    
    # for non-breeders
    not_bre = whether_bre == False
    X_t_nf = X_t[whether_surv & not_bre]
    mean_zprime_nf = models["m_grw_nf"].predict_y(X_t_nf)
    #mean_zprime_nf = models["m_grw_nf"].predict_y(X_t_nf)
    zprime[whether_surv & not_bre] = np.random.normal(mean_zprime_nf[0], 
                                                      np.sqrt(mean_zprime_nf[1]))
    
    # in our case, age 8 is an absorbing state.
    ageprime[whether_surv] = np.minimum(age[whether_surv]+1, 8)
    
    

    # store the simulation data 
    # Here: age is for the age at current not ageNext
    if (num_bre != 0):
        zprime = np.concatenate((zprime, rep_size))
        zt = np.concatenate((zt, np.repeat(np.nan, num_recruits)[:, None]))
        rep_breeding = np.concatenate((rep_breeding, np.repeat(np.nan, num_recruits)[:, None]))
        rep_stalks = np.concatenate((rep_stalks, np.repeat(np.nan, num_recruits)[:, None]))
        surv = np.concatenate((surv, np.repeat(np.nan, num_recruits)[:, None]))
        age = np.concatenate((age, np.repeat(np.nan, num_recruits)[:, None]))
        ageprime = np.concatenate((ageprime, np.repeat(1, num_recruits)[:, None]))

    data = pd.DataFrame(np.concatenate((zt, zprime, rep_breeding, rep_stalks, surv, age, ageprime), axis=1),
                        columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

    return(data)


# Cache
from typing import Optional, Tuple

import tensorflow as tf
from check_shapes import check_shapes

import gpflow

from gpflow import posteriors
from gpflow.base import InputData, MeanAndVariance
from gpflow.posteriors import VGPPosterior
from gpflow.models.model import GPModel
from gpflow.models.gpr import GPR_deprecated


def predict_f_loaded_cache(
    model: GPModel, 
    Xnew: InputData,
    Cache: Optional[Tuple[tf.Tensor, ...]],
    full_cov: bool = False,
    full_output_cov: bool = False,
) -> MeanAndVariance:
    """
    For backwards compatibility, GPR's predict_f uses the fused (no-cache)
    computation, which is more efficient during training.

    For faster (cached) prediction, predict directly from the posterior object, i.e.,:
        model.posterior().predict_f(Xnew, ...)
    """

    if isinstance(model, gpflow.models.gpr.GPR):
        posterior = model.posterior(posteriors.PrecomputeCacheType.NOCACHE)
        posterior.cache = Cache
        return posterior.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
    
    elif isinstance(model, gpflow.models.gpmc.GPMC):
        X_data, _Y_data = model.data
        posterior = VGPPosterior(
            kernel = model.kernel,
            X = X_data,
            q_mu = model.V,
            q_sqrt = None,
            white = True,
            precompute_cache=None,
        )
        posterior.cache = Cache
        return posterior.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        
    else:
        raise ValueError(f"{model} is not a supported GPmodel type for faster predictions.")
         


def predict_y_loaded_cache(
    model: GPModel, 
    Xnew: InputData,
    Cache: Optional[Tuple[tf.Tensor, ...]],
    full_cov: bool = False,
    full_output_cov: bool = False,
) -> MeanAndVariance:
    """
    For backwards compatibility, GPR's predict_f uses the fused (no-cache)
    computation, which is more efficient during training.

    For faster (cached) prediction, predict directly from the posterior object, i.e.,:
        model.posterior().predict_f(Xnew, ...)
    """

    f_mean, f_var = predict_f_loaded_cache(
        model=model, Xnew=Xnew, Cache=Cache, full_cov=full_cov, full_output_cov=full_output_cov
    )

    return model.likelihood.predict_mean_and_var(Xnew, f_mean, f_var)



def GPMC_posterior(
    model: gpflow.models.gpmc.GPMC,
    precompute_cache: posteriors.PrecomputeCacheType = posteriors.PrecomputeCacheType.TENSOR,
) -> posteriors.VGPPosterior:
    
    X_data, _Y_data = model.data
    return posteriors.VGPPosterior(
        kernel=model.kernel,
        X=X_data,
        q_mu=model.V,
        q_sqrt=None,
        white=True,
        precompute_cache=precompute_cache,
    )

# IBM_1step_cashe make predictions based on the GP models with cached informations
def IBM_1step_cashe(zt, age, models, env, tmax=np.nan, tmin=np.nan, precip=np.nan):

    # initialize 
    current_n = np.shape(zt)[0]
    rep_stalks = np.repeat(np.nan, current_n)[:, None]
    surv = np.repeat(np.nan, current_n)[:, None]
    zprime = np.repeat(np.nan, current_n)[:, None]
    ageprime = np.repeat(np.nan, current_n)[:, None]
    num_recruits = 0 
    zt = np.array(zt)[:, None]
    age = np.array(age)[:, None]

    if env==True:
        x_tmax = np.repeat(tmax, age.shape[0])
        x_tmin = np.repeat(tmin, age.shape[0])
        x_precip = np.repeat(precip, age.shape[0])
        X_t = np.concatenate((zt, age, x_tmax[:, None], x_tmin[:, None], x_precip[:, None]), axis=1)
    else: 
        X_t = np.concatenate((zt, age), axis=1)

    # we simulate those breeders first
    p_breeding = predict_y_loaded_cache(model=models["m_fec"], Xnew=X_t, Cache=models["m_fec"].cache)[0]
    rep_breeding = np.random.binomial(n = 1, p = p_breeding)
    whether_bre = (rep_breeding == 1).reshape(1, current_n)[0]        
    num_bre = np.sum(whether_bre)
    num_nonbre = current_n - num_bre

    lambda_t = predict_y_loaded_cache(model=models["m_flow_poi"], Xnew=X_t[whether_bre, :], Cache=models["m_flow_poi"].cache)[0]
    if np.any(np.isnan(lambda_t)):
        warnings.warn('nan lambda_t produced!!')
        print('nan lambda_t produced!!', flush=True)
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)

    if np.any(np.array(lambda_t) >= 10000):
        warnings.warn('lambda_t is too large !!')
        print('lambda_t is too large !!', flush=True) 
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)
    else:
        rep_stalks[whether_bre] = np.random.poisson(lambda_t) + 1
           
    # simulate recruits based on the number of flowering stalks
    if (num_bre != 0):
        num_recruits = np.random.binomial(n = np.nansum(rep_stalks), p = models["recruit_p"])
        # assign size for those recruits
        rep_size = stats.gamma(a = models["alpha"], scale = 1/models["beta"]).rvs(size=num_recruits)[:, None]
            
        
    # now, simulating surviving.
    p_surv = predict_y_loaded_cache(model=models["m_sur"], Xnew=X_t, Cache=models["m_sur"].cache)[0]
    surv = np.random.binomial(n = 1, p = p_surv)
    whether_surv = (surv == 1).reshape(1, current_n)[0]
    num_surv = np.sum(whether_surv)

    # let these survivors grow up
    # for breeders
    X_t_f = X_t[whether_surv & whether_bre]
    mean_zprime_f = predict_y_loaded_cache(model=models["m_grw_f"], Xnew=X_t_f, Cache=models["m_grw_f"].cache) 
    # mean_zprime_f = models["m_grw_f"].predict_y(X_t_f)
    zprime[whether_surv & whether_bre] = np.random.normal(mean_zprime_f[0], 
                                                          np.sqrt(mean_zprime_f[1]))
    
    # for non-breeders
    not_bre = whether_bre == False
    X_t_nf = X_t[whether_surv & not_bre]
    mean_zprime_nf = predict_y_loaded_cache(model=models["m_grw_nf"], Xnew=X_t_nf, Cache=models["m_grw_nf"].cache)
    zprime[whether_surv & not_bre] = np.random.normal(mean_zprime_nf[0], 
                                                      np.sqrt(mean_zprime_nf[1]))
    
    # in our case, age 8 is an absorbing state.
    ageprime[whether_surv] = np.minimum(age[whether_surv]+1, 8)
    
    # store the simulation data 
    # Here: age is for the gae at current not ageNext
    if (num_bre != 0):
        zprime = np.concatenate((zprime, rep_size))
        zt = np.concatenate((zt, np.repeat(np.nan, num_recruits)[:, None]))
        rep_breeding = np.concatenate((rep_breeding, np.repeat(np.nan, num_recruits)[:, None]))
        rep_stalks = np.concatenate((rep_stalks, np.repeat(np.nan, num_recruits)[:, None]))
        surv = np.concatenate((surv, np.repeat(np.nan, num_recruits)[:, None]))
        age = np.concatenate((age, np.repeat(np.nan, num_recruits)[:, None]))
        ageprime = np.concatenate((ageprime, np.repeat(1, num_recruits)[:, None]))

    data = pd.DataFrame(np.concatenate((zt, zprime, rep_breeding, rep_stalks, surv, age, ageprime), axis=1),
                        columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

    return(data)


# Test stats

from test_stats import num_ffs_group, D_chi2, fer_dis, D_emd_p, zero_correction, D_hilbert, D_ssd
from test_stats import prob_indi_group, age_group, size_group, zero_fixed, sizeNext_group
# a function used to calculate all summary statst interested



def list_comparisons_interested(exp_1step_data, simu_1step_data):
    current_s = np.array(0)
    size_group_simu = size_group(simu_1step_data)
    size_group_exp = size_group(exp_1step_data)
    sizeNext_group_simu = sizeNext_group(simu_1step_data) 
    sizeNext_group_exp = sizeNext_group(exp_1step_data)

    # (15) # survivors 
    #              in age-size groups
    s15_simudata = age_group(size_group_simu)
    result = getattr(s15_simudata.groupby(["label_age", "label_size"])["surv"], 'sum')
    s15 = zero_fixed(result=result(), bysize=True, byage=True)
    s15_expdata = age_group(size_group_exp)
    result = getattr(s15_expdata.groupby(["label_age", "label_size"])["surv"], 'sum')
    exp15 = zero_fixed(result=result(), bysize=True, byage=True) 
    # (a) Chi2 distance metric comes with the correcation.
    current_s = np.append(current_s, 
                          D_chi2(y_exp=exp15, y_obs=s15, details=False))

    # (27)(e) & (25)(g)
    result = sizeNext_group_simu.groupby("label_sizeNext")['fec']
    result_sum = result.sum()
    result_sum = zero_fixed(result=result_sum, bysize=True, byage=False)
    result_total = result.count()
    result_total = zero_fixed(result=result_total, bysize=True, byage=False)
    result_n = result_total - result_sum

    # (27) non-breeders' sizeNext and ageNext distribution
    s27 = pd.Series(result_n.values / result_n.values.sum(), index=result_n.index, name=result_n.name)
    # (25) breeders' sizeNext and ageNext distribution 
    s25 = pd.Series(result_sum.values / result_sum.values.sum(), index=result_sum.index, name=result_sum.name)

    # (27) & (25)
    result = sizeNext_group_exp.groupby("label_sizeNext")['fec']
    result_sum = result.sum()
    result_sum = zero_fixed(result=result_sum, bysize=True, byage=False)
    result_total = result.count()
    result_total = zero_fixed(result=result_total, bysize=True, byage=False)
    result_n = result_total - result_sum
    # (27) non-breeders' sizeNext and ageNext distribution
    exp27 = pd.Series(result_n.values / result_n.values.sum(), index=result_n.index, name=result_n.name)
    # (25) breeders' sizeNext and ageNext distribution 
    exp25 = pd.Series(result_sum.values / result_sum.values.sum(), index=result_sum.index, name=result_sum.name)

    # (e) EMD 
    current_s = np.append(current_s, 
                          D_emd_p(p_exp=exp27, p_obs=s27))
    # (g) Hilbert projective metric 
    (p_exp_t_fixed, p_obs_t_fixed) = zero_correction(p_exp=exp25, p_obs=s25)
    current_s = np.append(current_s, 
                          D_hilbert(p_exp_0fixed=p_exp_t_fixed, p_obs_0fixed=p_obs_t_fixed))


    # (19) the mean/propotion of # breeders 
    #                                            in size groups
    result = getattr(size_group_simu.groupby("label_size")['fec'], 'mean')
    s19 = zero_fixed(result=result(), bysize=True, byage=False)
    result = getattr(size_group_exp.groupby("label_size")['fec'], 'mean')
    exp19 = zero_fixed(result=result(), bysize=True, byage=False)
    # (b) Sum of squared differences
    current_s = np.append(current_s, 
                          D_ssd(y_exp=exp19, y_obs=s19, details=False))

    # (7) # flowering stalks
    #              in size groups
    result = getattr(size_group_simu.groupby("label_size")['flow'], 'sum')
    s7 = zero_fixed(result=result(), bysize=True, byage=False)
    result = getattr(size_group_exp.groupby("label_size")['flow'], 'sum')
    exp7 = zero_fixed(result=result(), bysize=True, byage=False)
    # (b) Sum of squared differences
    current_s = np.append(current_s, 
                          D_ssd(y_exp=exp7, y_obs=s7, details=False))
    
    current_s = np.delete(current_s, 0)
    return current_s

