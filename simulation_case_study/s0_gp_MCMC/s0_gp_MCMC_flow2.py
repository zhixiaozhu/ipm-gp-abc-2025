import pickle
import os
from tensorflow_probability import distributions as tfd
from s0_fun_base import XY_flow_compu
import gpflow
import numpy as np
f64 = gpflow.utilities.to_default_float
from gpflow.ci_utils import reduce_in_tests as ci_niter
import tensorflow as tf
import tensorflow_probability as tfp


gp_mle_true_population = pickle.load(open(file = os.getcwd()+f"/true_popu/gp_mle_true_population/gp_mle_true_population.pkl", mode="rb"))
df_flow = XY_flow_compu(gp_mle_true_population)  

m_flow_new = gpflow.models.GPMC(data=(df_flow[0], df_flow[1]), 
                                kernel=gpflow.kernels.RBF(), likelihood=gpflow.likelihoods.Poisson())

# Secondly, we add priors to the hyperparameters.
m_flow_new.kernel.lengthscales.prior = tfd.HalfNormal(scale=f64(100.))
m_flow_new.kernel.variance.prior = tfd.HalfNormal(scale=f64(100.))

# We now sample from the posterior using HMC.
hmc_helper = gpflow.optimizers.SamplingHelper(
    m_flow_new.log_posterior_density, m_flow_new.trainable_parameters
)
num_burnin_steps = ci_niter(5000)
num_samples = ci_niter(5000)

hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=3
)
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    hmc, num_adaptation_steps=int(num_burnin_steps * 0.8), target_accept_prob=f64(0.78), adaptation_rate=0.1
)

@tf.function
def run_chain_fn():
    return tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin_steps,
        current_state=hmc_helper.current_state,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )

print('\n\n\n' + 'Re-generating MCMC samples' + '\n\n\n')
samples, traces = run_chain_fn()

print('\n\n\n' + 'Re-generating MCMC: DONE' + '\n\n\n')

pickle.dump(samples, open(file = os.getcwd()+f"/true_popu/gp_mle_true_population/MCMC/gp/sep/samples_flow.pkl", mode="wb"))