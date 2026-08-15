import pickle
import os
from tensorflow_probability import distributions as tfd
from s0_fun_base import  XY_grw_nf_compu
import gpflow
f64 = gpflow.utilities.to_default_float
from gpflow.ci_utils import reduce_in_tests as ci_niter
import tensorflow as tf
import tensorflow_probability as tfp


gp_mle_true_population = pickle.load(open(file = os.getcwd()+f"/true_popu/gp_mle_true_population/gp_mle_true_population.pkl", mode="rb"))
df_grw_nf = XY_grw_nf_compu(gp_mle_true_population)   

m_grw_nf_new = gpflow.models.GPR(data=(df_grw_nf[0], df_grw_nf[1]), kernel=gpflow.kernels.RBF())

m_grw_nf_new.kernel.lengthscales.prior = tfd.HalfNormal(scale=f64(100.))
m_grw_nf_new.kernel.variance.prior = tfd.HalfNormal(scale=f64(100.))
m_grw_nf_new.likelihood.variance.prior = tfd.HalfNormal(scale=f64(100.))


# We now sample from the posterior using HMC.
hmc_helper = gpflow.optimizers.SamplingHelper(
    m_grw_nf_new.log_posterior_density, m_grw_nf_new.trainable_parameters
)

num_burnin_steps = ci_niter(5000)
num_samples = ci_niter(5000)

hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=9
)
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    hmc, num_adaptation_steps=int(num_burnin_steps * 0.8), target_accept_prob=f64(0.8), adaptation_rate=0.1
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
pickle.dump(samples, open(file = os.getcwd()+f"/true_popu/gp_mle_true_population/MCMC/gp/sep/samples_grw_nf.pkl", mode="wb"))

