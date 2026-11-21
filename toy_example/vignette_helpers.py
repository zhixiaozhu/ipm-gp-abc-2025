"""Helper functions for toy_example/vignette_new.ipynb.

The code below was moved out of the notebook without changing it, so that the
notebook stays short. 
"""
import contextlib
import io
import logging
import os
import pickle
import sys
import types
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) # toy_example/
REPO = os.path.dirname(HERE) # repository root
SIM = os.path.join(REPO, "simulation_case_study") # the analysis code
TOY_DATA = os.path.join(HERE, "toy_data") # the toy dataset
SANDBOX = os.path.join(HERE, "sandbox")  # files written by the notebook

# the directory the real code expects under its working directory
TP = "true_popu/glm_mle_true_population"


# --Setup---------------------------------------------------------------

def setup():
    """Make the analysis code importable, create sandbox/ and work inside it."""
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    sys.path.insert(0, SIM)  # so `import s0_fun_base` finds the real code
    for sub in ([TP, TP + "/MCMC/gp/sep", TP + "/MCMC/gp/nonsep"]
                + [f"true_popu/mcmc/Lm/glm/{r}" for r in ["sur","grw_f","grw_nf","grw","fec","flow"]]
                + ["ABC_details/glmsep"]):
        os.makedirs(os.path.join(SANDBOX, sub), exist_ok=True)
    os.chdir(SANDBOX)
    try:   # sharper figures in Jupyter: same size on screen, twice the pixels
        from IPython import get_ipython
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    except Exception:
        pass
    print("Setup done: the notebook now reads and writes files in toy_example/sandbox/.")


# --Step 0--------------------------------------------------------------

def load_toy_data(N_ROWS=200, SEED=0):
    """Load the toy dataset (or first time: subsample the source D_glm)."""
    toy_path = os.path.join(TOY_DATA, "toy_population.pkl")
    if os.path.exists(toy_path):
        toy = pickle.load(open(toy_path, "rb")) # committed tiny dataset
    else:  # first time: subsample the source
        src = os.path.join(SIM, TP, "glm_mle_true_population.pkl")
        toy = pickle.load(open(src, "rb")).sample(n=N_ROWS, random_state=SEED).reset_index(drop=True)
        os.makedirs(TOY_DATA, exist_ok=True); pickle.dump(toy, open(toy_path, "wb"))
    # the real code loads the population from its working directory:
    pickle.dump(toy, open(TP + "/glm_mle_true_population.pkl", "wb"))
    return toy


# --Step 1--------------------------------------------------------------

def run_mcmc(model, step_size, N_DRAWS, N_BURNIN, N_LEAPFROG):
    "Sample the GP hyperparameters (and latent values, for GPMC) by MCMC (HMC)."
    import gpflow, tensorflow as tf, tensorflow_probability as tfp
    tf.get_logger().setLevel("ERROR")
    f64 = gpflow.utilities.to_default_float
    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters)
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn,
        num_leapfrog_steps=N_LEAPFROG, step_size=f64(step_size))
    adaptive = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc, num_adaptation_steps=int(N_BURNIN*0.7),
        target_accept_prob=f64(0.8), adaptation_rate=f64(0.1))
    @tf.function
    def run():
        return tfp.mcmc.sample_chain(
            num_results=N_DRAWS, num_burnin_steps=N_BURNIN,
            current_state=hmc_helper.current_state, kernel=adaptive,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)
    return run()[0]


def make_gpmc(data, likelihood, prior_scale):
    import gpflow
    from tensorflow_probability import distributions as tfd
    f64 = gpflow.utilities.to_default_float
    m = gpflow.models.GPMC(data=data, kernel=gpflow.kernels.RBF(), likelihood=likelihood)
    m.kernel.variance.prior = tfd.HalfNormal(f64(prior_scale))
    m.kernel.lengthscales.prior = tfd.HalfNormal(f64(prior_scale))
    return m


def make_gpr(data):
    import gpflow
    from tensorflow_probability import distributions as tfd
    f64 = gpflow.utilities.to_default_float
    m = gpflow.models.GPR(data=data, kernel=gpflow.kernels.RBF())
    m.kernel.lengthscales.prior = tfd.HalfNormal(f64(100.))
    m.kernel.variance.prior = tfd.HalfNormal(f64(100.))
    m.likelihood.variance.prior = tfd.HalfNormal(f64(100.))
    return m


def fit_other_gps(toy, samples_sur, N_DRAWS, N_BURNIN, N_LEAPFROG):
    """The remaining five vital rates, same pattern as survival.

    Returns {vital rate: (model, MCMC draws)}, used for plotting.
    """
    import gpflow
    from s0_fun_base import (XY_fec_compu, XY_flow_compu, XY_grw_f_compu,
                             XY_grw_nf_compu, XY_grw_compu)
    fits = {}
    RATES = [ # name, kind, XY builder, prior scale, step size, sep/nonsep folder
        ("fec", "bern", XY_fec_compu, 100., 1.0, "sep"),
        ("flow", "poi", XY_flow_compu, 100., 0.01, "sep"),
        ("grw_f", "gpr", XY_grw_f_compu, None, 0.5, "sep"),
        ("grw_nf", "gpr", XY_grw_nf_compu,None, 0.5, "sep"),
        ("grw", "gpr", XY_grw_compu, None, 0.5, "nonsep"),
    ]
    pickle.dump(samples_sur, open(f"{TP}/MCMC/gp/sep/samples_sur.pkl", "wb"))
    for name, kind, xy, ks, step, sub in RATES:
        d = xy(toy)
        if kind == "bern": m = make_gpmc((d[0], d[1]), gpflow.likelihoods.Bernoulli(), ks)
        elif kind == "poi":  m = make_gpmc((d[0], d[1]), gpflow.likelihoods.Poisson(), ks)
        else: m = make_gpr((d[0], d[1]))
        samples = run_mcmc(m, step, N_DRAWS, N_BURNIN, N_LEAPFROG)
        pickle.dump(samples, open(f"{TP}/MCMC/gp/{sub}/samples_{name}.pkl", "wb"))
        fits[name] = (m, samples)
        print(f" fitted {name} on {d[0].shape[0]} data points")
    print("all vital rate fitted by MCMC and saved.")
    return fits


# --Step 2--------------------------------------------------------------

def mle_models(popu, alpha, beta, recruit_p):
    """Maximum-likelihood GP fit of each vital rate (as in s0_preprocess.ipynb)."""
    import gpflow
    from s0_fun_base import (XY_sur_compu, XY_fec_compu, XY_flow_compu,
                             XY_grw_f_compu, XY_grw_nf_compu)
    def mle_vgp(xy, likelihood):
        X, Y = xy(popu)
        m = gpflow.models.VGP((X, Y), kernel=gpflow.kernels.RBF(), likelihood=likelihood)
        gpflow.optimizers.Scipy().minimize(m.training_loss, variables=m.trainable_variables, options=dict(maxiter=10000))
        return m
    def mle_gpr(xy):
        X, Y = xy(popu)
        m = gpflow.models.GPR(data=(X, Y), kernel=gpflow.kernels.RBF())
        gpflow.optimizers.Scipy().minimize(m.training_loss, variables=m.trainable_variables, options=dict(maxiter=10000))
        return m
    return {'m_sur': mle_vgp(XY_sur_compu, gpflow.likelihoods.Bernoulli()),
            'm_fec': mle_vgp(XY_fec_compu, gpflow.likelihoods.Bernoulli()),
            'm_flow_poi': mle_vgp(XY_flow_compu, gpflow.likelihoods.Poisson()),
            'm_grw_f': mle_gpr(XY_grw_f_compu),
            'm_grw_nf': mle_gpr(XY_grw_nf_compu),
            'alpha': alpha, 'beta': beta, 'recruit_p': recruit_p}


def nlpp_of_draws(model, samples):
    """NLPP (negative log posterior probability) of each MCMC draw, and the draws on the constrained scale."""
    import gpflow
    hmc_helper = gpflow.optimizers.SamplingHelper(model.log_posterior_density, model.trainable_parameters)
    constrained_samples = hmc_helper.convert_to_constrained_values(samples)
    nlpp = []
    for i in range(int(samples[0].shape[0])):
        for var, var_samples in zip(hmc_helper.current_state, samples):
            var.assign(var_samples[i])
        nlpp.append(-float(model.log_posterior_density()))
    return constrained_samples, np.array(nlpp)


def perturbed_ipm(popu, models_mle, target_model, constrained_samples, nlpp, near_percentage):
    """The analysis code's Perted_IPM object, with the fixed settings of this example (sep growth, D_glm)."""
    from s0_class_GP_IPM import Perted_IPM
    return Perted_IPM(popu_data=popu, GPmodel_true=models_mle, grw_setting='sep', target=target_model,
                      truedata_style='glm', mcmc_para_sample=constrained_samples,
                      nlog_post=nlpp, opt_percentage=near_percentage)


def model_set(ipm_pert, target_model, constrained_samples, i):
    """All vital rates at their MLE fits, except the target vital rate at MCMC draw i."""
    m = getattr(ipm_pert, 'build_new_' + target_model)
    for var, var_samples in zip(m.parameters, constrained_samples):
        var.assign(var_samples[i])
    return ipm_pert.build_models(m)


def discrepancies(models_1, models_2, seed, rep=100):
    """Simulate rep pairs of datasets (one from each model set) and compute the summary statistics of each pair."""
    import pandas as pd
    from s0_fun_IBMs import IBM_1step_gp
    from s0_fun_IPMs import kernel_setting
    from s0_fun_ss import list_comparisons_interested
    a = kernel_setting(m_size=100, lower=0, upper=5)
    full_data = pd.DataFrame({'size': a[0], 'age': np.repeat(1, len(a[0]))})   # complete dataset for one-step projection
    np.random.seed(seed)
    out = []
    for _ in range(rep):
        d1 = IBM_1step_gp(zt=full_data["size"], age=full_data["age"], models=models_1)
        d2 = IBM_1step_gp(zt=full_data["size"], age=full_data["age"], models=models_2)
        out.append(np.array(list_comparisons_interested(exp_1step_data=d1, simu_1step_data=d2,
                                                        truedata_style="glm")).ravel())
    return np.array(out)


def auc_table(s_ref, s_near):
    """AUC of each statistic for 'theta_opt vs theta_k' against the baseline 'theta_opt vs theta_opt'."""
    import pandas as pd
    from sklearn.metrics import roc_curve, auc
    rep = s_ref.shape[0]   # number of simulated pairs per model
    return pd.DataFrame(
        {f"theta_k = draw {k}": [auc(*roc_curve(np.r_[np.ones(rep), np.zeros(rep)],
                                                np.r_[s_ref[:, j], s_near[k][:, j]], pos_label=0)[:2])
                                 for j in range(s_ref.shape[1])]
         for k in s_near}, index=[f"statistic ({j + 1})" for j in range(s_ref.shape[1])]).T


# --Step 3--------------------------------------------------------------

def build_whole(popu, ALPHA, BETA, RECRUIT_P):
    "Attach the GP models + MCMC draws to an ipmmcmc_whole (as s1_cache_produce.py / s2_ABCSMC_sep.py do)."
    import gpflow
    from s0_class_ABCPMC import ipmmcmc_whole
    from s0_fun_IBMs import popu_structure
    from s0_fun_base import (XY_sur_compu, XY_fec_compu, XY_flow_compu,
                             XY_grw_f_compu, XY_grw_nf_compu, XY_grw_compu)
    wy = ipmmcmc_whole(popu_data=popu, grw_setting="sep", z0=popu_structure(popu),
                       truedata_style="glm", alpha=ALPHA, beta=BETA, recruit_p=RECRUIT_P)
    def gpmc(xy, ks, lik):
        dd = xy(popu); m = make_gpmc((dd[0], dd[1]), lik, ks)
        return m, gpflow.optimizers.SamplingHelper(m.log_posterior_density, m.trainable_parameters)
    def gpr(xy):
        dd = xy(popu); m = make_gpr((dd[0], dd[1]))
        return m, gpflow.optimizers.SamplingHelper(m.log_posterior_density, m.trainable_parameters)
    wy.m_fec_new, wy.hmc_helper_fec = gpmc(XY_fec_compu, 100., gpflow.likelihoods.Bernoulli())
    wy.m_flow_poi_new,wy.hmc_helper_flow = gpmc(XY_flow_compu, 100., gpflow.likelihoods.Poisson())
    wy.m_sur_new, wy.hmc_helper_sur = gpmc(XY_sur_compu, 50., gpflow.likelihoods.Bernoulli())
    wy.m_grw_f_new, wy.hmc_helper_grw_f = gpr(XY_grw_f_compu)
    wy.m_grw_nf_new, wy.hmc_helper_grw_nf = gpr(XY_grw_nf_compu)
    wy.m_grw_new, wy.hmc_helper_grw = gpr(XY_grw_compu)
    L = lambda p: pickle.load(open(f"{TP}/MCMC/gp/{p}", "rb"))
    wy.samples_sur=L("sep/samples_sur.pkl"); wy.samples_fec=L("sep/samples_fec.pkl")
    wy.samples_flow=L("sep/samples_flow.pkl"); wy.samples_grw_f=L("sep/samples_grw_f.pkl")
    wy.samples_grw_nf=L("sep/samples_grw_nf.pkl"); wy.samples_grw=L("nonsep/samples_grw.pkl")
    wy.rep = 1
    return wy


def cache_gp_posteriors(wy, N_DRAWS, ALPHA, BETA, RECRUIT_P):
    """Precompute and store the GP posterior cache for every MCMC draw, then check it."""
    from s0_fun_IBMs import GPMC_posterior, IBM_1step_gp, IBM_1step_gp_cache
    LM = "true_popu/mcmc/Lm/glm"
    with contextlib.redirect_stdout(io.StringIO()): # silence the internal "cache only" message
        for i in range(N_DRAWS):
            wy.GPmodels_givenindex_cache([i, i, i, i, i, i])
            pickle.dump(GPMC_posterior(wy.m_sur_new).cache, open(f"{LM}/sur/{i}", "wb"))
            pickle.dump(wy.m_grw_f_new.posterior().cache, open(f"{LM}/grw_f/{i}", "wb"))
            pickle.dump(wy.m_grw_nf_new.posterior().cache, open(f"{LM}/grw_nf/{i}", "wb"))
            pickle.dump(GPMC_posterior(wy.m_flow_poi_new).cache, open(f"{LM}/flow/{i}", "wb"))
            pickle.dump(GPMC_posterior(wy.m_fec_new).cache, open(f"{LM}/fec/{i}", "wb"))
            pickle.dump(wy.m_grw_new.posterior().cache, open(f"{LM}/grw/{i}", "wb"))

        # correctness: the cached IBM step must reproduce the uncached one (same seed)
        wy.GPmodels_givenindex_cache([0, 0, 0, 0, 0, 0])
        for k, r in [("m_sur","sur"),("m_grw_f","grw_f"),("m_grw_nf","grw_nf"),("m_fec","fec"),("m_flow_poi","flow")]:
            getattr(wy, k + "_new").cache = pickle.load(open(f"{LM}/{r}/0", "rb"))
        models0 = {"m_sur":wy.m_sur_new,"m_grw_f":wy.m_grw_f_new,"m_grw_nf":wy.m_grw_nf_new,
                   "m_fec":wy.m_fec_new,"m_flow_poi":wy.m_flow_poi_new,
                   "alpha":ALPHA,"beta":BETA,"recruit_p":RECRUIT_P}
        np.random.seed(4567); a = IBM_1step_gp(zt=wy.z0[0], age=wy.z0[1], models=models0)
        np.random.seed(4567); b = IBM_1step_gp_cache(zt=wy.z0[0], age=wy.z0[1], models=models0)
        np.testing.assert_allclose(a, b)
    print(f"cached {N_DRAWS} draws x 6 GP models (including the pooled-growth).")


# --Step 5--------------------------------------------------------------

def prepare_toy_abc(wy, ABC_BATCH, N_DRAWS):
    """Run the ABC sampler in series and perturb within the toy's cache range (no source file is changed)."""
    from s0_fun_ss import list_comparisons_interested  # (used inside IPM_whole_givenindex)

    def _serial(self, total_samples, given_theta=False, given_weight=False, random=True):
        out = []
        for _ in range(int(min(total_samples, self.toy_batch))):
            if given_theta and given_weight and random: out.append(self.random_IPM_ABC_weight())
            elif given_theta and given_weight and not random: out.append(self.not_random_IPM_ABC_weight())
            else: out.append(self.random_IPM_whole())
        return out

    def _perturb(self): # perturb within the toy's cache range
        idx = self.p_index[np.random.choice(self.p_index.shape[0], p=self.weight)].copy()
        idx[np.random.randint(0, 5)] = np.random.randint(0, self.n_draws)
        return self.IPM_whole_givenindex(idx)

    wy.toy_batch = ABC_BATCH; wy.n_draws = N_DRAWS
    wy.random_IPM_whole_para = types.MethodType(_serial, wy)
    wy.random_IPM_ABC_weight = types.MethodType(_perturb, wy)


def load_abc_round(c):
    """Read what ABC_SMC saved for round c (details=True)."""
    path = lambda f: f"ABC_details/glmsep/{f}{c}.pkl"
    out = {k: np.load(path(k), allow_pickle=True) for k in ["particles_summary", "threshold", "mad", "weight"]}
    out["p_index"] = np.load(f"ABC_details/glmsep/p_index_{c}.pkl", allow_pickle=True)
    return out


# --plotting ------------------------------------------------------------

def gp_curves(model, draws, grid):
    """Predicted mean of a fitted GP on `grid`, one row per MCMC draw."""
    import gpflow
    helper = gpflow.optimizers.SamplingHelper(model.log_posterior_density,
                                              model.trainable_parameters)
    curves = []
    for i in range(int(draws[0].shape[0])):
        for var, var_draws in zip(helper.current_state, draws):
            var.assign(var_draws[i])
        mean, _ = model.predict_y(grid[:, None])
        curves.append(mean.numpy().ravel())
    return np.array(curves)


def plot_data(toy, alpha, beta, fits=None):
    """Plot datasets each vital rate is fitted to, and the recruit sizes.
    """
    import matplotlib.pyplot as plt
    small_fonts = {"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                   "xtick.labelsize": 7, "ytick.labelsize": 7}
    with plt.rc_context(small_fonts):   # the figure is 75% of its original size, so the fonts are smaller too
        _plot_data(toy, alpha, beta, fits)


def _plot_data(toy, alpha, beta, fits=None):
    import matplotlib.pyplot as plt
    from scipy import stats
    from s0_fun_base import (XY_sur_compu, XY_fec_compu, XY_flow_compu,
                             XY_grw_f_compu, XY_grw_nf_compu)

    rng = np.random.default_rng(0)
    jitter = lambda n: rng.uniform(-0.04, 0.04, n) 
    fig, ax = plt.subplots(2, 3, figsize=(8.25, 4.9))
    ax = ax.ravel()
    size_max = np.nanmax(toy["size"]) + 0.2
    lim = (0, np.nanmax(toy[["size", "sizeNext"]].to_numpy()) + 0.3)

    # (vital rate, panel, data, title, shift added to the response)
    panels = [("sur", ax[0], XY_sur_compu, "(A) Survival", 0),
              ("fec", ax[1], XY_fec_compu, "(B) Flowering", 0),
              ("flow", ax[2], XY_flow_compu, "(C) Flowering stalks", 1),   # XY_flow_compu returns stalks - 1
              ("grw_f", ax[3], XY_grw_f_compu, "(D) Growth, flowered", 0),
              ("grw_nf", ax[4], XY_grw_nf_compu, "(E) Growth, did not flower", 0)]

    for name, a, xy, title, shift in panels:
        X, Y = xy(toy)
        x, y = X.to_numpy().ravel(), Y.to_numpy().ravel() + shift
        if fits is not None and name in fits:
            grid = np.linspace(0, size_max, 100)
            curves = gp_curves(*fits[name], grid) + shift
            for k, c in enumerate(curves):
                a.plot(grid, c, color="C1", lw=0.9, alpha=0.7,
                       label="one MCMC draw" if k == 0 else None)
        if name in ("sur", "fec"):
            y = y + jitter(len(y))
        a.scatter(x, y, s=8, alpha=0.6, color="C0", label="data", zorder=3)
        a.set_title(f"{title} ({len(x)} rows)", loc="left")
        a.set_xlabel("size")

    for a, labels in [(ax[0], ["died", "survived"]), (ax[1], ["did not flower", "flowered"])]:
        a.set_yticks([0, 1]); a.set_yticklabels(labels); a.set_ylim(-0.3, 1.3)
    for a in ax[:3]:
        a.set_xlim(0, size_max) # same size axis for A to C
    ax[2].set_ylabel("number of stalks")
    for a in ax[3:5]:
        a.plot(lim, lim, "k--", lw=0.8)
        a.set_xlim(lim); a.set_ylim(lim); a.set_ylabel("size next year")
    if fits is not None:
        ax[0].legend(loc="center right", fontsize=6)

    recruits = toy.loc[toy["size"].isna(), "sizeNext"].to_numpy()
    grid = np.linspace(0.01, recruits.max() + 0.5, 200)
    ax[5].hist(recruits, bins=10, density=True, alpha=0.6)
    ax[5].plot(grid, stats.gamma(a=alpha, scale=1 / beta).pdf(grid), "k-", lw=1)
    ax[5].set_xlabel("recruit size"); ax[5].set_ylabel("density")
    ax[5].set_title(f"(F) Recruit size ({len(recruits)} recruits)", loc="left")

    fig.tight_layout()
    plt.show()


def plot_nlpp(nlpp, ref, near):
    """NLPP of each MCMC draw, sorted, with theta_opt and the perturbed models marked."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    order = np.argsort(nlpp)
    colors = ["C3" if i == ref else ("C1" if i in near else "0.7") for i in order]
    ax.bar(range(len(nlpp)), nlpp[order], color=colors)
    ax.set_ylim(nlpp.min() - 1, nlpp.max() + 1)
    ax.set_xlabel("MCMC draws, sorted by NLPP"); ax.set_ylabel("NLPP")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in ["C3", "C1", "0.7"]],
              labels=["$\\theta_{opt}$", "perturbed models $\\theta_k$", "other draws"], fontsize=8)
    fig.tight_layout(); plt.show()


def plot_abc_rounds(n_rounds):
    """Discrepancies: round 0 candidates versus accepted particles."""
    import matplotlib.pyplot as plt
    r = [load_abc_round(c) for c in range(n_rounds)]
    dist = lambda s, mad: np.sqrt(np.sum((s / mad) ** 2, axis=1))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    # each round is filtered with the threshold and scaling set in the round before
    bins = np.linspace(0, np.quantile(dist(r[0]["particles_summary"], r[0]["mad"]), 0.98), 25)
    ax.hist(dist(r[0]["particles_summary"], r[0]["mad"]), bins=bins, alpha=0.5,
            color="0.5", label="round 0: all candidates")
    for c in range(1, n_rounds):
        ax.hist(dist(r[c]["particles_summary"], r[c - 1]["mad"]), bins=bins, alpha=0.6,
                color=f"C{c + 2}", label=f"round {c}: accepted")
        ax.axvline(r[c - 1]["threshold"], color=f"C{c + 2}", ls="--", lw=1.2,
                   label=f"tolerance used in round {c}")
    ax.set_xlabel("distance to the observed data"); ax.set_ylabel("number of particles")
    ax.legend(fontsize=7)
    fig.tight_layout()
    plt.show()


def plot_lambdas(lam):
    """Histogram of lambda for particles drawn according to their ABC weights."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(lam, bins=15, alpha=0.7)
    ax.set_xlabel("population growth rate $\\lambda$"); ax.set_ylabel("number of draws")
    fig.tight_layout()
    plt.show()
