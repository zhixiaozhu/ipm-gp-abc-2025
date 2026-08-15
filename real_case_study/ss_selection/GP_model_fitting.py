from pyexpat import model
from s0_initial import *
from gpflow.likelihoods import ScalarLikelihood
from gpflow import Parameter
from gpflow.utilities import positive

################################################################################################################################################################
# binomial set up
class NegativeBinomial(ScalarLikelihood):
    """
    The negative-binomial likelihood with pmf:
    .. math::
        NB(y \mid \mu, \psi) =
            \frac{\Gamma(y + \psi)}{y! \Gamma(\psi)}
            \left( \frac{\mu}{\mu + \psi} \right)^y
            \left( \frac{\psi}{\mu + \psi} \right)^\psi
    where :math:`\mu = \exp(\nu)`. Its expected value is :math:`\mathbb{E}[y] = \mu `
    and variance :math:`Var[Y] = \mu + \frac{\mu^2}{\psi}`.
    """
    def __init__(self, psi=1.0, **kwargs):
        super().__init__(**kwargs)
        self.invlink = tf.exp
        self.psi = Parameter(
            psi,
            transform=positive(lower=0.01)
        )
    def _scalar_log_prob(self, F, Y):
        mu = self.invlink(F)
        mu_psi = mu + self.psi
        psi_y = self.psi + Y
        f1 = (
                tf.math.lgamma(psi_y) -
                tf.math.lgamma(Y + 1.0) -
                tf.math.lgamma(self.psi)
        )
        f2 = Y * tf.math.log(mu / mu_psi)
        f3 = self.psi * tf.math.log(self.psi / mu_psi)
        return f1 + f2 + f3
    def _conditional_mean(self, F):
        return self.invlink(F)
    def _conditional_variance(self, F):
        mu = self.invlink(F)
        return mu + tf.pow(mu, 2) / self.psi
    
################################################################################################################################################################
def baseline_model_fit(data_path, print_mode = 'OFF', NB=False):
    # data loading.
    if type(data_path) == pd.core.frame.DataFrame:
        data_ori = data_path.copy()
        data_ori = data_ori.sort_values(by=['size'])
        data_ori = data_ori.reset_index(drop=True)

    elif type(data_path) == str:
        data_ori = pd.read_csv(data_path)

        data_ori = data_ori.drop(data_ori.columns[0], axis="columns")
        data_ori = data_ori.drop(data_ori[data_ori['size'] == 0].index)
        data_ori = data_ori.sort_values(by=['size'])
        data_ori = data_ori.reset_index(drop=True)

        # classify individuals into several different age classes for latter analysis
        # set M= 13 to be consistent with data_89
        data_ori['age'][data_ori['age'] == 999] = 15
        data_ori['age'][data_ori['age'] >= 14] = 13
    else:

        assert False, '\n data_path should be a string or ps.DataFrame !'


    ############################################################################################################################################################# 
    # datasets constructing

    # 1 - surv
    index_sur = np.logical_not(np.isnan(data_ori["surv"]))
    size_sur = data_ori["size"][index_sur].to_numpy()
    surv_sur = data_ori["surv"][index_sur].to_numpy()
    age_sur = data_ori["age"][index_sur].to_numpy()

    # 2 - growth
    # i - if we conserding "fec" effect
    # data for breeders (fec == 1)
    index_grw = np.logical_not(np.isnan(data_ori["sizeNext"])) & np.logical_not(np.isnan(data_ori["size"]))
    fec_grw = data_ori["fec"][index_grw]

    size_grw_f = data_ori["size"][index_grw][fec_grw == 1].to_numpy()
    sizeNext_grw_f = data_ori["sizeNext"][index_grw][fec_grw == 1].to_numpy()
    age_grw_f = data_ori["age"][index_grw][fec_grw == 1].to_numpy()

    # data for breeders (fec == 0)
    size_grw_nf = data_ori["size"][index_grw][fec_grw == 0].to_numpy()
    sizeNext_grw_nf = data_ori["sizeNext"][index_grw][fec_grw == 0].to_numpy()
    age_grw_nf = data_ori["age"][index_grw][fec_grw == 0].to_numpy()

    # 3 - fec
    index_fec = np.logical_not(np.isnan(data_ori["fec"]))
    size_fec = data_ori["size"][index_fec].to_numpy()
    fec_fec = data_ori["fec"][index_fec].to_numpy()
    age_fec = data_ori["age"][index_fec].to_numpy()
    random_index = np.random.choice(500, 50)

    # 4 - number of flowering stalks
    index_flow = data_ori["fec"] == 1
    size_flow = data_ori["size"][index_flow].to_numpy()
    flow_flow = data_ori["flow"][index_flow].to_numpy()
    age_flow = data_ori["age"][index_flow].to_numpy()

    # 5 - recruit size (indp with the parent size)
    index_recr = np.isnan(data_ori["size"])


    if type(data_path) == pd.core.frame.DataFrame:
        # log have already taken for these simulated datasets

        # 1 - surv
        X_sur = np.concatenate((size_sur[:, None], age_sur[:, None]), axis=1)
        Y_sur = surv_sur[:, None]

        # 2 - growth
        X_grw_f = np.concatenate((size_grw_f[:, None], age_grw_f[:, None]), axis=1)
        Y_grw_f = sizeNext_grw_f[:, None]
        X_grw_nf = np.concatenate((size_grw_nf[:, None], age_grw_nf[:, None]), axis=1)
        Y_grw_nf = sizeNext_grw_nf[:, None]

        # 3 - fec
        X_fec = np.concatenate((size_fec[:, None], age_fec[:, None]), axis=1)
        Y_fec = fec_fec[:, None]

        # 4 - number of flowering stalks
        X_flow = np.concatenate((size_flow[:, None], age_flow[:, None]), axis=1)
        Y_flow = flow_flow[:, None] - 1

        sizeNext_recr = data_ori["sizeNext"][index_recr].to_numpy()


    else: #type(data_path) == str:
        X_sur = np.concatenate((np.log(size_sur)[:, None], age_sur[:, None]), axis=1)
        Y_sur = surv_sur[:, None]

        X_grw_f = np.concatenate((np.log(size_grw_f)[:, None], age_grw_f[:, None]), axis=1)
        Y_grw_f = np.log(sizeNext_grw_f)[:, None]
        X_grw_nf = np.concatenate((np.log(size_grw_nf)[:, None], age_grw_nf[:, None]), axis=1)
        Y_grw_nf = np.log(sizeNext_grw_nf)[:, None]

        X_fec = np.concatenate((np.log(size_fec)[:, None], age_fec[:, None]), axis=1)
        Y_fec = fec_fec[:, None]
        
        X_flow = np.concatenate((np.log(size_flow)[:, None], age_flow[:, None]), axis=1)
        Y_flow = flow_flow[:, None] - 1

        sizeNext_recr = np.log(data_ori["sizeNext"][index_recr].to_numpy())


    
    ############################################################################################################################################################# 
    # model fitting

    # 1 - survive 
    l_sur = gpflow.likelihoods.Bernoulli()
    k_sur = gpflow.kernels.RBF(lengthscales=np.array([1.4,0.7]))

    m_sur1 = gpflow.models.VGP((X_sur, Y_sur), kernel=k_sur, likelihood=l_sur)

    opt_sur = gpflow.optimizers.Scipy()
    opt_sur.minimize(m_sur1.training_loss, variables=m_sur1.trainable_variables)

    # 2 - growth: if we conserding "fec" effect
    # growth model fitting for breeders (fec == 1)

    k_grw_f = gpflow.kernels.RBF(lengthscales=np.array([1,1000]))

    m_grw_f = gpflow.models.GPR(data=(X_grw_f, Y_grw_f), 
                                kernel=k_grw_f, mean_function=None)

    opt_grw_f = gpflow.optimizers.Scipy()
    opt_grw_f.minimize(m_grw_f.training_loss, variables=m_grw_f.trainable_variables)


    # growth model fitting for the remainings (fec == 0)
    k_grw_nf = gpflow.kernels.RBF(lengthscales=np.array([1,1]))

    m_grw_nf = gpflow.models.GPR(data=(X_grw_nf, Y_grw_nf), 
                                kernel=k_grw_nf, mean_function=None)

    opt_grw_nf = gpflow.optimizers.Scipy()
    opt_grw_nf.minimize(m_grw_nf.training_loss, variables=m_grw_nf.trainable_variables)


    # 3 - fec
    l_fec = gpflow.likelihoods.Bernoulli()
    k_fec = gpflow.kernels.RBF(lengthscales=np.array([1,990]), variance=6)
    m_fec = gpflow.models.VGP((X_fec, Y_fec), kernel=k_fec, likelihood=l_fec)

    opt_fec = gpflow.optimizers.Scipy()
    opt_fec.minimize(m_fec.training_loss, variables=m_fec.trainable_variables)



    # 4 - number of flowering stalks 
    # (i) Poisson
    l_flow_poi = gpflow.likelihoods.Poisson()
    k_flow_poi = gpflow.kernels.RBF(lengthscales=np.array([1,1]))

    m_flow_poi = gpflow.models.VGP((X_flow, Y_flow), 
                                kernel=k_flow_poi, likelihood=l_flow_poi)
    opt_flow_poi = gpflow.optimizers.Scipy()
    result = opt_flow_poi.minimize(m_flow_poi.training_loss, variables=m_flow_poi.trainable_variables, 
                        method="L-BFGS-B")


    # 4 - number of flowering stalks 
    #  (ii) NB

    if NB == True:

        l_flow_nb = NegativeBinomial()
        k_flow_nb = gpflow.kernels.RBF(lengthscales=np.array([1,1]))

        m_flow_nb = gpflow.models.VGP((X_flow, Y_flow), 
                                    kernel=k_flow_nb, likelihood=l_flow_nb)
        opt_flow_nb = gpflow.optimizers.Scipy()
        result = opt_flow_nb.minimize(m_flow_nb.training_loss, variables=m_flow_nb.trainable_variables, 
                            method="L-BFGS-B")
    else:
        m_flow_nb = None


    # 6 - recruits' establishment probability
    # there is no information about the number of seed in the data set.
    # we may use the #offspring/#flowering stalks to estimate it.
    # mean_log_recr = np.mean(np.log(sizeNext_recr))
    # var_log_recr = np.var(np.log(sizeNext_recr))
    mean_log_recr = np.mean(sizeNext_recr)
    var_log_recr = np.var(sizeNext_recr)

    v = var_log_recr * sizeNext_recr.shape[0] / (sizeNext_recr.shape[0] - 1)
    # for gamma distribution
    alpha = np.power(mean_log_recr, 2) / v
    beta = mean_log_recr / v

    recruit_estab = sum(np.isnan(data_ori["size"])) / data_ori["flow"].sum()
    m_recruits = pd.DataFrame([alpha,beta,recruit_estab], 
                            index=['gamma_alpha','gamma_beta','recruit_estab'],
                            columns=['values'])


    if print_mode == 'ON':
        print(m_sur1.training_loss())
        gpflow.utilities.print_summary(m_sur1, fmt="notebook")
        print(m_grw_f.training_loss())
        gpflow.utilities.print_summary(m_grw_f, fmt="notebook")
        print(m_grw_nf.training_loss())
        gpflow.utilities.print_summary(m_grw_nf, fmt="notebook")
        print(m_fec.training_loss())
        gpflow.utilities.print_summary(m_fec, fmt="notebook")
        print(m_flow_poi.training_loss())
        gpflow.utilities.print_summary(m_flow_poi, fmt="notebook")
        print(m_flow_nb.training_loss())
        gpflow.utilities.print_summary(m_flow_nb, fmt="notebook")


    models_true = {
        "m_sur": m_sur1,
        "m_grw_f": m_grw_f,
        "m_grw_nf": m_grw_nf,
        "m_fec": m_fec,
        "m_flow_poi": m_flow_poi,
        "m_flow_nb": m_flow_nb,
        "alpha": m_recruits.iloc[0,0],
        "beta": m_recruits.iloc[1,0],
        "recruit_p": m_recruits.iloc[2,0]
    }

    return models_true




def baseline_model_fit_whole(data_whole, NB=False):
    data_ori = data_whole.copy()

    ############################################################################################################################################################# 
    # datasets constructing

    # 1 - surv
    index_sur = np.logical_not(np.isnan(data_ori["surv"]))
    size_sur = data_ori["size"][index_sur].to_numpy()
    surv_sur = data_ori["surv"][index_sur].to_numpy()
    age_sur = data_ori["age"][index_sur].to_numpy()
    year_sur = data_ori["year"][index_sur].to_numpy() 

    # 2 - growth
    # i - if we conserding "fec" effect
    # data for breeders (fec == 1)
    index_grw = np.logical_not(np.isnan(data_ori["sizeNext"])) & np.logical_not(np.isnan(data_ori["size"]))
    fec_grw = data_ori["fec"][index_grw]

    size_grw_f = data_ori["size"][index_grw][fec_grw == 1].to_numpy()
    sizeNext_grw_f = data_ori["sizeNext"][index_grw][fec_grw == 1].to_numpy()
    age_grw_f = data_ori["age"][index_grw][fec_grw == 1].to_numpy()
    year_grw_f = data_ori["year"][index_grw][fec_grw == 1].to_numpy()

    # data for breeders (fec == 0)
    size_grw_nf = data_ori["size"][index_grw][fec_grw == 0].to_numpy()
    sizeNext_grw_nf = data_ori["sizeNext"][index_grw][fec_grw == 0].to_numpy()
    age_grw_nf = data_ori["age"][index_grw][fec_grw == 0].to_numpy()
    year_grw_nf = data_ori["year"][index_grw][fec_grw == 0].to_numpy()

    # 3 - fec
    index_fec = np.logical_not(np.isnan(data_ori["fec"]))
    size_fec = data_ori["size"][index_fec].to_numpy()
    fec_fec = data_ori["fec"][index_fec].to_numpy()
    age_fec = data_ori["age"][index_fec].to_numpy()
    year_fec = data_ori["year"][index_fec].to_numpy() 
    random_index = np.random.choice(500, 50)

    # 4 - number of flowering stalks
    index_flow = data_ori["fec"] == 1
    size_flow = data_ori["size"][index_flow].to_numpy()
    flow_flow = data_ori["flow"][index_flow].to_numpy()
    age_flow = data_ori["age"][index_flow].to_numpy()
    year_flow = data_ori["year"][index_flow].to_numpy() 

    # 5 - recruit size (indp with the parent size)
    index_recr = np.isnan(data_ori["size"])
    sizeNext_recr = data_ori["sizeNext"][index_recr].to_numpy()

    # log have already taken for these simulated datasets
    # 1 - surv
    X_sur = np.concatenate((size_sur[:, None], age_sur[:, None], year_sur[:, None]), axis=1)
    Y_sur = surv_sur[:, None]

    # 2 - growth
    X_grw_f = np.concatenate((size_grw_f[:, None], age_grw_f[:, None], year_grw_f[:, None]), axis=1)
    Y_grw_f = sizeNext_grw_f[:, None]
    X_grw_nf = np.concatenate((size_grw_nf[:, None], age_grw_nf[:, None], year_grw_nf[:, None]), axis=1)
    Y_grw_nf = sizeNext_grw_nf[:, None]

    # 3 - fec
    X_fec = np.concatenate((size_fec[:, None], age_fec[:, None], year_fec[:, None]), axis=1)
    Y_fec = fec_fec[:, None]

    # 4 - number of flowering stalks
    X_flow = np.concatenate((size_flow[:, None], age_flow[:, None], year_flow[:, None]), axis=1)
    Y_flow = flow_flow[:, None] - 1
    
    ############################################################################################################################################################# 
    # model fitting

    # 1 - survive 
    l_sur = gpflow.likelihoods.Bernoulli()
    k_sur = gpflow.kernels.RBF(lengthscales=np.array([1,1,1]))

    m_sur1 = gpflow.models.VGP((X_sur, Y_sur), kernel=k_sur, likelihood=l_sur)

    opt_sur = gpflow.optimizers.Scipy()
    opt_sur.minimize(m_sur1.training_loss, variables=m_sur1.trainable_variables)

    # 2 - growth: if we conserding "fec" effect
    # growth model fitting for breeders (fec == 1)

    k_grw_f = gpflow.kernels.RBF(lengthscales=np.array([1,1, 1]))

    m_grw_f = gpflow.models.GPR(data=(X_grw_f, Y_grw_f), 
                                kernel=k_grw_f, mean_function=None)

    opt_grw_f = gpflow.optimizers.Scipy()
    opt_grw_f.minimize(m_grw_f.training_loss, variables=m_grw_f.trainable_variables)


    # growth model fitting for the remainings (fec == 0)
    k_grw_nf = gpflow.kernels.RBF(lengthscales=np.array([1,1, 1]))

    m_grw_nf = gpflow.models.GPR(data=(X_grw_nf, Y_grw_nf), 
                                kernel=k_grw_nf, mean_function=None)

    opt_grw_nf = gpflow.optimizers.Scipy()
    opt_grw_nf.minimize(m_grw_nf.training_loss, variables=m_grw_nf.trainable_variables)


    # 3 - fec
    l_fec = gpflow.likelihoods.Bernoulli()
    k_fec = gpflow.kernels.RBF(lengthscales=np.array([1,990, 1]), variance=6)
    m_fec = gpflow.models.VGP((X_fec, Y_fec), kernel=k_fec, likelihood=l_fec)

    opt_fec = gpflow.optimizers.Scipy()
    opt_fec.minimize(m_fec.training_loss, variables=m_fec.trainable_variables)



    # 4 - number of flowering stalks 
    # (i) Poisson
    l_flow_poi = gpflow.likelihoods.Poisson()
    k_flow_poi = gpflow.kernels.RBF(lengthscales=np.array([1,1, 1]))

    m_flow_poi = gpflow.models.VGP((X_flow, Y_flow), 
                                kernel=k_flow_poi, likelihood=l_flow_poi)
    opt_flow_poi = gpflow.optimizers.Scipy()
    result = opt_flow_poi.minimize(m_flow_poi.training_loss, variables=m_flow_poi.trainable_variables, 
                        method="L-BFGS-B")


    # 4 - number of flowering stalks 
    #  (ii) NB

    if NB == True:

        l_flow_nb = NegativeBinomial()
        k_flow_nb = gpflow.kernels.RBF(lengthscales=np.array([1,1, 1]))

        m_flow_nb = gpflow.models.VGP((X_flow, Y_flow), 
                                    kernel=k_flow_nb, likelihood=l_flow_nb)
        opt_flow_nb = gpflow.optimizers.Scipy()
        result = opt_flow_nb.minimize(m_flow_nb.training_loss, variables=m_flow_nb.trainable_variables, 
                            method="L-BFGS-B")
    else:
        m_flow_nb = None


    # 6 - recruits' establishment probability
    # there is no information about the number of seed in the data set.
    # we may use the #offspring/#flowering stalks to estimate it.
    # mean_log_recr = np.mean(np.log(sizeNext_recr))
    # var_log_recr = np.var(np.log(sizeNext_recr))
    mean_log_recr = np.mean(sizeNext_recr)
    var_log_recr = np.var(sizeNext_recr)
    v = var_log_recr * sizeNext_recr.shape[0] / (sizeNext_recr.shape[0] - 1)
    # for gamma distribution
    alpha = np.power(mean_log_recr, 2) / v
    beta = mean_log_recr / v
    recruit_estab = sum(np.isnan(data_ori["size"])) / data_ori["flow"].sum()
    m_recruits = pd.DataFrame([alpha,beta,recruit_estab], 
                            index=['gamma_alpha','gamma_beta','recruit_estab'],
                            columns=['values'])

    models = {
        "m_sur": m_sur1,
        "m_grw_f": m_grw_f,
        "m_grw_nf": m_grw_nf,
        "m_fec": m_fec,
        "m_flow_poi": m_flow_poi,
        "m_flow_nb": m_flow_nb,
        "alpha": m_recruits.iloc[0,0],
        "beta": m_recruits.iloc[1,0],
        "recruit_p": m_recruits.iloc[2,0]
    }

    return models