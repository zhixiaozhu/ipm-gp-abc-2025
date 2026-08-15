from GP_IPM_class import *
from s1_simu_setup import *
from s0_fun_IBM import kernel_setting 

if __name__ == '__main__':
    # We'll refit the m_grw_f model only, and keep the other vital rate models as the MLE one.

    # 1 #####################################################################################################
    # Set-up
    target = 'm_grw_f'
    # Do you want to generate posterior samples?
    simu_mcmc_mode = 'OFF' # 'ON'
    # Do you want to check the MCMC plots?
    plot_mode = 'OFF' # 'ON'
    # Do you want to simulated populations based on all MCMC samples?
    popu_all_mode = 'ON' # 'ON'
    # Do you want to simulated populations based on the MCMC samples around optimum?
    popu_opt_mode = 'OFF' # 'ON'


    # initializing models
    IPM_true = GP_IPM(popu_data=popu_dataset, GPmodel_true=models_true)
    IPM_pret_grw_f = Perted_IPM(popu_data=popu_dataset, GPmodel_true=models_true, target=target, NB=False)
    lower_p = 0
    upper_p = 5
    m_size_p = 100
    max_age_p = 13
    mesh_setting = kernel_setting(m_size_p, lower_p, upper_p, max_age_p)
    #########################################################################################################
    

    # 2 ######################################################################################################
    # cCalculating the posterior through HMC.
    # Firstly, we build the GPR model.
    print('\n\n\n' + 'Processing: HMC' + '\n\n\n')
    kernel = gpflow.kernels.RBF(lengthscales=np.array([1,1]))
    m_grw_f_new = gpflow.models.GPR(data=(IPM_pret_grw_f.X_grw_f, IPM_pret_grw_f.Y_grw_f), 
                                    kernel=kernel, mean_function=None)

    # Secondly, initialize the model to the maximum likelihood solution.
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(m_grw_f_new.training_loss, m_grw_f_new.trainable_variables)
    # Thirdly, we add priors to the hyperparameters.
    m_grw_f_new.kernel.lengthscales.prior = tfd.InverseGamma(f64(0.001),f64(0.001))
    m_grw_f_new.kernel.variance.prior = tfd.InverseGamma(f64(0.001),f64(0.001))
    m_grw_f_new.likelihood.variance.prior = tfd.InverseGamma(f64(0.001),f64(0.001))
    #########################################################################################################


    # 3 #####################################################################################################
    # We now sample from the posterior using HMC.
    hmc_helper = gpflow.optimizers.SamplingHelper(
        m_grw_f_new.log_posterior_density, m_grw_f_new.trainable_parameters
    )
    num_burnin_steps = 30000
    num_samples = 20000

    if simu_mcmc_mode == 'ON':

        tf.random.set_seed(12345)
        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
        )
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
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
        parameter_samples = hmc_helper.convert_to_constrained_values(samples)
        pickle.dump(samples, open(file = os.getcwd()+"/mcmc_samples/m_grw_f" +"/samples.pkl", mode="wb"))
        pickle.dump(parameter_samples, open(file = os.getcwd()+"/mcmc_samples/m_grw_f" +"/parameter_samples.pkl", mode="wb"))
        
        #  assign samples to our models
        IPM_pret_grw_f.mcmc_para_sample = parameter_samples
    
    elif simu_mcmc_mode == 'OFF':
        print('\n\n\n' + 'Loading MCMC samples' + '\n\n\n')
        samples = pickle.load(open(file = os.getcwd()+"/mcmc_samples/m_grw_f" +"/samples.pkl", mode="rb"))
        parameter_samples = pickle.load(open(file = os.getcwd()+"/mcmc_samples/m_grw_f" +"/parameter_samples.pkl", mode="rb"))
    
         #  assign samples to our models
        IPM_pret_grw_f.mcmc_para_sample = parameter_samples

    else:
        print("Error. The simu_mode is unkown.")
        exit()
    #########################################################################################################



    # 4 #####################################################################################################
    # ploting
    param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(m_grw_f_new).items()}
    if plot_mode == 'ON':
        print('\n\n\n' + 'Plotting' + '\n\n\n')
        # However, we often wish to sample the constrained parameter values, 
        # not the unconstrained one. The SamplingHelper helps us convert our unconstrained values 
        # to constrained parameter ones.
        plot_samples(samples, m_grw_f_new.trainable_parameters, "unconstrained values", param_to_name)
        plot_samples(parameter_samples, m_grw_f_new.trainable_parameters, "constrained parameter values", param_to_name)

        # inspect the marginal distribution of samples.
        marginal_samples(samples, m_grw_f_new.trainable_parameters, "unconstrained variable samples", param_to_name)
        marginal_samples(parameter_samples, m_grw_f_new.trainable_parameters, "constrained parameter samples", param_to_name)

    elif plot_mode == 'OFF':
        print('\n\n\n' + 'Plots skipped' + '\n\n\n')
    else:
        print("Error. The plot_mode is unkown.")
        exit()
    #########################################################################################################



    # 5 #####################################################################################################
    # Using MCMC samples to generate populations

    if popu_all_mode == 'ON':
        print('\n\n\n' + 'Re-calculating summary_data for all the MCMC samples' + '\n\n\n')
        summary_data = IPM_pret_grw_f.obsVSsimu_fun_parallel(rep=1, random_seed=1, 
                                                             MCMC_boolean_list='All')      

        pickle.dump(summary_data, open(file = os.getcwd() + "/mcmc_samples/m_grw_f" + "/summary_data.pkl", mode="wb"))
        IPM_pret_grw_f.nlog_post = np.array(summary_data['nlpo'])
        IPM_pret_grw_f.nlog_likeli = np.array(summary_data['nll'])

    elif popu_all_mode == 'OFF':
        print('\n\n\n' + 'Loading summary_data for all the MCMC samples' + '\n\n\n')
        summary_data = pickle.load(open(file = os.getcwd() + "/mcmc_samples/m_grw_f" + "/summary_data.pkl", mode="rb"))
        IPM_pret_grw_f.nlog_post = np.array(summary_data['nlpo'])
        IPM_pret_grw_f.nlog_likeli = np.array(summary_data['nll'])
    else:
        print("Error. The popu_all_mode is unkown.")
        exit()

    #########################################################################################################



    # 6 #####################################################################################################
    # Now, if we consider the estimates with the likelihood scores around the optimum.
    #      Calculating the summary tests around the optimum for full data.
    rep = 1000
    opt_percentage = 2

    if popu_opt_mode == 'ON':
        print('\n\n\n' + 'Re-calculating summary_data for the opt MCMC samples' + '\n\n\n')
        summary_opt = IPM_pret_grw_f.simuVSsimu_singleModel_fun_parallel(rep=rep, random_seed=1,
                                            interested_models='Opt', evaluate_at_training=False)

        pickle.dump(summary_opt, 
                    open(file = os.getcwd() + "/mcmc_samples/m_grw_f" + "/summary_opt.pkl", mode="wb"))
        
        summary_around_opt = IPM_pret_grw_f.simuVSsimu_fun_parallel(rep=rep, random_seed=1, 
                                       opt_percentage=opt_percentage, interested_models='Opt', 
                                       MCMC_boolean_list='Opt', evaluate_at_training=False)
        pickle.dump(summary_around_opt, 
                        open(file = os.getcwd() + "/mcmc_samples/m_grw_f/summary_around_opt.pkl", mode="wb"))

    elif popu_opt_mode == 'OFF':
        print('\n\n\n' + 'Loading summary_data for the opt MCMC samples' + '\n\n\n')
        summary_opt = pickle.load(open(file = os.getcwd() + "/mcmc_samples/m_grw_f" + "/summary_opt.pkl", mode="rb"))
        summary_around_opt = pickle.load(open(file = os.getcwd() + "/mcmc_samples/m_grw_f" + "/summary_around_opt.pkl", mode="rb"))

    else:
        print("Error. The popu_opt_mode is unkown.")
        exit()

