from r0_function import *

reproc_data_mode = 'OFF'
re_eastimate_mle = 'OFF'
re_mcmc = 'OFF'

# Time = 5 mean the first 5 datasets will be used for model training
Time = 5

if reproc_data_mode == 'ON':
    # we treat popu_dataset as the dataset at T=-1
    print("\n\n Re-processing 'raw' population datasets... \n\n")

    for i in range(9):
        if i == 0:
            data = pd.read_csv("control_34.csv")
        elif i == 1:
            data = pd.read_csv("control_45.csv")
        elif i == 2:
            data = pd.read_csv("control_56.csv")
        elif i == 3:
            data = pd.read_csv("control_67.csv")
        elif i == 4:
            data = pd.read_csv("control_78.csv")
        elif i == 5:
            data = pd.read_csv("control_89.csv")
        elif i == 6:
            data = pd.read_csv("control_910.csv")
        elif i == 7:
            data = pd.read_csv("control_1011.csv")
        elif i == 8:
            data = pd.read_csv("control_1112.csv")
        
        data = data.drop(data.columns[0], axis="columns")
        data = data.sort_values(by=['size']).reset_index(drop=True)

        # some inidividuals' size are zero, whcih mighe be sampling errors.
        data = data.drop(np.where(data['size'] == 0)[0], axis=0).reset_index(drop=True)
        data = data.drop(np.where(data['sizeNext'] == 0)[0], axis=0).reset_index(drop=True)
        data['size'] = np.log(data['size'])
        data['sizeNext'] = np.log(data['sizeNext'])
        # classify individuals into several different age classes for latter analysis
        # the largest value of age in dataset 2003-2004 is 7, so setting 8 as an obsorbing age.
        data['age'][data['age'] > 7] = 8
        data['ageNext'][data['ageNext'] > 7] = 8
        pickle.dump(data, open(file = os.getcwd()+f"/true_popu/popu_dataset{i}.pkl", mode="wb"))

else:
    print("\n\n Loading processed population datasets... \n\n")
    for i in range(Time):
        globals()[f'popu_dataset{i}'] = pickle.load(open(file = os.getcwd()+f"/true_popu/popu_dataset{i}.pkl", mode="rb"))
    w = pd.read_csv(os.getcwd()+'/true_popu/weather_summary_2.txt', sep='\t')



popu_0toT = {i: globals()[f'popu_dataset{i}'].copy() for i in range(Time)}
p = {i: globals()[f'popu_dataset{i}'].copy() for i in range(Time)}
for i in range(Time):
    p[i]['tmax'] = w.iloc[0, i]
    p[i]['tmin'] = w.iloc[1, i] 
    p[i]['precip'] = w.iloc[2, i]  

popu_whole = pd.concat([p[i] for i in range(Time)]).sort_values(by=['size']).reset_index(drop=True); del(p)

if re_eastimate_mle == 'ON':
    print("\n\n Re-calculating MLEs for the processed population datasets in two ways... \n\n")
    # Way 1: fitting the model year by year
    print('Way 1')
    theta_mle_0toT = pd.DataFrame(0, range(Time), para_names)

    for i in range(Time):
        print(i)
        m = baseline_model_fit(data_path=popu_0toT[i], print_mode = 'OFF')
        theta_mle_0toT.iloc[i,:] = extract_parameters(m)

    pickle.dump(theta_mle_0toT, open(file = os.getcwd()+"/true_popu/mle/theta_mle_0toT.pkl", mode="wb"))

    # Way 2: for each vital rate, fitting a single model by using the whole datasets.
    print('Way 2')
    theta_mle_whole = pd.DataFrame(0, range(1), para_names_whole)
    m = baseline_model_fit_whole(data_whole=popu_whole)
    theta_mle_whole.iloc[0,:] = extract_parameters_whole(m); del(m)

    pickle.dump(theta_mle_whole, open(file = os.getcwd()+"/true_popu/mle/theta_mle_whole.pkl", mode="wb"))
 
else:
    print("\n\n Loading MLEs for the processed population datasets... \n\n")
    theta_mle_whole = pickle.load(open(file = os.getcwd()+f"/true_popu/mle/theta_mle_whole.pkl", mode="rb")) 



if re_mcmc == 'ON':
    print("\n\n Re-generateing MCMC samples for the processed population datasets in two ways... \n\n")
    # Way 1 (DISCARDED) : fitting the model year by year
    # print('Way 1')
    # gpflow.config.set_default_float(np.float64)
    # gpflow.config.set_default_jitter(1e-4)
    
    # for i in range(Time):
    #     tf.random.set_seed(9861) 
    #     samples_grw_f = mcmc_grw_f(popu_0toT[i], env=False, num_burnin_steps = ci_niter(30000), num_samples = ci_niter(5000))
    #     pickle.dump(samples_grw_f, open(file = os.getcwd()+f"/true_popu/mcmc/samples_grw_f{i}.pkl", mode="wb"))

    #     tf.random.set_seed(9862) 
    #     samples_grw_nf = mcmc_grw_nf(popu_0toT[i], env=False, num_burnin_steps = ci_niter(30000), num_samples = ci_niter(5000))
    #     pickle.dump(samples_grw_nf, open(file = os.getcwd()+f"/true_popu/mcmc/samples_grw_nf{i}.pkl", mode="wb")) 

    #     tf.random.set_seed(98631) 
    #     samples_sur = mcmc_sur(popu_0toT[i], env=False, num_burnin_steps = ci_niter(30000), num_samples = ci_niter(5000))
    #     pickle.dump(samples_sur, open(file = os.getcwd()+f"/true_popu/mcmc/samples_sur{i}.pkl", mode="wb"))

    #     tf.random.set_seed(9864) 
    #     samples_fec = mcmc_fec(popu_0toT[i], env=False, num_burnin_steps = ci_niter(30000), num_samples = ci_niter(5000))
    #     pickle.dump(samples_fec, open(file = os.getcwd()+f"/true_popu/mcmc/samples_fec{i}.pkl", mode="wb"))

    #     tf.random.set_seed(98651) 
    #     samples_flow = mcmc_flow_poi(popu_0toT[i], env=False, num_burnin_steps = ci_niter(30000), num_samples = ci_niter(5000))
    #     pickle.dump(samples_flow, open(file = os.getcwd()+f"/true_popu/mcmc/samples_flow{i}.pkl", mode="wb"))
        


    # Way 2: for each vital rate, fitting a single model by using the whole datasets.
    print('Way 2')
    tf.random.set_seed(9863161551)
    samples_grw_f, p_samples_grw_f = mcmc_grw_f(popu_whole, env=True, num_burnin_steps = ci_niter(150000), num_samples = ci_niter(5000))
    pickle.dump(samples_grw_f, open(file = os.getcwd()+f"/true_popu/mcmc/samples_grw_f_whole.pkl", mode="wb"))
    pickle.dump(p_samples_grw_f, open(file = os.getcwd()+f"/true_popu/mcmc/p_samples_grw_f_whole.pkl", mode="wb"))


    tf.random.set_seed(9586471551)
    samples_grw_nf, p_samples_grw_nf = mcmc_grw_nf(popu_whole, env=True, num_burnin_steps = ci_niter(150000), num_samples = ci_niter(5000))
    pickle.dump(samples_grw_nf, open(file = os.getcwd()+f"/true_popu/mcmc/samples_grw_nf_whole.pkl", mode="wb")) 
    pickle.dump(p_samples_grw_nf, open(file = os.getcwd()+f"/true_popu/mcmc/p_samples_grw_nf_whole.pkl", mode="wb"))

    tf.random.set_seed(986855)
    samples_sur, p_samples_sur = mcmc_sur(popu_whole, env=True, num_burnin_steps = ci_niter(50000), num_samples = ci_niter(5000))
    # tf.random.set_seed(9868)
    # samples_sur, p_samples_sur = mcmc_sur(popu_whole, env=True, num_burnin_steps = ci_niter(30000), num_samples = ci_niter(5000))
    pickle.dump(samples_sur, open(file = os.getcwd()+f"/true_popu/mcmc/samples_sur_whole.pkl", mode="wb"))
    pickle.dump(p_samples_sur, open(file = os.getcwd()+f"/true_popu/mcmc/p_samples_sur_whole.pkl", mode="wb"))


    tf.random.set_seed(9869)
    samples_fec, p_samples_fec = mcmc_fec(popu_whole, env=True, num_burnin_steps = ci_niter(30000), num_samples = ci_niter(5000))
    pickle.dump(samples_fec, open(file = os.getcwd()+f"/true_popu/mcmc/samples_fec_whole.pkl", mode="wb"))
    pickle.dump(p_samples_fec, open(file = os.getcwd()+f"/true_popu/mcmc/p_samples_fec_whole.pkl", mode="wb"))


    tf.random.set_seed(98610112)
    samples_flow, p_samples_flow = mcmc_flow_poi(popu_whole, env=True, num_burnin_steps = ci_niter(1000), num_samples = ci_niter(5000))
    pickle.dump(samples_flow, open(file = os.getcwd()+f"/true_popu/mcmc/samples_flow_whole.pkl", mode="wb"))
    pickle.dump(p_samples_flow, open(file = os.getcwd()+f"/true_popu/mcmc/p_samples_flow_whole.pkl", mode="wb"))

 

# else:
#     print("\n\n Loading MCMC samples for the 'observed' population datasets... \n\n")
#     for i in range(Time):
#         globals()[f'samples_grw_f{i}'] = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_grw_f{i}.pkl", mode="rb"))
#         globals()[f'samples_grw_nf{i}'] = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_grw_nf{i}.pkl", mode="rb"))
#         globals()[f'samples_sur{i}'] = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_sur{i}.pkl", mode="rb"))
#         globals()[f'samples_fec{i}'] = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_fec{i}.pkl", mode="rb"))
#         globals()[f'samples_flow{i}'] = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_flow{i}.pkl", mode="rb"))
 
#     samples_grw_f_whole = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_grw_f_whole.pkl", mode="rb"))
#     samples_grw_nf_whole = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_grw_nf_whole.pkl", mode="rb"))
#     samples_sur_whole = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_sur_whole.pkl", mode="rb"))
#     samples_fec_whole = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_fec_whole.pkl", mode="rb"))
#     samples_flow_whole = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/samples_flow_whole.pkl", mode="rb")) 