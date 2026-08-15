from r1_data_generating import *
import bambi as bmb
import pymc as pm

# grw_f
df_grw_f = pd.DataFrame(XY_grw_f_compu(popu_whole, env=True)[0], columns=['size', 'age', 'tmax', 'tmin', 'precip'])
df_grw_f['size_Next'] = XY_grw_f_compu(popu_whole, env=True)[1]
model_grw_f = bmb.Model("size_Next ~ size + age + tmax + tmin + precip", df_grw_f)
trace_grw_f = model_grw_f.fit(draws=5000, tune=100000, discard_tuned_samples=True, chains=1, progressbar=False)
pickle.dump(trace_grw_f, open(file = os.getcwd()+"/true_popu/glm_mcmc/trace_grw_f.pkl", mode="wb"))

# grw_nf
df_grw_nf = pd.DataFrame(XY_grw_nf_compu(popu_whole, env=True)[0], columns=['size', 'age', 'tmax', 'tmin', 'precip'])
df_grw_nf['size_Next'] = XY_grw_nf_compu(popu_whole, env=True)[1]
model_grw_nf = bmb.Model("size_Next ~ size + age + tmax + tmin + precip", df_grw_nf)
trace_grw_nf = model_grw_nf.fit(draws=5000, tune=100000, discard_tuned_samples=True, chains=1, progressbar=False)
pickle.dump(trace_grw_nf, open(file = os.getcwd()+"/true_popu/glm_mcmc/trace_grw_nf.pkl", mode="wb"))

# sur
df_sur = pd.DataFrame(XY_sur_compu(popu_whole, env=True)[0], columns=['size', 'age', 'tmax', 'tmin', 'precip'])
df_sur['size_Next'] = XY_sur_compu(popu_whole, env=True)[1]
model_sur = bmb.Model("size_Next ~ size + age + tmax + tmin + precip", df_sur, family='bernoulli')
trace_sur = model_sur.fit(draws=5000, tune=100000, discard_tuned_samples=True, chains=1, progressbar=False)
pickle.dump(trace_sur, open(file = os.getcwd()+"/true_popu/glm_mcmc/trace_sur.pkl", mode="wb"))

# fec
df_fec = pd.DataFrame(XY_fec_compu(popu_whole, env=True)[0], columns=['size', 'age', 'tmax', 'tmin', 'precip'])
df_fec['size_Next'] = XY_fec_compu(popu_whole, env=True)[1]
model_fec = bmb.Model("size_Next ~ size + age + tmax + tmin + precip", df_fec, family='bernoulli')
trace_fec = model_fec.fit(draws=5000, tune=100000, discard_tuned_samples=True, chains=1, progressbar=False)
pickle.dump(trace_fec, open(file = os.getcwd()+"/true_popu/glm_mcmc/trace_fec.pkl", mode="wb"))

# flow
df_flow = pd.DataFrame(XY_flow_compu(popu_whole, env=True)[0], columns=['size', 'age', 'tmax', 'tmin', 'precip'])
df_flow['size_Next'] = XY_flow_compu(popu_whole, env=True)[1]
model_flow = bmb.Model("size_Next ~ size + age + tmax + tmin + precip", df_flow, family='poisson')
trace_flow = model_flow.fit(draws=5000, tune=100000, discard_tuned_samples=True, chains=1, progressbar=False)
pickle.dump(trace_flow, open(file = os.getcwd()+"/true_popu/glm_mcmc/trace_flow.pkl", mode="wb"))











