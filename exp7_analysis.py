import numpy as np 
import matplotlib.pyplot as plt 

beta_0_accs = []
beta_point1_accs = []
beta_point3_accs = []
beta_point5_accs = []
beta_point9_accs = []

for run_num in range(5):
    beta_0_accs.append(np.load("beta_0.0_run_"+str(run_num)+"/accs.npy"))
    beta_point1_accs.append(np.load("beta_0.1_run_"+str(run_num)+"/accs.npy"))
    beta_point3_accs.append(np.load("beta_0.3_run_"+str(run_num)+"/accs.npy"))
    beta_point5_accs.append(np.load("beta_0.5_run_"+str(run_num)+"/accs.npy"))
    beta_point9_accs.append(np.load("beta_0.9_run_"+str(run_num)+"/accs.npy"))

xs = range(101)
accs_beta_0_means = np.mean(beta_0_accs, axis=0)
accs_beta_point1_means = np.mean(beta_point1_accs, axis=0)
accs_beta_point3_means = np.std(beta_point3_accs, axis=0)
accs_beta_point5_means = np.std(beta_point5_accs, axis=0)
accs_beta_point9_means = np.std(beta_point9_accs, axis=0)


accs_beta_0_stddevs = np.std(beta_0_accs, axis=0)
accs_beta_point1_stddevs = np.std(beta_point1_accs, axis=0)
accs_beta_point3_stddevs = np.std(beta_point3_accs, axis=0)
accs_beta_point5_stddevs = np.std(beta_point5_accs, axis=0)
accs_beta_point9_stddevs = np.std(beta_point9_accs, axis=0)


plt.plot(xs, accs_beta_0_means, label="$\beta = 0$" )
plt.plot(xs, accs_beta_point1_means, label="$\beta = 0.1$")
plt.plot(xs, accs_beta_point3_means, label="$\beta = 0.3$")
plt.plot(xs, accs_beta_point5_means, label="$\beta = 0.5$")
plt.plot(xs, accs_beta_point9_means, label="$\beta = 0.9$")
plt.fill_between(xs, accs_beta_0_means - accs_beta_0_stddevs,accs_beta_0_means + accs_beta_0_stddevs, alpha=0.4)
plt.fill_between(xs, accs_beta_point1_means - accs_beta_point1_stddevs,accs_beta_point1_means + accs_beta_point1_stddevs, alpha=0.4)
plt.fill_between(xs, accs_beta_point3_means - accs_beta_point3_stddevs,accs_beta_point3_means + accs_beta_point3_stddevs, alpha=0.4)
plt.fill_between(xs, accs_beta_point5_means - accs_beta_point5_stddevs,accs_beta_point5_means + accs_beta_point5_stddevs, alpha=0.4)
plt.fill_between(xs, accs_beta_point9_means - accs_beta_point9_stddevs,accs_beta_point9_means + accs_beta_point9_stddevs, alpha=0.4)