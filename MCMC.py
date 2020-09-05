# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 21:28:29 2020

@author: Vivian Imbriotis
"""

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

from scipy.stats import norm

sns.set_style("dark")
sns.set_context("paper")

#generate normal data with sd 1 and mean 0
data = np.random.randn(20)

#define our prior as a gaussian...
mu_prior_mu = 0
mu_prior_sd = 1

posterior_chain = []

#Assuming we know the standard deviation
#and wish to estimate the mean...
mu_current = 0
proposal_width = 0.1
#Draw a new mu from a gaussian around current mu
for i in range(1000):
    mu_proposed = norm(mu_current,
                    proposal_width).rvs()
    likelihood_current = norm(mu_current,1).pdf(data).prod()
    prior_current = norm(mu_prior_mu,mu_prior_sd).pdf(mu_current)
    pr_current_given_data = likelihood_current*prior_current
    
    #how much more likely is the proposal than the current value, given the data?
    likelihood_proposed = norm(mu_proposed, 1).pdf(data).prod()
    prior_proposed = norm(mu_prior_mu,mu_prior_sd).pdf(mu_proposed)
    pr_proposal_given_data = likelihood_proposed * prior_proposed
    
    p_accept = pr_proposal_given_data / pr_current_given_data
    accept = np.random.rand() < p_accept
    if accept:
        mu_current = mu_proposed
        posterior_chain.append(mu_current)

x = np.linspace(-3,3,500)
fig, (data_ax, prior, posterior, trace) = plt.subplots(ncols = 4,
                                                       tight_layout = True,
                                                       figsize = [12,4])
prior.set_title("Prior of mean")
prior.set_xlabel("Mean of data")
prior.set_ylabel("Probability Density")
prior.vlines(norm(mu_prior_mu,mu_prior_sd).ppf((0.025,0.975)),*prior.get_ylim(),
             label = "95% credible interval", linestyle = "--")
prior.legend()
prior.plot(x, norm(mu_prior_mu,mu_prior_sd).pdf(x))
data_ax.set_title("Data")
data_ax.set_xlabel("Value")
data_ax.set_ylabel("Frequency")
data_ax.hist(data,bins = 10)
posterior.set_title("Empirical Posterior with MH MCMC")
posterior.set_xlabel("Mean of data")
posterior.set_ylabel("Frequency visited by Markov Chain")
posterior.hist(posterior_chain, bins = 20)
posterior.vlines(np.percentile(posterior_chain,(2.5,97.5)),*posterior.get_ylim(),
                 linestyle = "--")
for a in (data_ax,prior,posterior):
    a.set_xlim((-2.5,2.5))
trace.set_title("Trace of Markov Chain values")
trace.plot(posterior_chain)
fig.show()