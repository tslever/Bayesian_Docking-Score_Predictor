import arviz
import numpy as np
from matplotlib import pyplot as plt
import pymc
import pymc_bart

def main():

    RANDOM_SEED = 5781
    np.random.seed(RANDOM_SEED)

    '''
    https://www.pymc.io/projects/examples/en/latest/case_studies/BART_introduction.html
    BART overview
    Bayesian Additive Regression Trees (BART) is a non-parametric regression approach.
    If we have some covariates X [e.g., vector of centers of bins of histogram] and we want to use them to model Y [e.g., vector of frequencies corresponding to bins],
    a BART model (omitting the priors) can be represented as
    Y = f(X) + epsilon = BART(X) + epsilon
    where we use a sum of predictions of m regression trees to model f, and epsilon is some noise.
    In the most typical examples epsilon is normally distributed, i.e., epsilon ~ N(mu = 0, sigma).
    So we can also write
    Y ~ N[u = BART(X), sigma]
    In principle nothing restricts us from using a sum of predictions of trees to model another relationship. For example we may have
    Y ~ Poisson[mu = BART(X)]
    One of the reasons BART is Bayesian is the use of prior probability density distributions over the regression trees.
    The prior probability density distributions are defined in such a way that they favor shallow trees with leaf values close to zero.
    A key idea is that a single tree in a BART model is not very good at fitting the data but when we sum many of these trees we get a good and flexible approximation.

    Coal mining with BART
    To better understand BART in practice we are going to use the oldie but goldie coal mining disaster dataset.
    Demonstrating using the coal mining disaster dataset is one of the classic examples in PyMC.
    Instead of thinking about a switch-point model with two Poisson distributions, as in the original PyMC example,
    we are going to think about a non-parametric regression with a Poisson response.
    Because our data is just a single column with dates, we need to do some pre-processing.
    We are going to discretize the data, just as if we were building a histogram.
    We are going to use the centers of the bins as the variable X and the counts per bin as the variable Y.
    '''

    array_of_years_and_numbers_of_coal_mining_disasters_in_the_UK = np.loadtxt('Data_Frame_Of_Years_And_Numbers_Of_Coal_Mining_Disasters_In_The_UK.csv', delimiter = ',')
    array_of_years = array_of_years_and_numbers_of_coal_mining_disasters_in_the_UK[:, 0]
    array_of_numbers_of_coal_mining_disasters_corresponding_to_years = array_of_years_and_numbers_of_coal_mining_disasters_in_the_UK[:, 1]
    difference_between_maximum_and_minimum_years = int(array_of_years.max() - array_of_years.min())
    number_of_bins = difference_between_maximum_and_minimum_years // 4 # // divides by 4 and converts to int
    array_of_numbers_of_coal_mining_disasters_corresponding_to_bins_of_years, array_of_edges_of_bins_of_years = np.histogram(array_of_years, bins = number_of_bins, weights = array_of_numbers_of_coal_mining_disasters_corresponding_to_years)
    array_of_numbers_of_coal_mining_disasters_corresponding_to_bins_of_years, array_of_edges_of_bins_of_years, bar_container_of_27_artists = plt.hist(array_of_years, bins = number_of_bins, weights = array_of_numbers_of_coal_mining_disasters_corresponding_to_years)
    one_dimensional_array_of_centers_of_bins_of_years = array_of_edges_of_bins_of_years[:-1] + (array_of_edges_of_bins_of_years[1] - array_of_edges_of_bins_of_years[0]) / 2
    two_dimensional_array_of_centers_of_bins_of_years = one_dimensional_array_of_centers_of_bins_of_years[:, np.newaxis]

    '''
    In PyMC a BART variable can be defined very similarly to other random variables.
    One important difference is that we have to pass our X's [e.g., vector of centers of bins of histogram]
    and Y's [e.g., vector of frequencies corresponding to bins] to the BART variable.
    A BART variable is used when sampling trees
    The prior probability density distribution over the sum of predictions of trees is so huge that without any information from our data
    training a BART model will be an impossible task.
    Here we are also making explicit that we are going to use a sum over 20 trees (m = 20).
    Low number of trees like 20 could be good enough for simple models like the following PyMC model on the coal mining disaster dataset
    and could also work very well as a quick approximation for more complex models, in particular during early stages of modeling,
    when we may want to try a few things as quickly as possible in order to better grasp which model may be a good idea for our problem.
    When creating simple models or approximating more complex models, once we have more certainty about the model(s) we really like,
    we can improve the approximation by increasing m.
    In literature it is common to find reports of good results with numbers like 50, 100, or 200.
    '''

    with pymc.Model() as pymc_model:
        tensor_variable_representing_natural_logarithm_based_pymc_bart_BART_model = pymc_bart.BART("tensor_variable_representing_natural_logarithm_based_pymc_bart_BART_model", X = two_dimensional_array_of_centers_of_bins_of_years, Y = np.log(array_of_numbers_of_coal_mining_disasters_corresponding_to_bins_of_years), m = 20)
        tensor_variable_representing_pymc_bart_BART_model_and_parameter_and_mean_mu = pymc.Deterministic("tensor_variable_representing_pymc_bart_BART_model_and_parameter_and_mean_mu", pymc.math.exp(tensor_variable_representing_natural_logarithm_based_pymc_bart_BART_model))
        tensor_variable_representing_pymc_Poisson_model_and_predictions = pymc.Poisson("tensor_variable_representing_pymc_Poisson_model_and_predictions", mu = tensor_variable_representing_pymc_bart_BART_model_and_parameter_and_mean_mu, observed = array_of_numbers_of_coal_mining_disasters_corresponding_to_bins_of_years)
        inference_data_with_groups_posterior_sample_stats_and_observed_data = pymc.sample(random_seed = RANDOM_SEED)

    '''
    Before checking the result, we need to discuss one more detail.
    The BART variable always samples over the set of all real numbers, meaning that in principle we can get values that go from negative infinity to infinity.
    Thus, we may need to transform predictions of a pymc_bart BART model.
    For example, in the PYMC model of number of coal mining disasters vs. bin of years
    we computed a tensor variable representing a natural logarithm based PYMC_bart BART model and exponentiated this variable
    because the Poisson probability density distribution expects values that go from 0 to infinity.
    This is business as usual.
    The novelty is that we may need to apply the inverse transformation to the values of Y [e.g., vector of frequencies corresponding to bins],
    as we did in computing our natural logarithm based PYMC bart BART model when we took the natural logarithm of Y.
    The main reason to apply the inverse transformation is that the values of Y are used to get reasonable initial values for the sum of predictions of trees
    and the variance of the predictions of leaf nodes.
    Thus, applying the inverse transformation is a simple way to improve the efficiency and accuracy of the PYMC Poisson model.
    Should we apply the inverse transformation for every possible likelihood?
    Well, no.
    If we are using a BART model for the location parameter of normal or Student's t probability density distributions, we don't need to apply the inverse transformation
    as the support for these distributions is the set of all real numbers.
    We should apply a logistic function to a pymc_bart BART model if the transformed BART model will be used to provide a 0 or 1 indicating failure or success
    to a PYMC Bernoulli model.
    In this case, there is no need to apply the inverse of the logistic function to Y; pymc_bart takes care of this case.

    Now let's see the result of pymc_model.
    '''

    _, ax = plt.subplots(figsize = (10, 6))
    number_of_chains_or_jobs = 4
    '''
    A chain is one of the individual sequences of samples from the posterior probability density distribution of the parameters of the pymc_bart BART model
    generated during a Markov Chain Monte Carlo simulation.
    Each chain represents a trajectory of parameter values as the MCMC algorithm explores the posterior probability density distribution.
    We run multiple chains in parallel (often on separate CP cores or even different machines) to help diagnose convergence
    and to reduce the risk of getting stuck in a local minimum of the posterior probability density distribution.
    By examining multiple chains, we can assess whether the chains have converged to the same posterior probability density distribution
    and whether they exhibit similar patterns.
    If all chains look similar, the MCMC algorithm has likely converged to the true posterior probability density distribution.
    If they look quite different, there might be isues with the converge of the algorithm.
    '''
    data_array_of_pooled_predicted_numbers_of_disasters_27_bins_of_years_by_1000_draws_by_4_chains = inference_data_with_groups_posterior_sample_stats_and_observed_data.posterior['tensor_variable_representing_pymc_bart_BART_model_and_parameter_and_mean_mu'] / number_of_chains_or_jobs
    '''
    When we divide a data array by a number of chains, we are pooling the samples from all the chains.
    We divide the data array by the number of chains in the process of computing statistics or performing inference on the posterior probability density distribution
    of our pymc_bart BART model while taking into account the multiple chains used in the MCMC sampling process.
    Dividing the data array by the number of chains ensures that our posterior probability density distribution summary statistics and plots are based
    on a combined, pooled dataset that accurately represents the posterior probability density distribution of our pymc_bart BART model.
    Different chains may explore the posterior probability density distribution differently, leading to some degree of chain-specific variability.
    Pooling the samples reduces this variabillity, providing a more stable estimate of the posterior probability density distribution.
    By pooling samples from multiple chains, we increase statistical efficiency and effective sample size, which can improve the accuracy of
    our posterior probability density distribution estimates and make it easier to compute more precise summary statistics.
    When we divide a data array by a number of chains, we're essentially taking the average of the samples from each chain for each parameter or variable.
    This average is used for posterior probability density distribution inference and summarization.
    '''
    array_of_means_of_pooled_predicted_numbers_of_disasters_one_for_each_bin_of_years = data_array_of_pooled_predicted_numbers_of_disasters_27_bins_of_years_by_1000_draws_by_4_chains.mean(dim = ['draw', 'chain'])
    ax.plot(one_dimensional_array_of_centers_of_bins_of_years, array_of_means_of_pooled_predicted_numbers_of_disasters_one_for_each_bin_of_years, 'w', lw = 3, label = 'mean of pooled predicted numbers of disasters')
    ax.plot(one_dimensional_array_of_centers_of_bins_of_years, array_of_numbers_of_coal_mining_disasters_corresponding_to_bins_of_years / number_of_chains_or_jobs, 'k.', label = 'pooled number of disasters')
    arviz.plot_hdi(one_dimensional_array_of_centers_of_bins_of_years, data_array_of_pooled_predicted_numbers_of_disasters_27_bins_of_years_by_1000_draws_by_4_chains, hdi_prob = 0.95, smooth = False)
    ax.text(0.5, 0.5, "95% HDI for mean", transform=plt.gca().transAxes)
    '''
    The Highest Posterior Probability Density Interval (HDI) or Credible Interval is a way to describe the uncertainty or variability
    associated with a parameter estimate in a Bayesian framework.
    In Bayesian statistics, we work with probability density distributions to represent our uncertainty about a parameter of interest.
    The HDI is an interval on a parameter's probability density distribution that contains the most probable values for the parameter.
    Specifically, the HDI is the narrowest interval that contains a specified percentage of the probability density distribution's total probability density / area.
    For example, a 95-percent HDI would be the narrowest interval that contains 95 percent of the total probability density for the parameter.
    The HDI represents the range of values for the parameter that are most credible or probable given the available data and the chosen prior probability distribution.
    To compute the HDI in a Bayesian analysis, you would typically use tools like Markov Chain Monte Carlo (MCMC) or other Bayesian inference methods
    to sample from the posterior probability density distribution of the parameter and then identify the interval that contains the specified percentage of the samples
    with the highest density.
    The HDI is a way to quantify uncertainty in a Bayesian parameter estimate by providing an interval that contains the most probable values for the parameter
    based on the available data and prior knowledge.
    
    hdi_prob is a parameter that we can specify when computing the HDI in Bayesian analysis.
    hdi_prob determines the percentage of the total probability density distribution that you want the HDI to encompass.
    The default value for hdi_prob can vary depending on the software or library you are using for Bayesian analysis.
    In many cases, a common default value for hdi_prob is 0.95, which corresponds to a 95-percent HDI.
    When you set hdi_prob to 0.95, you are asking the software or library to compute the narrowest interval that contains 95 percent of the total probability density
    of the parameter's posterior density distribution.
    A 95 percent HDI is often used to summarize the credible range of parameter values in Bayesian analysis,
    Other values such as 0.90 (for a 90-percent HDI) or 0.99 (for a 99-percent HDI) can also be specified depending on the specific requirements of the analysis.
    '''
    arviz.plot_hdi(one_dimensional_array_of_centers_of_bins_of_years, data_array_of_pooled_predicted_numbers_of_disasters_27_bins_of_years_by_1000_draws_by_4_chains, hdi_prob = 0.5, smooth = False, plot_kwargs = {'alpha': 0})
    ax.text(0.6, 0.4, "50% HDI for mean", transform=plt.gca().transAxes)
    ax.plot(array_of_years, np.zeros_like(array_of_numbers_of_coal_mining_disasters_corresponding_to_years) - 0.5, 'k|', label = 'date of datum')
    ax.set_xlabel('year / center of bin of years')
    ax.set_ylabel('mean of pooled predicted numbers of disasters')
    ax.set_title('Mean Of Pooled Predicted Numbers Of Disasters Vs. Center Of Bin Of Years')
    ax.set_xlim(left = array_of_years.min(), right = array_of_years.max())
    ax.legend()
    plt.show()
    '''
    The white line in our plot shows the mean of pooled predicted numbers of disasters vs. center of bins of years.
    The dark orange band represents the 50-percent HDI.
    The light orange band represents the 95-percent HDI.
    We can see a rapid decrease in disasters between 1880 and 1900.
    In our plot the white line is the mean over 4000 pooled predicted numbers of disasters drawn from the posterior probability distribution for the mean number of disasters.
    Each draw from the posterior probability distribution for the mean number of disasters is a sum over m = 20 trees.
    '''

if __name__ == '__main__':
    main()