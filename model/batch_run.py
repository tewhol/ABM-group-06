import random
from model import AdaptationModel
import mesa
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import pandas as pd

# Input values for the batch run as dictionary called parameters. Each key-value pair corresponds with a model input
# variable and its defined input value. By declaring multiple input variables in a tuple the batch_run function of
# mesa can do different senarios in which it runs each possible combination of input variables from the parameter
# dictionary.
parameters = {
    'network_type': 'watts_strogatz',
    'num_households': 50,
    'prob_network_connection': 0.4,
    'num_edges': 3,
    'num_nearest_neighbours': 5,
    'flood_time_ticks': 5,
    'lower_bound_flood_severity_probability': 0.5,
    'higher_bound_flood_severity_probability': 1.2,
    'random_seed': 1,
    'wealth_factor': 0.2,
    'wealth_distribution_type': 'UI',
    'wealth_distribution_range_min': 0,
    'wealth_distribution_range_max': 3,
    'has_child_factor': 0.2,
    'has_child_distribution_type': 'B',
    'has_child_distribution_value': 0.4,
    'house_size_factor': 0.05,
    'house_size_distribution_type': 'UI',
    'house_size_distribution_range_min': 0,
    'house_size_distribution_range_max': 2,
    'house_type_factor': 0.05,
    'house_type_distribution_type': 'UI',
    'house_type_distribution_range_min': 0,
    'house_type_distribution_range_max': 1,
    'education_level_factor': 0.05,
    'education_level_distribution_type': 'UI',
    'education_level_distribution_range_min': 0,
    'education_level_distribution_range_max': 3,
    'social_preference_factor': 0.2,
    'social_preference_distribution_type': 'U',
    'social_preference_distribution_range_min': -1,
    'social_preference_distribution_range_max': 1,
    'age_factor': 0.2,
    'age_distribution_type': 'N',
    'age_distribution_mean': 33.4,
    'age_distribution_std_dev': 5,
    'household_tolerance': 0.15,
    'bias_change_per_tick': 0.02,
    'flood_impact_on_bias_factor': 1,
    'prob_positive_bias_change': 0.5,
    'prob_negative_bias_change': 0.1,
    'adaption_threshold': 0.5
}

# Other input variables for the mesa batch run function
iterations = 2  # number of iterations for each parameter combination
max_steps = 80  # max steps of each model run/ iteration
number_processes = None  # how many processors are used
data_collection_period = 1  # number of steps after which data is collected by the model and agent data collectors
display_progress = True  # To display the progress on the batch runs


def bias_change_over_time(results):
    results_df = pd.DataFrame(results)
    results_filtered = results_df
    results_filtered[["Step", "Bias"]].reset_index(drop=True).head()
    # Create a scatter plot
    g = sns.lineplot(data=results_filtered,
                     x="Step",
                     y="Bias",
                     hue='AgentID')
    g.set(
    xlabel="Step",
    ylabel="Bias change in network",
    title="Bias change over time",
    )
    plt.show()


def number_of_adaptions(results):
    results_df = pd.DataFrame(results)
    results_filtered = results_df[(results_df.Step == 80)]
    results_filtered[["total_adapted_households", "bias_change_all_agents"]].reset_index(drop=True).head()
    # Create a scatter plot
    g = sns.scatterplot(data=results_filtered,
                        x="bias_change_all_agents",
                        y="total_adapted_households")
    g.set(
    xlabel="Total bias change in network",
    ylabel="total adapted households",
    title="Bias change vs flood adaption",
    )
    plt.show()


def total_number_of_damage(results):
    results_df = pd.DataFrame(results)
    results_filtered = results_df[(results_df.Step == 80)]
    results_filtered[["total_adapted_households", "total_flood_damage", "bias_change_all_agents"]].reset_index(drop=True).head()
    # Create a scatter plot
    g = sns.scatterplot(data=results_filtered,
                        x="total_adapted_households",
                        y="total_flood_damage")
    g.set(
    xlabel="total adapted households",
    ylabel="total flood damage",
    title="adaption vs flood damage",
    )
    plt.show()


def sensitivity_analysis_adapted_threshold(results):
    results_df = pd.DataFrame(results)
    results_filtered = results_df[(results_df.Step == 80)]
    results_filtered[["iteration","adaption_threshold", "total_adapted_households"]].reset_index(drop=True).head()
    print(results_filtered.keys())
    # Create a scatter plot
    g = sns.pointplot(data=results_filtered,
                      y="total_adapted_households",
                      linestyles='none',
                      hue="adaption_threshold"
                      )
    g.set(
        xlabel="adaption_threshold",
        ylabel="total_adapted_households",
        title="sensitivity of adaption threshold on total adapted households",
    )
    plt.show()


if __name__ == "__main__":
    results = mesa.batch_run(
        AdaptationModel,
        parameters=parameters,
        iterations=iterations,
        max_steps=max_steps,
        number_processes=number_processes,
        data_collection_period=data_collection_period,
        display_progress=display_progress,
    )
