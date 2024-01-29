import random
from model import AdaptationModel
import mesa
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import pandas as pd
import batch_run_config as config


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
    'adaption_threshold': 0.7
}


parameters_sensitivity_adaption_threshold = {
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
    'adaption_threshold': [0.35, 0.7, 1.15]
}


parameters_sensitivity_three_variables = {
    'network_type': 'watts_strogatz',
    'num_households': [30, 50, 70],
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
    'prob_positive_bias_change': [0.2, 0.5, 0.8],
    'prob_negative_bias_change': 0.1,
    'adaption_threshold': [0.5, 0.6, 0.7, 0.8, 0.9]
}


parameters_sensitivity_households = {
    'network_type': 'watts_strogatz',
    'num_households': [25, 50, 75],
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
    'adaption_threshold': 0.7
}


parameters_sensitivity_positive_prob = {
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
    'prob_positive_bias_change': [0.25, 0.5, 0.75],
    'prob_negative_bias_change': 0.1,
    'adaption_threshold': 0.7
}


parameters_sensitivity_negative_prob = {
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
    'prob_negative_bias_change': [0.05, 0.1, 0.15],
    'adaption_threshold': 0.7
}


parameters_sensitivity_bias_change_per_tick = {
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
    'bias_change_per_tick': [0.001, 0.02, 0.2],
    'flood_impact_on_bias_factor': 1,
    'prob_positive_bias_change': 0.5,
    'prob_negative_bias_change': 0.1,
    'adaption_threshold': 0.7
}

# all mehtods to display results for all parameters
def sensitivity_analysis_adapted_threshold(results):
    # use different scenarios by giving a list as different adaption threshold values in the parameters.
    results_df = pd.DataFrame(results)
    results_filtered = results_df[(results_df.Step == 80)]
    results_filtered[["iteration", "adaption_threshold", "total_adapted_households"]].reset_index(drop=True).head()

    # Create a figure with three subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Plot for total adapted households
    sns.boxplot(data=results_filtered,
                x="adaption_threshold",
                y="total_adapted_households",
                ax=axes[0])
    axes[0].set(xlabel="threshold needed for household to take adaption measurements",
                ylabel="total adapted households",
                title="Total Adapted Households")

    # Plot for total flood damage
    sns.boxplot(data=results_filtered,
                x="adaption_threshold",
                y="total_flood_damage",
                ax=axes[1])
    axes[1].set(xlabel="threshold needed for household to take adaption measurements",
                ylabel="total flood damage",
                title="Total Flood Damage")

    # Plot for total bias change
    sns.boxplot(data=results_filtered,
                x="adaption_threshold",
                y="bias_change_all_agents",
                ax=axes[2])
    axes[2].set(xlabel="threshold needed for household to take adaption measurements",
                ylabel="absolute bias change in system",
                title="Total Bias Change")

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the plots
    plt.show()


def sensitivity_analysis_households(results):
    # use different scenarios by giving a list as different household numbers in the parameters.
    results_df = pd.DataFrame(results)
    results_filtered = results_df[(results_df.Step == 80)]
    results_filtered[["bias_change_all_agents", "num_households", "total_adapted_households", ]].reset_index(drop=True).head()

    # Create a figure with three subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Plot for total adapted households
    sns.boxplot(data=results_filtered,
                x="num_households",
                y="total_adapted_households",
                ax=axes[0])
    axes[0].set(xlabel="number of households",
                ylabel="total adapted households",
                title="Total Adapted Households")

    # Plot for total flood damage
    sns.boxplot(data=results_filtered,
                x="num_households",
                y="total_flood_damage",
                ax=axes[1])
    axes[1].set(xlabel="number of households",
                ylabel="total flood damage",
                title="Total Flood Damage")

    # Plot for total bias change
    sns.boxplot(data=results_filtered,
                x="num_households",
                y="bias_change_all_agents",
                ax=axes[2])
    axes[2].set(xlabel="num_households",
                ylabel="absolute bias change in system",
                title="Total Bias Change")

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the plots
    plt.show()


def sensitivity_analysis_positive_chance(results):
    # use different scenarios by giving a list as different probabilities of changing the bias towards adaption
    # values in the parameters.
    results_df = pd.DataFrame(results)
    results_filtered = results_df[(results_df.Step == 80)]
    results_filtered[["iteration", "prob_positive_bias_change", "total_adapted_households"]].reset_index(drop=True).head()

    # Create a figure with three subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Plot for total adapted households
    sns.boxplot(data=results_filtered,
                x="prob_positive_bias_change",
                y="total_adapted_households",
                ax=axes[0])
    axes[0].set(xlabel="probability of household changing bias towards adaption",
                ylabel="total adapted households",
                title="Total Adapted Households")

    # Plot for total flood damage
    sns.boxplot(data=results_filtered,
                x="prob_positive_bias_change",
                y="total_flood_damage",
                ax=axes[1])
    axes[1].set(xlabel="probability of household changing bias towards adaption",
                ylabel="total flood damage",
                title="Total Flood Damage")

    # Plot for total bias change
    sns.boxplot(data=results_filtered,
                x="prob_positive_bias_change",
                y="bias_change_all_agents",
                ax=axes[2])
    axes[2].set(xlabel="probability of household changing bias towards adaption",
                ylabel="absolute bias change in system",
                title="Total Bias Change")

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the plots
    plt.show()


def sensitivity_analysis_negative_chance(results):
    # use different scenarios by giving a list as different adaption threshold values in the parameters.
    results_df = pd.DataFrame(results)
    results_filtered = results_df[(results_df.Step == 80)]
    results_filtered[["iteration", "prob_negative_bias_change", "total_adapted_households"]].reset_index(drop=True).head()

    # Create a figure with three subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Plot for total adapted households
    sns.boxplot(data=results_filtered,
                x="prob_negative_bias_change",
                y="total_adapted_households",
                ax=axes[0])
    axes[0].set(xlabel="probability of household changing bias against adaption",
                ylabel="total adapted households",
                title="Total Adapted Households")

    # Plot for total flood damage
    sns.boxplot(data=results_filtered,
                x="prob_negative_bias_change",
                y="total_flood_damage",
                ax=axes[1])
    axes[1].set(xlabel="probability of household changing bias against adaption",
                ylabel="total flood damage",
                title="Total Flood Damage")

    # Plot for total bias change
    sns.boxplot(data=results_filtered,
                x="prob_negative_bias_change",
                y="bias_change_all_agents",
                ax=axes[2])
    axes[2].set(xlabel="probability of household changing bias against adaption",
                ylabel="absolute bias change in system",
                title="Total Bias Change")

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the plots
    plt.show()


def sensitivity_analysis_bias_change_probability(results):
    results_df = pd.DataFrame(results)
    results_filtered = results_df[(results_df.Step == 80)]
    results_filtered[["iteration", "adaption_threshold", "total_adapted_households"]].reset_index(drop=True).head()

    # Combine multiple variables into one hue
    results_filtered['combined_hue'] = (results_filtered['prob_positive_bias_change'].astype(str) +
                                        " _ " + results_filtered['prob_negative_bias_change'].astype(str))

    # Specify a custom color palette with 10 distinct colors
    custom_palette = sns.color_palette("tab10", len(results_filtered['combined_hue'].unique()))

    # Create a scatter plot
    g = sns.scatterplot(data=results_filtered,
                        x='total_adapted_households',
                        y="bias_change_all_agents",
                        hue='combined_hue',
                        palette=custom_palette
                        )
    g.set(
        xlabel="total_adapted_households",
        ylabel="Total bias change all agents",
        title="effects of different bias spread probabilities on the agents adaption",
    )
    plt.show()


def different_adapted_threshold(results):
    results_df = pd.DataFrame(results)
    results_filtered = results_df
    results_filtered[["iteration", "adaption_threshold", "total_adapted_households"]].reset_index(drop=True).head()

    custom_palette = sns.color_palette("tab10", 10)
    # Create a scatter plot
    g = sns.lineplot(data=results_filtered,
                     x='Step',
                     y="Bias",
                     hue="adaption_threshold",
                     palette=custom_palette
                     )
    g.set(
        xlabel="Step",
        ylabel="Agent Bias (positive is pro adaption, negative is against adaption)",
        title="effects of different adaption thresholds on the agents bias towards adaption",
    )
    plt.show()


def different_sensitivities(results):
    combined_results = pd.DataFrame(results)
    # Create a bar chart for each input
    plt.figure(figsize=(12, 8))

    # Bar chart for adaption_threshold
    sns.barplot(data=combined_results, x='adaption_threshold', y='total_adapted_households', hue='num_households',
                palette='viridis', ci=None)
    plt.title('Effect of Adaption Threshold on Total Adapted Households')
    plt.xlabel('Adaption Threshold')
    plt.ylabel('Total Adapted Households')

    # Bar chart for prob_positive_bias_change
    plt.figure(figsize=(12, 8))
    sns.barplot(data=combined_results, x='prob_positive_bias_change', y='total_adapted_households',
                hue='num_households', palette='viridis', ci=None)
    plt.title('Effect of Prob Positive Bias Change on Total Adapted Households')
    plt.xlabel('Prob Positive Bias Change')
    plt.ylabel('Total Adapted Households')

    # Bar chart for num_households
    plt.figure(figsize=(12, 8))
    sns.barplot(data=combined_results, x='num_households', y='total_adapted_households', hue='adaption_threshold',
                palette='viridis', ci=None)
    plt.title('Effect of Num Households on Total Adapted Households')
    plt.xlabel('Num Households')
    plt.ylabel('Total Adapted Households')

    # Show the plots
    plt.show()
