import random
from model import AdaptationModel
import matplotlib.pyplot as plt
import networkx as nx

# Input values for network dynamics
network_type = 'watts_strogatz'  # The social network structure that is used
num_households = 50  # number of household agents
prob_network_connection = 0.4  # likeliness of edge being created between two nodes
num_edges = 3  # number of edges for BA network
num_nearest_neighbours = 5  # number of nearest neighbours for WS social network
flood_time_ticks = [5]  # time of flood
lower_bound_flood_severity_probability = 0.5  # Bounds between a Uniform distribution of the flood severity
higher_bound_flood_severity_probability = 1.2
random_seed = 1  # The seed used to generate pseudo random numbers

# Agent attributes; the distribution types and the corresponding variables, needed for these distributions
wealth_factor = 0.2
wealth_distribution_type = 'UI'
wealth_distribution_range_min = 0
wealth_distribution_range_max = 3

has_child_factor = 0.2
has_child_distribution_type = 'B'
has_child_distribution_value = 0.2

house_size_factor = 0.05
house_size_distribution_type = 'UI'
house_size_distribution_range_min = 0
house_size_distribution_range_max = 2

house_type_factor = 0.05
house_type_distribution_type = 'UI'
house_type_distribution_range_min = 0
house_type_distribution_range_max = 1

education_level_factor = 0.05
education_level_distribution_type = 'UI'
education_level_distribution_range_min = 0
education_level_distribution_range_max = 3

social_preference_factor = 0.2
social_preference_distribution_type = 'U'
social_preference_distribution_range_min = -1
social_preference_distribution_range_max = 1

age_factor = 0.2
age_distribution_type = 'N'
age_distribution_mean = 33.4
age_distribution_std_dev = 5

# Input values for agent interaction dynamics
household_tolerance = 0.15  # the tolerance level for each agent
bias_change_per_tick = 0.02  # Bias change per tick when an agent when it's influenced by its network
flood_impact_on_bias_factor = 1  # Defines factor regarding the actual and expected damage of flooding
prob_positive_bias_change = 0.5  # Probability that agent changes it's bias positively
prob_negative_bias_change = 0.1  # Probability that agent changes it's bias negatively
adaption_threshold = 0.7  # Threshold of bias an agent needs to adapt


def run_model(iterations):
    all_results = []

    for iteration in range(iterations):
        # Create dictionaries that serve as model input
        network_dynamics_dictionary = {
            "network": network_type,
            "number_of_households": num_households,
            "probability_of_network_connection": prob_network_connection,
            "number_of_edges": num_edges,
            "number_of_nearest_neighbours": num_nearest_neighbours,
            "flood_time_tick": flood_time_ticks,
            "flood_severity_probability": (
                lower_bound_flood_severity_probability, higher_bound_flood_severity_probability),
            "seed": random_seed
        }

        attribute_dictionary = {
            "wealth": [wealth_factor, wealth_distribution_type,
                       (wealth_distribution_range_min, wealth_distribution_range_max)],
            "has_child": [has_child_factor, has_child_distribution_type, (has_child_distribution_value)],
            "house_size": [house_size_factor, house_size_distribution_type,
                           (house_size_distribution_range_min, house_size_distribution_range_max)],
            "house_type": [house_type_factor, house_type_distribution_type,
                           (house_type_distribution_range_min, house_type_distribution_range_max)],
            "education_level": [education_level_factor, education_level_distribution_type,
                                (education_level_distribution_range_min, education_level_distribution_range_max)],
            "social_preference": [social_preference_factor, social_preference_distribution_type, (
                social_preference_distribution_range_min, social_preference_distribution_range_max)],
            "age": [age_factor, age_distribution_type, (age_distribution_mean, age_distribution_std_dev)]
        }

        agent_interaction_dictionary = {
            "household_tolerance": household_tolerance,
            "bias_change_per_tick": bias_change_per_tick,
            "flood_impact_on_bias_factor": flood_impact_on_bias_factor,
            "probability_positive_bias_change": prob_positive_bias_change,
            "probability_negative_bias_change": prob_negative_bias_change,
            "adaption_threshold": adaption_threshold
        }

        # Initialize the Adaptation Model with 50 household agents.
        model = AdaptationModel(
            network_dynamics_dictionary=network_dynamics_dictionary,
            attribute_dictionary=attribute_dictionary,
            agent_interaction_dictionary=agent_interaction_dictionary
        )

        # Run the model for 80 steps and generate plots every 5 steps.
        for step in range(80):
            model.step()

        agent_data = model.datacollector.get_agent_vars_dataframe()
        model_data = model.datacollector.get_model_vars_dataframe()

        result = [iteration, agent_data, model_data]
        all_results.append(result)

    return all_results


if __name__ == "__main__":
    iterations = 30
    run_model(iterations)
