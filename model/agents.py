# Importing necessary libraries
import random
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy
from statistics import mean
import numpy as np

# Import functions from functions.py
from functions import generate_random_location_within_map_domain, get_flood_depth, calculate_basic_flood_damage, \
    floodplain_multipolygon


# Define the Households agent class
class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.
    """

    def __init__(self, unique_id, model,
                 radius_network,  # Size of social network radius
                 tolerance,  # Tolerance level for each agent
                 bias_change_per_tick,  # Bias change per tick when an agent is influenced by its network
                 flood_impact_on_bias_factor,  # Factor regarding the actual and expected damage of flooding
                 adaption_threshold,  # Threshold of bias an agent needs to adapt
                 probability_positive_bias_change,  # Probability that agent changes its bias positively
                 probability_negative_bias_change,  # Probability that agent changes its bias negatively
                 wealth_factor,  # Factor affecting the importance of wealth on the household identity
                 wealth_distribution_type,  # Type of distribution for wealth (e.g., 'UI' for Uniform, 'N' for Normal)
                 wealth_distribution_range_min,  # Range for wealth distribution
                 wealth_distribution_range_max,
                 has_child_factor,  # Factor affecting the importance of has_child on the household identity
                 has_child_distribution_type,  # Type of distribution for has_child (e.g., 'B' for Bernoulli)
                 has_child_distribution_value,  # Value for has_child distribution
                 house_size_factor,  # Factor affecting the importance of house_size on the household identity
                 house_size_distribution_type,  # Type of distribution for house_size (e.g., 'UI' for Uniform)
                 house_size_distribution_range_min,  # Range for house_size distribution
                 house_size_distribution_range_max,
                 house_type_factor,  # Factor affecting the importance of house_type on the household identity
                 house_type_distribution_type,  # Type of distribution for house_type (e.g., 'UI' for Uniform)
                 house_type_distribution_range_min,  # Range for house_type distribution
                 house_type_distribution_range_max,
                 education_level_factor,  # Factor affecting the importance of education_level on the household identity
                 education_level_distribution_type,  # Type of distribution for education_level (e.g., 'UI' for Uniform)
                 education_level_distribution_range_min,  # Range for education_level distribution
                 education_level_distribution_range_max,
                 social_preference_factor,  # Factor affecting the importance of social_preference on the household identity
                 social_preference_distribution_type,  # Type of distribution for social_preference (e.g., 'U' for Uniform)
                 social_preference_distribution_range_min,  # Range for social_preference distribution
                 social_preference_distribution_range_max,
                 age_factor,  # Factor affecting the importance of age on the household identity
                 age_distribution_type,  # Type of distribution for age (e.g., 'N' for Normal)
                 age_distribution_mean,  # Parameters for age distribution
                 age_distribution_std_dev,  # Parameters for age distribution
                 ):

        super().__init__(unique_id, model)
        self.is_adapted = False  # Initial adaptation status set to False
        self.actual_flood_impact_on_bias = 0  # Initial variable about how a real flood would impact an agent's bias

        # An attribute representing the built-up bias in an agent's network
        self.network_bias = 0
        self.tolerance = tolerance
        self.bias_change_per_tick = bias_change_per_tick
        self.flood_impact_on_bias_factor = flood_impact_on_bias_factor
        self.adaption_threshold = adaption_threshold
        self.probability_positive_bias_change = probability_positive_bias_change
        self.probability_negative_bias_change = probability_negative_bias_change

        # Attributes related to the size of one's social network, the radius, and list of (friends/friends of friends)
        self.radius_network = radius_network if radius_network is not None else 1
        self.social_network = []

        # Attributes directly related to the households' identity
        self.wealth = 0  # Potential values, integer: 0 is low-income, 1 below average, 2 above average, 3 rich
        self.house_type = 0  # Potential values: 0 is an apartment in a block, and 1 is a freestanding house
        self.house_size = 0  # Potential values: simulating different types of sizes each house has
        self.has_child = False  # Potential values, boolean: True or false about having a child
        self.age = 0  # Potential values: at least 18
        self.social_preference = 0  # Potential values: between introvert -1 and extrovert 1
        self.education_level = 0  # Potential values, integer: 0 low-level - 3 high-level education
        self.identity = 0  # Potential values: between 0 and 1

        # Additional class attributes
        self.wealth_factor = wealth_factor
        self.wealth_distribution_type = wealth_distribution_type
        self.wealth_distribution_range = (wealth_distribution_range_min, wealth_distribution_range_max)
        self.has_child_factor = has_child_factor
        self.has_child_distribution_type = has_child_distribution_type
        self.has_child_distribution_value = has_child_distribution_value
        self.house_size_factor = house_size_factor
        self.house_size_distribution_type = house_size_distribution_type
        self.house_size_distribution_range = (house_size_distribution_range_min, house_size_distribution_range_max)
        self.house_type_factor = house_type_factor
        self.house_type_distribution_type = house_type_distribution_type
        self.house_type_distribution_range = (house_type_distribution_range_min, house_type_distribution_range_max)
        self.education_level_factor = education_level_factor
        self.education_level_distribution_type = education_level_distribution_type
        self.education_level_distribution_range = (education_level_distribution_range_min, education_level_distribution_range_max)
        self.social_preference_factor = social_preference_factor
        self.social_preference_distribution_type = social_preference_distribution_type
        self.social_preference_distribution_range = (social_preference_distribution_range_min, social_preference_distribution_range_max)
        self.age_factor = age_factor
        self.age_distribution_type = age_distribution_type
        self.age_distribution_params = (age_distribution_mean, age_distribution_std_dev)

        # getting flood map values
        # Get a random location on the map
        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)

        # Check whether the location is within floodplain
        self.in_floodplain = False
        if contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y):
            self.in_floodplain = True

        # Get the estimated flood depth at those coordinates. the estimated flood depth is calculated based on the
        # flood map (i.e., past data) so this is not the actual flood depth. Flood depth can be negative if the
        # location is at a high elevation
        self.flood_depth_estimated = get_flood_depth(corresponding_map=model.flood_map, location=self.location,
                                                     band=model.band_flood_img)
        # handle negative values of flood depth
        if self.flood_depth_estimated < 0:
            self.flood_depth_estimated = 0

        # Calculate the estimated flood damage given the estimated flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated)

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since
        # there is not flood yet and will update its value when there is a shock (i.e., actual flood). Shock happens
        # at some point during the simulation
        self.flood_depth_actual = 0

        # Calculate the actual flood damage given the actual flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_actual = calculate_basic_flood_damage(flood_depth=self.flood_depth_actual)

    def find_social_network(self):
        """Gives each agent a social network based on the size of the radius, this is a social network,
        not topological"""
        self.social_network = self.model.grid.get_neighborhood(self.pos, include_center=False,
                                                               radius=self.radius_network)

    # Function to count friends who can be influencial.
    def count_friends(self):
        """Count the number of neighbors within a given radius (number of edges away). This is social relation and
        not spatial"""
        return len(self.social_network)

    def generate_attribute_values(self):
        """Generate attribute values for the household based on the provided input variables."""
        self.age = max(int(self.attribute_distribution(self.age_distribution_type, self.age_distribution_params)), 18)
        self.wealth = self.attribute_distribution(self.wealth_distribution_type, self.wealth_distribution_range)
        self.has_child = self.attribute_distribution(self.has_child_distribution_type,
                                                     self.has_child_distribution_value)
        self.house_size = self.attribute_distribution(self.house_size_distribution_type,
                                                      self.house_size_distribution_range)
        self.house_type = self.attribute_distribution(self.house_type_distribution_type,
                                                      self.house_type_distribution_range)
        self.education_level = self.attribute_distribution(self.education_level_distribution_type,
                                                           self.education_level_distribution_range)
        self.social_preference = self.attribute_distribution(self.social_preference_distribution_type,
                                                             self.social_preference_distribution_range)

        self.calculate_identity()

    def calculate_identity(self):
        """Calculate the household's identity by determining each attribute's share."""
        # By creating an auxiliary dictionary that defines how each partial share of the agent's attribute
        # contributes to the agent's identity.
        factors = {
            'wealth': lambda x: x * self.wealth_factor,
            'has_child': lambda x: self.has_child_factor if x else 0,  # No child means this partial is 0
            'house_size': lambda x: x * self.house_size_factor,
            'house_type': lambda x: x * self.house_type_factor,
            'education_level': lambda x: x * self.education_level_factor,
            'social_preference': lambda x: x * self.social_preference_factor,
            'age': lambda x: (x / 100) * self.age_factor
        }

        self.identity = sum(factors[attribute](getattr(self, attribute)) for attribute in factors)

    # Ensuring that different types of distributions can be selected for agent attributes as the model input.
    def attribute_distribution(self, dist_type, dist_values):
        """Calculate attribute values based on the specified distribution type and values."""
        if dist_type == 'UI':
            return random.randint(dist_values[0], dist_values[1])
        elif dist_type == 'B':
            return random.random() < dist_values
        elif dist_type == 'U':
            return random.uniform(dist_values[0], dist_values[1])
        elif dist_type == 'N':
            return np.random.normal(dist_values[0], dist_values[1])
        elif dist_type == 'T':
            return random.triangular(dist_values[0], dist_values[1], dist_values[2])
        elif dist_type == 'E':
            return random.expovariate(dist_values)
        else:
            print(f'Improper distribution type {dist_type} for household {self}')

    def bias_change(self):
        """Makes the bounds of which the agent will tolerate influence from agents different them itself.
        For each agent it will determine the dominant opinion within their network and return this. A positive
        number is pro adaption, negative is against."""
        lower_bound_identity = self.identity - self.tolerance if self.identity - self.tolerance > 0 else 0
        higher_bound_identity = self.identity + self.tolerance if self.identity + self.tolerance < 1 else 1

        for agent in self.social_network:
            # check for each social connection whether there is enough similarity between the agents to warrant a
            # change in opinion
            if lower_bound_identity <= self.model.schedule.agents[agent].identity <= higher_bound_identity:
                if self.model.schedule.agents[
                    agent].is_adapted and random.random() < self.probability_positive_bias_change:
                    self.network_bias += self.bias_change_per_tick
                if not self.model.schedule.agents[
                    agent].is_adapted and random.random() < self.probability_negative_bias_change:
                    self.network_bias -= self.bias_change_per_tick

    def step(self):
        """Logic for adaptation based on estimated flood damage and a random chance.
        These conditions are examples and should be refined for real-world applications."""
        self.bias_change()

        # Checking for potential flood adaption using an auxiliary adaption factor, made of the bias in an agent's
        # network, the estimated flood damage and the difference in this estimation compared to an actual flooding
        adaption_factor = self.flood_damage_estimated + self.network_bias + self.actual_flood_impact_on_bias
        if adaption_factor > self.adaption_threshold and not self.is_adapted:
            self.is_adapted = True
        self.age += 0.25  # ageing every tick
