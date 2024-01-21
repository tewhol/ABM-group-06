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

    def __init__(self, unique_id, model, radius_network, agent_interaction_dictionary):
        super().__init__(unique_id, model)
        self.is_adapted = False  # Initial adaptation status set to False
        self.actual_flood_impact_on_bias = 0  # Initial variable about how a real flood would impact an agent's bias

        # An attribute representing the built-up bias in an agents network
        self.general_bias_in_network = 0
        self.tolerance = agent_interaction_dictionary['household_tolerance']
        self.bias_change_per_tick = agent_interaction_dictionary['bias_change_per_tick']
        self.adaption_threshold = agent_interaction_dictionary['adaption_threshold']
        self.probability_positive_bias_change = agent_interaction_dictionary['probability_positive_bias_change']
        self.probability_negative_bias_change = agent_interaction_dictionary['probability_negative_bias_change']

        # Attributes related to the size of one's social network, the radius and list of (friends/ friends of friends)
        self.radius_network = radius_network if radius_network is not None else 1
        self.social_network = []

        # Attributes directly related to the households identity
        self.wealth = 0  # Potential values, integer: 0 is low-income, 1 below average, 2 above average, 3 rich
        self.house_type = 0  # Potential values: 0 is apartment in a block, and 1 is freestanding house
        self.house_size = 0  # Potential values: simulating different type of sizes each house has
        self.has_child = False  # Potential values, boolean: True of false about having child
        self.age = 0  # Potential values: at least 18
        self.social_preference = 0  # Potential values: between introvert -1 and extrovert is 1
        self.education_level = 0  # Potential values, integer: 0 low-level - 3 high level education
        self.identity = 0  # Potential values: between 0 and 1

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

    def generate_attribute_values(self, attribute_dictionary):
        """Generate attribute values for the household based on the provided attribute dictionary."""
        for attribute, values in attribute_dictionary.items():
            # Ensure age is at least 18 and an integer
            if attribute == 'age':
                age_value = max(int(self.attribute_distribution(values[1], values[2])), 18)
                setattr(self, attribute, age_value)
            else:
                setattr(self, attribute, self.attribute_distribution(values[1], values[2]))

        self.calculate_identity(attribute_dictionary)

    def calculate_identity(self, attribute_dictionary):
        """Calculate the household's identity by determining each attribute's share."""
        # By creating an auxiliary dictionary that defines how each partial share of the agent's attribute
        # contributes to the agent's identity.
        factors = {
            'wealth': lambda x: x * attribute_dictionary['wealth'][0],
            'has_child': lambda x: attribute_dictionary['has_child'][0] if x else 0,  # No child means this partial is 0
            'house_size': lambda x: x * attribute_dictionary['house_size'][0],
            'house_type': lambda x: x * attribute_dictionary['house_type'][0],
            'education_level': lambda x: x * attribute_dictionary['education_level'][0],
            'social_preference': lambda x: x * attribute_dictionary['social_preference'][0],
            'age': lambda x: (x / 100) * attribute_dictionary['age'][0]
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
        lower_identity = self.identity - self.tolerance if self.identity - self.tolerance > 0 else 0
        higher_identity = self.identity + self.tolerance if self.identity + self.tolerance < 1 else 1

        for agent in self.social_network:
            # check for each social connection whether there is enough similarity between the agents to warrant a
            # change in opinion
            if lower_identity <= self.model.schedule.agents[agent].identity <= higher_identity:
                if self.model.schedule.agents[agent].is_adapted and random.random() < self.probability_positive_bias_change:
                        self.general_bias_in_network += self.bias_change_per_tick
                if not self.model.schedule.agents[agent].is_adapted and random.random() < self.probability_negative_bias_change:
                    self.general_bias_in_network -= self.bias_change_per_tick

    def step(self):
        """Logic for adaptation based on estimated flood damage and a random chance.
        These conditions are examples and should be refined for real-world applications."""
        self.bias_change()

        # Checking for potential flood adaption using an auxiliary adaption factor, made of the bias in an agent's
        # network, the estimated flood damage and the difference in this estimation compared to an actual flooding
        adaption_factor = self.flood_damage_estimated + self.general_bias_in_network + self.actual_flood_impact_on_bias
        if adaption_factor > self.adaption_threshold and not self.is_adapted:
            self.is_adapted = True
            print(
                f'step: {self.model.schedule.steps} : {self.unique_id} adapted with a bias of {self.general_bias_in_network}'
                f' and a estimated damage of {self.flood_damage_estimated}!')
        self.age += 0.25  # ageing every tick
