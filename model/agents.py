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

    def __init__(self, unique_id, model, radius_network, tolerance, bias_change_per_tick):
        super().__init__(unique_id, model)
        self.is_adapted = False  # Initial adaptation status set to False
        self.actual_flood_impact_on_bias = 0  # Initial variable about how a real flood would impact an agent's bias

        # An attribute representing the built-up bias in an agents network
        self.general_bias_in_network = 0
        self.bias_change_per_tick = bias_change_per_tick
        self.tolerance = tolerance

        # Attributes related to the size of one's social network, the radius and list of (friends/ friends of friends)
        self.radius_network = radius_network if radius_network is not None else 1
        self.social_network = []

        # Attributes directly related to the households identity
        self.wealth = 0  # Potential values, integer: 1 is low-income, 2 below average, 3 above average, 4 rich
        self.house_type = 0  # Potential values: 1 is apartment in a block, and 2 is freestanding house
        self.house_size = 0  # Potential values: simulating different type of sizes each house has
        self.has_child = False  # Potential values, boolean: True of false about having child
        self.age = 0  # Potential values: at least 18
        self.social_preference = 0  # Potential values: between introvert -1 and extrovert is 1
        self.education_level = 0  # Potential values, integer: 1 low-level -  4 high level education
        self.conviction = 0  # Potential values: between 0 and 1


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

    # This function calculates each household's attributes using the auxiliary distribution function and the
    # attribute dictionary, provided by the model. This dictionary dictates how agent attributes should be
    # constructed across the entire model population. Based on all generated attribute values calculates the agent's
    # conviction.
    def generate_attribute_values(self, attribute_dictionary):
        """It takes each element from the model's attribute dictionary, checks the corresponding agent attribute and
        then uses the auxiliary distribution function to assign a value to the attribute value of this particular
        agent. Furhtermore, it calculates the attribute's share in determining the agent's conviction. This
        share/partial impact factor is also pre-defined in the attribute dictionary"""
        for attribute in attribute_dictionary:
            impact_on_conviction = attribute_dictionary[attribute][0]
            dist_type = attribute_dictionary[attribute][1]
            dist_values = attribute_dictionary[attribute][2]
            if attribute == 'wealth':
                self.wealth = self.attribute_distribution(dist_type, dist_values)
                wealth_factor = (self.wealth - 1) * impact_on_conviction
            elif attribute == 'child':
                self.has_child = self.attribute_distribution(dist_type, dist_values)
                child_factor = impact_on_conviction if self.has_child else 0
            elif attribute == 'house_size':
                self.house_size = self.attribute_distribution(dist_type, dist_values)
                scaled_house_size = (self.house_size - 1) * impact_on_conviction
            elif attribute == 'education':
                self.education_level = self.attribute_distribution(dist_type, dist_values)
                scaled_education_level = (self.education_level - 1) * impact_on_conviction
            elif attribute == 'social':
                self.social_preference = self.attribute_distribution(dist_type, dist_values)
                scaled_social_preference = self.social_preference * impact_on_conviction
            elif attribute == 'age':
                self.age = max(int(self.attribute_distribution(dist_type, dist_values), 18))  # The minimum age is 18
                scaled_age = (self.age / 100) * impact_on_conviction
            else:
                print(f'{attribute} does not exist in model!')
        self.conviction = wealth_factor + child_factor + scaled_house_size + scaled_education_level + scaled_social_preference + scaled_age

    # Ensuring that different types of distributions can be selected for agent attributes as the model input.
    def attribute_distribution(self, dist_type, dist_values):
        """The function takes the attribute dictionary as input. This dictionary is created on the model level and
        defines the agent attributes and their impacts on the model level. In this function the actual result value
        of these predefined distribution can be calculated"""
        if dist_type == 'UI':
            return random.randint(dist_values[0],
                                  dist_values[1])  # the lower and upper bound, returns integer
        elif dist_type == 'B':
            return random.random() < dist_values
        elif dist_type == 'U':
            return random.uniform(dist_values[0],
                                  dist_values[1])  # The lower and upper bound
        elif dist_type == 'N':
            return np.random.normal(dist_values[0],
                                    dist_values[1])  # The mean and deviation
        elif dist_type == 'T':
            return random.triangular(dist_values[0],
                                     dist_values[1],
                                     dist_values[2])  # The lower and upper bound and the mode
        elif dist_type == 'E':
            return random.expovariate(dist_values)  # The lambda for an exponential distribution
        else:
            print(f'{dist_type} of household {self} has improper distribution values!')

    def bias_change(self):
        """Makes the bounds of which the agent will tolerate influence from agents different them itself.
        For each agent it will determine the dominant opinion within their network and return this. A positive
        number is pro adaption, negative is against."""
        lower_conviction = self.conviction - self.tolerance if self.conviction - self.tolerance > 0 else 0
        higher_conviction = self.conviction + self.tolerance if self.conviction + self.tolerance < 1 else 1

        for agent in self.social_network:
            # check for each social connection whether there is enough similarity between the agents to warrant a
            # change in opinion
            if lower_conviction < self.model.schedule.agents[
                agent].conviction < higher_conviction and random.random() < 0.5:
                if self.model.schedule.agents[agent].is_adapted:
                    self.general_bias_in_network += self.bias_change_per_tick
                if not self.model.schedule.agents[agent].is_adapted and random.random() < 0.25:
                    self.general_bias_in_network -= self.bias_change_per_tick

    def step(self):
        """Logic for adaptation based on estimated flood damage and a random chance.
        These conditions are examples and should be refined for real-world applications."""
        self.bias_change()

        # Checking for potential flood adaption using an auxiliary adaption factor, made of the bias in an agent's
        # network, the estimated flood damage and the difference in this estimation compared to an actual flooding
        adaption_factor = self.flood_damage_estimated + self.general_bias_in_network
        # The adaption factor gets capped between 0 and 2
        if adaption_factor < 0:
            adaption_factor = 0
        if adaption_factor > 2:
            adaption_factor = 2
        if adaption_factor > 0.7 and not self.is_adapted:
            self.is_adapted = True
            print(
                f'step: {self.model.schedule.steps} : {self.unique_id} adapted with a bias of {self.general_bias_in_network}'
                f' and a estimated damage of {self.flood_damage_estimated}!')
        self.age += 0.25  # ageing every tick
