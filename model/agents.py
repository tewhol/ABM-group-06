# Importing necessary libraries
import random
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy
from statistics import mean
import numpy as np

# Import functions from functions.py
from functions import generate_random_location_within_map_domain, get_flood_depth, calculate_basic_flood_damage, floodplain_multipolygon


# Define the Households agent class
class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.
    """

    def __init__(self, unique_id, model, radius_network, has_child=False):
        super().__init__(unique_id, model)
        self.is_adapted = False  # Initial adaptation status set to False
        # A randomly assigned conviction between 0 (very low) and 1 (very high), which represents fear of flooding
        self.conviction = self.calculate_initial_conviction()
        # An attribute representing the built-up bias in an agents network
        self.bias_network_adaption = 0
        # Attributes related to the size of one's social network, the radius and list of (friends/ friends of friends)
        self.radius_network = radius_network if radius_network is not None else 1
        self.social_network = []
        # Attributes directly related to the households identity
        self.wealth = random.randint(1,4) #1 is low income, 2 below average, 3 above average, 4 rich
        self.house_type = random.randint(1,2) #1 is appartement in a flat, and 2 is vrijstaandhuis
        self.has_child = has_child if has_child is not None else False

        #attribute for the age
        mean_age = 33.7
        std_dev_age = 5
        self.age = max(0, int(np.random.normal(mean_age, std_dev_age)))
        self.social_preference = random.uniform(0,1) # introvert -1 and extrovert is 1
        self.education_level = random.randint(1, 4) # 1 low-level -  4 high level education

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
        self.flood_depth_estimated = get_flood_depth(corresponding_map=model.flood_map, location=self.location, band=model.band_flood_img)
        # handle negative values of flood depth
        if self.flood_depth_estimated < 0:
            self.flood_depth_estimated = 0
        
        # calculate the estimated flood damage given the estimated flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated)

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since
        # there is not flood yet and will update its value when there is a shock (i.e., actual flood). Shock happens
        # at some point during the simulation
        self.flood_depth_actual = 0
        
        #calculate the actual flood damage given the actual flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_actual = calculate_basic_flood_damage(flood_depth=self.flood_depth_actual)
    
    def find_social_network(self):
        """Gives each agent a social network based on the size of the radius, this is a social network,
        not topological"""
        self.social_network = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=self.radius_network)

    # Function to count friends who can be influencial.
    def count_friends(self):
        """Count the number of neighbors within a given radius (number of edges away). This is social relation and
        not spatial"""
        return len(self.social_network)

    def calculate_initial_conviction(self):
        child_factor = 0.2 if self.has_child else 0
        scaled_house_size = (self.house_size - 1) * 0.05
        scaled_education_level = (self.education_level - 1) * 0.05
        scaled_social_preference = self.social_preference * 0.2
        scaled_age = (self.age / 100) * 0.2
        return child_factor + scaled_house_size + scaled_education_level + scaled_social_preference + scaled_age


    def bias_change(self):
        """"Makes the bounds of which the agent will tolerate influence from agents different them itself."""
        tolerance = 0.3
        lower_conviction = self.conviction - tolerance if self.conviction - tolerance > 0 else 0
        higher_conviction = self.conviction + tolerance if self.conviction + tolerance < 1 else 1

        """For each agent it will determine the dominant opinion within their network and return this. A positive 
        number is pro adaption, negative is against."""
        for agent in self.social_network:
            # check for each social connection whether there is enough similarity
            if lower_conviction < self.model.schedule.agents[agent].conviction < higher_conviction:
                if self.model.schedule.agents[agent].is_adapted:
                    self.bias_network_adaption += 0.05
                else:
                    self.bias_network_adaption -= 0.05

    def step(self):
        """Logic for adaptation based on estimated flood damage and a random chance.
        These conditions are examples and should be refined for real-world applications."""
        self.bias_change()
        adaption_factor = self.flood_damage_estimated + self.bias_network_adaption
        if adaption_factor < 0:
            adaption_factor = 0
        if adaption_factor > 2:
            adaption_factor = 2
        if adaption_factor > 0.7 and not self.is_adapted:
            self.is_adapted = True
            print(f'step: {self.model.schedule.steps} : {self.unique_id} adapted with a bias of {self.bias_network_adaption}'
                  f' and a estimated damage of {self.flood_damage_estimated}!')
        self.age += 0.25
        
# Define the Government agent class
class Government(Agent):
    """
    A government agent that currently doesn't perform any actions.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        # The government agent doesn't perform any actions.
        pass

# More agent classes can be added here, e.g. for insurance agents.
