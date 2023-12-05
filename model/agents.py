# Importing necessary libraries
import random
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy

# Import functions from functions.py
from functions import generate_random_location_within_map_domain, get_flood_depth, calculate_basic_flood_damage, floodplain_multipolygon


# Define the Households agent class
class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.
    """

    def __init__(self, unique_id, model, radius_network):
        super().__init__(unique_id, model)
        self.is_adapted = False  # Initial adaptation status set to False
        # A randomly assigned conviction between 0 (very low) and 1 (very high), which represents fear of flooding
        self.conviction = random.uniform(0, 1)
        self.adaption_factor = 0
        self.radius_network = radius_network if radius_network is not None else 1
        self.social_network = []

        # getting flood map values
        # Get a random location on the map
        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)

        # Check whether the location is within floodplain
        self.in_floodplain = False
        if contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y):
            self.in_floodplain = True

        # Get the estimated flood depth at those coordinates. 
        # the estimated flood depth is calculated based on the flood map (i.e., past data) so this is not the actual flood depth
        # Flood depth can be negative if the location is at a high elevation
        self.flood_depth_estimated = get_flood_depth(corresponding_map=model.flood_map, location=self.location, band=model.band_flood_img)
        # handle negative values of flood depth
        if self.flood_depth_estimated < 0:
            self.flood_depth_estimated = 0
        
        # calculate the estimated flood damage given the estimated flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated)

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since there is not flood yet
        # and will update its value when there is a shock (i.e., actual flood). Shock happens at some point during the simulation
        self.flood_depth_actual = 0
        
        #calculate the actual flood damage given the actual flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_actual = calculate_basic_flood_damage(flood_depth=self.flood_depth_actual)
    
    def find_social_network(self):
        """Gives each agent a social network based on the size of the radius, this is a social network, not topological"""
        self.social_network = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=self.radius_network)

    # Function to count friends who can be influencial.
    def count_friends(self):
        """Count the number of neighbors within a given radius (number of edges away). This is social relation and not spatial"""
        return len(self.social_network)

    def bias_change(self):
        # Makes the bounds of which the agent will tolerate influence from agents different them itself.
        tolerance = 0.05
        lower_conviction = self.conviction - tolerance if self.conviction + tolerance > 0 else 0
        higher_conviction = self.conviction + tolerance if self.conviction + tolerance < 1 else 1

        for agent in self.social_network:
            # check for each social connection whether there is enough similarity
            if lower_conviction < self.model.schedule.agents[agent].conviction < higher_conviction:
                print(f'agent {self.unique_id} and {self.model.schedule.agents[agent].unique_id} are similar enough!')

    def step(self):
        # Logic for adaptation based on estimated flood damage and a random chance.
        # These conditions are examples and should be refined for real-world applications.
        if self.flood_damage_estimated > 0.15 and random.random() < 0.2:
            self.is_adapted = True  # Agent adapts to flooding
        self.bias_change()
        
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
