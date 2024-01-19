# Importing necessary libraries
import networkx as nx
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import geopandas as gpd
import rasterio as rs
import matplotlib.pyplot as plt
import random

# Import the agent class(es) from agents.py
from agents import Households

# Import functions from functions.py
from functions import get_flood_map_data, calculate_basic_flood_damage
from functions import map_domain_gdf, floodplain_gdf


# Define the AdaptationModel class
class AdaptationModel(Model):
    """
    The main model running the simulation. It sets up the network of household agents,
    simulates their behavior, and collects data. The network type can be adjusted based on study requirements.
    """

    def __init__(self,
                 # Simplified argument for choosing flood map. Can currently be "harvey", "100yr", or "500yr".
                 flood_map_choice='harvey',
                 # A dictionary that provides all the necessary parameters to define the network of agents that is
                 # going to be created and some of the dynamic variables that the model needs (like the tolerance and
                 # the flood time)
                 network_dynamics_dictionary=None,
                 # A dictionary with entries that respectively defines the impact factor and the general distribution
                 # for agent attributes. The latter is in a tuple with the letter marking the type and the numbers
                 # the parameters for the distribution.
                 attribute_dictionary=None
                 ):

        # Defining the standard agent attributes variables and their distributions in case none are provided.
        if attribute_dictionary is None:
            attribute_dictionary = {
                "wealth": [0.2, 'UI', (0, 3)],
                "has_child": [0.2, 'B', (0.2)],
                "house_size": [0.05, 'UI', (0, 2)],
                "house_type": [0.05, 'UI', (0, 1)],
                "education_level": [0.05, 'UI', (0, 3)],
                "social_preference": [0.2, 'U', (-1, 1)],
                "age": [0.2, 'N', (33.4, 5)]
            }

        # Defining the standard model network and network interaction/social dynamics in case none are provided.
        if network_dynamics_dictionary is None:
            network_dynamics_dictionary = {
                # The social network structure that is used. Can currently be
                # "erdos_renyi", "barabasi_albert", "watts_strogatz", or "no_network"
                "network": 'watts_strogatz',
                "number_of_households": 25,  # number of household agents
                "probability_of_network_connection": 0.4,  # likeliness of edge being created between two nodes
                "number_of_edges": 3,  # number of edges for BA network
                "number_of_nearest_neighbours": 5,  # number of nearest neighbours for WS social network
                "household_tolerance": 0.1,  # the tolerance level for each agent,
                "flood_time_tick": 5,  # time of flood
                "seed": 1  # The seed used to generate pseudo random numbers
            }

        self.flood_time_tick = network_dynamics_dictionary['flood_time_tick']  # The exact tick on which a flood occurs
        self.number_of_households = network_dynamics_dictionary['number_of_households']  # Total number of household agents
        self.seed = network_dynamics_dictionary['seed'] # The model seed

        super().__init__(seed=self.seed)

        # network
        self.network = network_dynamics_dictionary['network']  # Type of network to be created
        self.probability_of_network_connection = network_dynamics_dictionary['probability_of_network_connection']
        self.number_of_edges = network_dynamics_dictionary['number_of_edges']
        self.number_of_nearest_neighbours = network_dynamics_dictionary['number_of_nearest_neighbours']
        # Determines the manner of similarity needed for households to influence each other
        self.household_tolerance = network_dynamics_dictionary['household_tolerance']

        # generating the graph according to the network used and the network parameters specified
        self.G = self.initialize_network()
        # create grid out of network graph
        self.grid = NetworkGrid(self.G)

        # Initialize maps
        self.initialize_maps(flood_map_choice)

        # set schedule for agents, and defining agent attributes
        self.schedule = RandomActivation(self)  # Schedule for activating agents

        # create households through initiating a household on each node of the network graph and create this
        # household's attribute values using the provided attribute dictionary.
        for i, node in enumerate(self.G.nodes()):
            household = Households(unique_id=i, model=self, radius_network=1, bias_change_per_tick=0.2,
                                   tolerance=self.household_tolerance)
            self.schedule.add(household)
            self.grid.place_agent(agent=household, node_id=node)
            household.generate_attribute_values(attribute_dictionary)

        # now that the network is established, let's give each agent their connections in the social network
        for agent in self.schedule.agents:
            agent.find_social_network()

        # Data collection setup to collect data
        model_metrics = {
            "total_adapted_households": self.total_adapted_households,
            # ... other reporters ...
        }

        agent_metrics = {
            "FloodDepthEstimated": "flood_depth_estimated",
            "FloodDamageEstimated": "flood_damage_estimated",
            "FloodDepthActual": "flood_depth_actual",
            "FloodDamageActual": "flood_damage_actual",
            "IsAdapted": "is_adapted",
            "Conviction": "conviction",
            "FriendsCount": lambda a: a.count_friends(),
            "location": "location",
            "HasChild": "has_child"
            # ... other reporters ...
        }
        # set up the data collector
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)

    def initialize_network(self):
        """
        Initialize and return the social network graph based on the provided network type using pattern matching.
        """
        if self.network == 'erdos_renyi':
            return nx.erdos_renyi_graph(n=self.number_of_households,
                                        p=self.number_of_nearest_neighbours / self.number_of_households,
                                        seed=self.seed)
        elif self.network == 'barabasi_albert':
            return nx.barabasi_albert_graph(n=self.number_of_households,
                                            m=self.number_of_edges,
                                            seed=self.seed)
        elif self.network == 'watts_strogatz':
            return nx.watts_strogatz_graph(n=self.number_of_households,
                                           k=self.number_of_nearest_neighbours,
                                           p=self.probability_of_network_connection,
                                           seed=self.seed)
        elif self.network == 'no_network':
            G = nx.Graph()
            G.add_nodes_from(range(self.number_of_households))
            return G
        else:
            raise ValueError(f"Unknown network type: '{self.network}'. "
                             f"Currently implemented network types are: "
                             f"'erdos_renyi', 'barabasi_albert', 'watts_strogatz', and 'no_network'")

    def initialize_maps(self, flood_map_choice):
        """
        Initialize and set up the flood map related data based on the provided flood map choice.
        """
        # Define paths to flood maps
        flood_map_paths = {
            'harvey': r'../input_data/floodmaps/Harvey_depth_meters.tif',
            '100yr': r'../input_data/floodmaps/100yr_storm_depth_meters.tif',
            '500yr': r'../input_data/floodmaps/500yr_storm_depth_meters.tif'  # Example path for 500yr flood map
        }

        # Throw a ValueError if the flood map choice is not in the dictionary
        if flood_map_choice not in flood_map_paths.keys():
            raise ValueError(f"Unknown flood map choice: '{flood_map_choice}'. "
                             f"Currently implemented choices are: {list(flood_map_paths.keys())}")

        # Choose the appropriate flood map based on the input choice
        flood_map_path = flood_map_paths[flood_map_choice]

        # Loading and setting up the flood map
        self.flood_map = rs.open(flood_map_path)
        self.band_flood_img, self.bound_left, self.bound_right, self.bound_top, self.bound_bottom = get_flood_map_data(
            self.flood_map)

    def total_adapted_households(self):
        """Return the total number of households that have adapted."""
        # BE CAREFUL THAT YOU MAY HAVE DIFFERENT AGENT TYPES SO YOU NEED TO FIRST CHECK IF THE AGENT IS ACTUALLY A HOUSEHOLD AGENT USING "ISINSTANCE"
        adapted_count = sum([1 for agent in self.schedule.agents if isinstance(agent, Households) and agent.is_adapted])
        return adapted_count

    def plot_model_domain_with_agents(self):
        fig, ax = plt.subplots()
        # Plot the model domain
        map_domain_gdf.plot(ax=ax, color='lightgrey')
        # Plot the floodplain
        floodplain_gdf.plot(ax=ax, color='lightblue', edgecolor='k', alpha=0.5)

        # Collect agent locations and statuses
        for agent in self.schedule.agents:
            color = 'blue' if agent.is_adapted else 'red'
            ax.scatter(agent.location.x, agent.location.y, color=color, s=10,
                       label=color.capitalize() if not ax.collections else "")
            ax.annotate(str(agent.unique_id), (agent.location.x, agent.location.y), textcoords="offset points",
                        xytext=(0, 1), ha='center', fontsize=9)
        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Red: not adapted, no concern, Blue: adapted, "
                                                            "concerned\n Purple: adapted, no concern, Yellow: not "
                                                            "adapted, concerned")

        # Customize plot with titles and labels
        plt.title(f'Model Domain with Agents at Step {self.schedule.steps}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def step(self):
        """
        introducing a shock: 
        at time step 5, there will be a global flooding.
        This will result in actual flood depth. Here, we assume it is a random number
        between 0.5 and 1.2 of the estimated flood depth. In your model, you can replace this
        with a more sound procedure (e.g., you can devide the floop map into zones and 
        assume local flooding instead of global flooding). The actual flood depth can be 
        estimated differently
        """
        if self.schedule.steps == self.flood_time_tick:
            for agent in self.schedule.agents:
                # Calculate actual flood depth as a random number between 0.5 and 1.2 times the estimated flood depth
                agent.flood_depth_actual = random.uniform(0.5, 1.2) * agent.flood_depth_estimated
                # calculate the actual flood damage given the actual flood depth
                agent.flood_damage_actual = calculate_basic_flood_damage(agent.flood_depth_actual)
                agent.actual_flood_impact_on_bias = agent.flood_damage_actual - agent.flood_damage_estimated

        # Collect data and advance the model by one step
        self.datacollector.collect(self)
        self.schedule.step()
