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
    def __init__(self,
                 # Simplified argument for choosing flood map. Can currently be "harvey", "100yr", or "500yr".
                 flood_map_choice='harvey',
                 # Network dynamics
                 network_type=None,  # The social network structure used: "erdos_renyi", "barabasi_albert", "watts_strogatz", or "no_network"
                 num_households=None,  # Number of household agents
                 prob_network_connection=None,  # Likelihood of edge being created between two nodes
                 num_edges=None,  # Number of edges for BA network
                 num_nearest_neighbours=None,  # Number of nearest neighbours for WS social network
                 flood_time_ticks=None,  # Time of flood
                 lower_bound_flood_severity_probability=None,  # Lower bound for flood severity probability
                 higher_bound_flood_severity_probability=None,  # Upper bound for flood severity probability
                 random_seed=None,  # The seed used to generate pseudo random numbers

                 # Agent attributes
                 wealth_factor=None,  # Factor affecting the importance of wealth on the household identity
                 wealth_distribution_type=None,  # Type of distribution for wealth (e.g., 'UI' for Uniform, 'N' for Normal)
                 wealth_distribution_range=None,  # Range for wealth distribution
                 has_child_factor=None,  # Factor affecting the importance of has_child on the household identity
                 has_child_distribution_type=None,  # Type of distribution for has_child (e.g., 'B' for Bernoulli)
                 has_child_distribution_value=None,  # Value for has_child distribution
                 house_size_factor=None,  # Factor affecting the importance of house_size on the household identity
                 house_size_distribution_type=None,  # Type of distribution for house_size (e.g., 'UI' for Uniform)
                 house_size_distribution_range=None,  # Range for house_size distribution
                 house_type_factor=None,  # Factor affecting the importance of house_type on the household identity
                 house_type_distribution_type=None,  # Type of distribution for house_type (e.g., 'UI' for Uniform)
                 house_type_distribution_range=None,  # Range for house_type distribution
                 education_level_factor=None,  # Factor affecting the importance of education_level on the household identity
                 education_level_distribution_type=None,  # Type of distribution for education_level (e.g., 'UI' for Uniform)
                 education_level_distribution_range=None,  # Range for education_level distribution
                 social_preference_factor=None,  # Factor affecting the importance of social_preference on the household identity
                 social_preference_distribution_type=None,  # Type of distribution for social_preference (e.g., 'U' for Uniform)
                 social_preference_distribution_range=None,  # Range for social_preference distribution
                 age_factor=None,  # Factor affecting the importance of age on the household identity
                 age_distribution_type=None,  # Type of distribution for age (e.g., 'N' for Normal)
                 age_distribution_params=None,  # Parameters for age distribution

                 # Agent interaction dynamics
                 household_tolerance=None,  # Tolerance level for each agent
                 bias_change_per_tick=None,  # Bias change per tick when an agent is influenced by its network
                 flood_impact_on_bias_factor=None,  # Factor regarding the actual and expected damage of flooding
                 prob_positive_bias_change=None,  # Probability that agent changes its bias positively
                 prob_negative_bias_change=None,  # Probability that agent changes its bias negatively
                 adaption_threshold=None  # Threshold of bias an agent needs to adapt
                 ):

        # Defining the standard agent attributes variables and their distributions in case none are provided.
        if network_type is None:
            network_type = 'watts_strogatz'
        if num_households is None:
            num_households = 25
        if prob_network_connection is None:
            prob_network_connection = 0.4
        if num_edges is None:
            num_edges = 3
        if num_nearest_neighbours is None:
            num_nearest_neighbours = 5
        if flood_time_ticks is None:
            flood_time_ticks = [5]
        if lower_bound_flood_severity_probability is None:
            lower_bound_flood_severity_probability = 0.5
        if higher_bound_flood_severity_probability is None:
            higher_bound_flood_severity_probability = 1.2
        if random_seed is None:
            random_seed = 1

        # Defining the standard agent attributes variables and their distributions in case none are provided.
        if wealth_distribution_type is None:
            wealth_distribution_type = 'UI'
        if wealth_distribution_range is None:
            wealth_distribution_range = (0, 3)
        if has_child_distribution_type is None:
            has_child_distribution_type = 'B'
        if has_child_distribution_value is None:
            has_child_distribution_value = 0.2
        if house_size_distribution_type is None:
            house_size_distribution_type = 'UI'
        if house_size_distribution_range is None:
            house_size_distribution_range = (0, 2)
        if house_type_distribution_type is None:
            house_type_distribution_type = 'UI'
        if house_type_distribution_range is None:
            house_type_distribution_range = (0, 1)
        if education_level_distribution_type is None:
            education_level_distribution_type = 'UI'
        if education_level_distribution_range is None:
            education_level_distribution_range = (0, 3)
        if social_preference_distribution_type is None:
            social_preference_distribution_type = 'U'
        if social_preference_distribution_range is None:
            social_preference_distribution_range = (-1, 1)
        if age_distribution_type is None:
            age_distribution_type = 'N'
        if age_distribution_params is None:
            age_distribution_params = (33.4, 5)

        # Defining the standard values for agent interaction dynamics in case none are provided.
        if household_tolerance is None:
            household_tolerance = 0.15
        if bias_change_per_tick is None:
            bias_change_per_tick = 0.2
        if flood_impact_on_bias_factor is None:
            flood_impact_on_bias_factor = 1
        if prob_positive_bias_change is None:
            prob_positive_bias_change = 0.5
        if prob_negative_bias_change is None:
            prob_negative_bias_change = 0.1
        if adaption_threshold is None:
            adaption_threshold = 0.7

        # Assigning parameters as attributes
        self.flood_map_choice = flood_map_choice
        self.network_type = network_type
        self.num_households = num_households
        self.prob_network_connection = prob_network_connection
        self.num_edges = num_edges
        self.num_nearest_neighbours = num_nearest_neighbours
        self.flood_time_ticks = flood_time_ticks
        self.lower_bound_flood_severity_probability = lower_bound_flood_severity_probability
        self.higher_bound_flood_severity_probability = higher_bound_flood_severity_probability
        self.random_seed = random_seed

        # Assigning attributes from agent attributes
        self.wealth_factor = wealth_factor
        self.wealth_distribution_type = wealth_distribution_type
        self.wealth_distribution_range = wealth_distribution_range
        self.has_child_factor = has_child_factor
        self.has_child_distribution_type = has_child_distribution_type
        self.has_child_distribution_value = has_child_distribution_value
        self.house_size_factor = house_size_factor
        self.house_size_distribution_type = house_size_distribution_type
        self.house_size_distribution_range = house_size_distribution_range
        self.house_type_factor = house_type_factor
        self.house_type_distribution_type = house_type_distribution_type
        self.house_type_distribution_range = house_type_distribution_range
        self.education_level_factor = education_level_factor
        self.education_level_distribution_type = education_level_distribution_type
        self.education_level_distribution_range = education_level_distribution_range
        self.social_preference_factor = social_preference_factor
        self.social_preference_distribution_type = social_preference_distribution_type
        self.social_preference_distribution_range = social_preference_distribution_range
        self.age_factor = age_factor
        self.age_distribution_type = age_distribution_type
        self.age_distribution_params = age_distribution_params

        # Assigning attributes from agent interaction dynamics
        self.household_tolerance = household_tolerance
        self.bias_change_per_tick = bias_change_per_tick
        self.flood_impact_on_bias_factor = flood_impact_on_bias_factor
        self.prob_positive_bias_change = prob_positive_bias_change
        self.prob_negative_bias_change = prob_negative_bias_change
        self.adaption_threshold = adaption_threshold

        super().__init__(seed=self.seed)

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
            household = Households(unique_id=i, model=self, radius_network=1, agent_interaction_dictionary=self.agent_interaction_dictionary)
            self.schedule.add(household)
            self.grid.place_agent(agent=household, node_id=node)
            household.generate_attribute_values(attribute_dictionary)

        # now that the network is established, let's give each agent their connections in the social network
        for agent in self.schedule.agents:
            agent.find_social_network()

        # Data collection setup to collect data
        model_metrics = {
            "total_adapted_households": self.total_adapted_households,
            "bias_change_all_agents": self.total_bias_change_all_households,
            # ... other reporters ...
        }

        agent_metrics = {
            "FloodDepthEstimated": "flood_depth_estimated",
            "FloodDamageEstimated": "flood_damage_estimated",
            "FloodDepthActual": "flood_depth_actual",
            "FloodDamageActual": "flood_damage_actual",
            "IsAdapted": "is_adapted",
            "Identity": "identity",
            "FriendsCount": lambda a: a.count_friends(),
            "location": "location",
            "HasChild": "has_child",
            "EndBias": "network_bias"
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
        adapted_count = sum([1 for agent in self.schedule.agents if agent.is_adapted])
        return adapted_count

    def total_bias_change_all_households(self):
        """Return the total amount of bias change of households"""
        total_bias_change_all_households = sum([abs(agent.network_bias) for agent in self.schedule.agents])
        return total_bias_change_all_households

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
        ax.legend(by_label.values(), by_label.keys(), title="Red: not adapted, Blue: adapted")

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
        if self.schedule.steps in self.flood_time_tick:
            for agent in self.schedule.agents:
                # Calculate actual flood depth as a random number between the defined severity times the estimated
                # flood depth
                a, b = self.flood_severity_probability  # splitting tuple in two bounds
                agent.flood_depth_actual = random.uniform(a, b) * agent.flood_depth_estimated
                if not agent.is_adapted:
                    # calculate the actual flood damage given the actual flood depth
                    agent.flood_damage_actual = calculate_basic_flood_damage(agent.flood_depth_actual)
                    # adapt agent bias regarding flood adaption depending on the difference of real and estimated damage
                    agent.actual_flood_impact_on_bias += (agent.flood_damage_actual - agent.flood_damage_estimated) * agent.flood_impact_on_bias_factor


        # Collect data and advance the model by one step
        self.datacollector.collect(self)
        self.schedule.step()
