from model import AdaptationModel
import mesa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import batch_run_config as config


def bias_change_over_time(results):
    results_df = pd.DataFrame(results)
    results_filtered = results_df
    results_filtered[["Step", "Bias"]].reset_index(drop=True).head()
    # Create a scatter plot
    g = sns.lineplot(data=results_filtered,
                     x="Step",
                     y="Bias",)
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
                        y="total_adapted_households",)
    g.set(
        xlabel="Total bias change in network",
        ylabel="total adapted households",
        title="Bias change vs flood adaption",
    )
    plt.show()


def total_number_of_damage(results):
    results_df = pd.DataFrame(results)
    results_filtered = results_df[(results_df.Step == 80)]
    results_filtered[["total_adapted_households", "total_flood_damage", "bias_change_all_agents"]].reset_index(
        drop=True).head()
    # Create a scatter plot
    g = sns.scatterplot(data=results_filtered,
                        x="total_adapted_households",
                        y="total_flood_damage",)
    g.set(
        xlabel="total adapted households",
        ylabel="total flood damage",
        title="adaption vs flood damage",
    )
    plt.show()


# Other input variables for the mesa batch run function
iterations = 50  # number of iterations for each parameter combination
max_steps = 80  # max steps of each model run/ iteration
number_processes = None  # how many processors are used
data_collection_period = 1  # number of steps after which data is collected by the model and agent data collectors
display_progress = True  # To display the progress on the batch runs

if __name__ == "__main__":
    results = mesa.batch_run(
        AdaptationModel,
        parameters=config.parameters,
        iterations=iterations,
        max_steps=max_steps,
        number_processes=number_processes,
        data_collection_period=data_collection_period,
        display_progress=display_progress,
    )





