
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_training_evaluation(datetime, Y_e, Y_p, fig_names):
    """TODO: docstring"""
    
    # evaluate metrics
    plt.rcParams.update({"font.size": 16})

    # set mdates formatter
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    # quick plot:
    time = np.arange(0, Y_e.shape[0])
    fig, ax1 = plt.subplots()

    ax1.plot(datetime, Y_e, label="Ground Truth")
    ax1.plot(datetime, Y_p, label="Predictions")
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    plt.xlabel("Date/Time")
    plt.ylabel("Electricity Demand (MWh)")
    plt.legend()
    plt.tight_layout()

    print("Created figure: ", fig_names["timeSeries"])
    plt.savefig(fig_names["timeSeries"])

    # close figure
    plt.close()

    return ax1
