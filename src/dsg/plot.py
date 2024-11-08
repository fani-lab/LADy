"""Chart creation utilities for tasks in the DSG module."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_eval_metrics(model_names, precision, recall, f1_scores, exact_match):
    """Plots evaluation metrics for the given models."""
    x = np.arange(len(model_names))
    bar_width = 0.2

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars for each metric
    bars1 = ax.bar(x - bar_width, precision, width=bar_width, label='Precision', color='blue')
    bars2 = ax.bar(x, recall, width=bar_width, label='Recall', color='red')
    bars3 = ax.bar(x + bar_width, f1_scores, width=bar_width, label='F1 Score', color='purple')
    bars4 = ax.bar(x + 2 * bar_width, exact_match, width=bar_width, label='Exact Matches', color='green')

    # Adding labels, title, and custom x-axis tick labels
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Accuracy, F1 Score and Exact Matches for Different Models')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig('./dsg/data/model_comparison.png')
