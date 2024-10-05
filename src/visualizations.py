import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def numerical_distribution(df, numerical_columns, style='whitegrid', nrows=2, ncols=3, figsize=(15,5), bins=30):
    # Set the style
    sns.set_style(style)

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    # Plot histograms
    for i, column in enumerate(numerical_columns):
        sns.histplot(df[column], kde=True, ax=axes[i], bins=bins)
        axes[i].set_title(f"Distribution of {column}")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()