import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scorecardpy as sc


def numerical_distribution(df, numerical_columns, style='whitegrid', nrows=4, ncols=3, figsize=(15,5), bins=30):
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
    

def correlation_analysis(df, numeric_columns):
    # Task 5: Correlation Analysis
    corr_matrix = df[numeric_columns].corr()
    f, ax = plt.subplots(figsize=(20,10))
    sns.heatmap(corr_matrix, square=True, annot=True, linewidth=0.8, cmap='RdBu')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.title('Correlation Matrix')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.show()


def outlier_detection(df, numeric_columns, style='whitegrid', nrows=4, ncols=3, figsize=(15,8), bins=30):
    # Set the style 
    sns.set_style(style)
    palette = sns.color_palette("Set2")
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    # Create a box plot for each numeric column
    for i, column in enumerate(numeric_columns):  # Use enumerate to get both index and value
        sns.boxplot(ax=axes[i], data=df[column], color=palette[i % len(palette)])  # Use the axes from the subplot
        axes[i].set_title(f'Box Plot for {column}')

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and show plot 
    plt.tight_layout()
    plt.show()

def plot_woe_binning(bins, feature):
    """
    Plot WoE binning results for a specific feature.
    """
    plt.figure(figsize=(10, 6))
    sc.woebin_plot(bins[feature])
    plt.title(f'WoE Binning Plot for {feature}')
    plt.tight_layout()
    plt.show()