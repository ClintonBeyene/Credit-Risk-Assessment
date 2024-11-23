import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scorecardpy as sc


def numerical_distribution(df, numerical_columns, style='whitegrid', nrows=2, ncols=2, figsize=(10, 8), bins=30):
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
    
def plot_categorical_distribution(df):
    """
    Function to plot the distribution of categorical features in a DataFrame and 
    display the count values on top of each bar.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the dataset to be analyzed.
    """
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    cols_with_few_categories = [col for col in categorical_cols if df[col].nunique() <= 10]

    # Set up the grid for subplots
    num_cols = len(cols_with_few_categories)
    num_rows = (num_cols + 1) // 2  # Automatically determine the grid size
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(cols_with_few_categories):
        ax = sns.countplot(data=df, x=col, ax=axes[i], hue=col, legend=False, palette="Set2")
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Frequency')

        # Add count labels to the top of the bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='baseline', fontsize=12, color='black', 
                        xytext=(0, 5), textcoords='offset points')
    
    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def correlation_analysis(df, numeric_columns):
    # Task 5: Correlation Analysis
    corr_matrix = df[numeric_columns].corr()
    f, ax = plt.subplots(figsize=(10,8))
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