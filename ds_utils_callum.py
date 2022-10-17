
# import core ds libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def basic_eda(df, df_name):
    """
    getting some basic information about each dataframe
    shape of dataframe i.e. number of rows and columns
    total number of rows with null values
    total number of duplicates
    data types of columns
    Args:
        df (dataframe): dataframe containing the data for analysis
        df_name (string): name of the dataframe (a tag you can use to track what's going on)
    """
    print(df_name.upper())
    print()
    print(f"Rows: {df.shape[0]} \t Columns: {df.shape[1]}")
    print()
    
    print(f"Total null rows: {df.isnull().sum().sum()}")
    print(f"Percentage null rows: {round(df.isnull().sum().sum() / df.shape[0] * 100, 2)}%")
    print()
    
    print(f"Total duplicate rows: {df[df.duplicated(keep=False)].shape[0]}")
    print(f"Percentage dupe rows: {round(df[df.duplicated(keep=False)].shape[0] / df.shape[0] * 100, 2)}%")
    print()
    
    print(df.dtypes)
    print("-----\n")
    
def plot_dist_by_dim(data, column, dim):
    """
    Plots the given column against the registration station in the data.
    The function assumes data is a dataframe, column is string (existing column in data),
    and data has a registered column too.
    """
    total_count = data.groupby([column, dim])[column].count()
    pct_contact_type = total_count/data.groupby(column)[column].count()
    pct_contact_type = pct_contact_type.unstack()
    print(pct_contact_type.sort_values([1]))
    
#     plt.rcParams['figure.dpi'] = 360
#     plt.rcParams['figure.figsize'] = (3.2, 2)
    # set the font name for a font family
#     plt.rcParams.update({'font.sans-serif':'Helvetica'})
    sns.set(style="whitegrid")
    pct_contact_type.sort_values([1]).plot(kind="bar", stacked=True, color=['#003F5C', '#FFA600', '#BC5090']) #I think need more colours if more bars being added
    sns.despine(left=True) #de-spine, gets rid of the boarders
    plt.title(f"{column} group distribution", size=10, color='#4F4E4E', fontweight="bold") # Consistent style
    plt.xlabel('')
    plt.xticks(size=8, color='#4F4E4E', rotation=90)
    plt.yticks(size=8, color='#4F4E4E')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()