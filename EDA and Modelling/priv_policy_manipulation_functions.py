
# import core ds libraries
import numpy as np
import pandas as pd
from pandas import json_normalize
import yaml

import matplotlib.pyplot as plt
import seaborn as sns


def load_all_policies():
    
    # Load the first policy into a dataframe
    with open("APP_350_v1_1/annotations/policy_1.yml", "r") as stream:
        try:
            all_policies_df = (json_normalize(yaml.safe_load(stream)))
        except yaml.YAMLError as exc:
            print(exc)
        
    # Get the locations of all policy files
    full_policy_list = [f"APP_350_v1_1/annotations/policy_{num}.yml" for num in range(2,351)]
    
    #Loop through all the policy file addresses, normalise it and add it to the bottom of all_policies_df
    for document in full_policy_list:
        with open(document, "r") as stream:
            try:
                current_df_policy_info = (json_normalize(yaml.safe_load(stream)))
                all_policies_df = pd.concat([all_policies_df, current_df_policy_info], axis=0)
            except yaml.YAMLError as exc:
                print(exc)
    
    all_policies_df.reset_index(drop=True, inplace=True)
    
    return all_policies_df



###

def add_metadata_to_policy_df(input_policies_df):
    
    all_policies_df = input_policies_df.copy()
    
    # New columns to populate
    all_policies_df["num_segments"] = 0
    all_policies_df["num_annotated_segs"] = 0
    all_policies_df["total_characters"] = 0
    
    # Loop through each policy
    
    for i in range(len(all_policies_df["segments"])):
        segment = all_policies_df.loc[i, "segments"] # grab the policy
        
        policy_segment_df = json_normalize(segment) # apply json_normalize
        
        policy_segment_df.set_index('segment_id', inplace=True)
        
        all_policies_df.loc[i, "num_segments"] = policy_segment_df["segment_text"].count() # count the sentences and add to the main df

        policy_segment_df.loc[ policy_segment_df["annotations"].str.len() == 0 , "annotations"] = None # clean the annotations column
        
        all_policies_df.loc[i, "num_annotated_segs"] = policy_segment_df["annotations"].count() # count the annotated segments

        all_policies_df.loc[i, "total_characters"] = policy_segment_df["segment_text"].str.len().sum() # count the characters
        
    return all_policies_df



###

def generate_segment_df(all_policies_df):
    
    # First create this for a single policy. Then loop through all the policies to apply the same manipulation.
    initial_segment = all_policies_df.loc[0,"segments"]
    initial_segment_df = json_normalize(initial_segment) # normalise it
    initial_segment_df.set_index('segment_id', inplace=True)
    initial_segment_df["source_policy_number"] = all_policies_df.loc[0,"policy_id"]

    segment_df = initial_segment_df
    
    for i in all_policies_df.index:
        
        this_segment = all_policies_df.loc[i,"segments"]
        this_segment_df = json_normalize(this_segment) # normalise it
        this_segment_df.set_index('segment_id', inplace=True)
        this_segment_df["source_policy_number"] = all_policies_df.loc[i,"policy_id"]

        segment_df = pd.concat( [segment_df, this_segment_df], axis=0 ) 
    
    segment_df['policy_segment_id'] = segment_df.index
    segment_df.reset_index(drop=True, inplace=True)
    segment_df.index.names = ['segment_id']
    segment_df = segment_df[['source_policy_number', 'policy_segment_id', 'segment_text', 'annotations', 'sentences']]
    
    return segment_df



###

def get_list_of_practice_groups():
    """
    Requires the folder "APP_350_v1_1" to be in the same directory as this file.
    
    Returns a list where each element is a list of practices, representing a category of practices
    e.g. ['Contact_E_Mail_Address_1stParty', 'Contact_E_Mail_Address_3rdParty']
    There are 29 categories.    
    
    A full list of individual practices can then be optained with a list comprehension: 
    
    list_of_practices = [practice for practice_group in list_of_practice_groups for practice in practice_group]
    There are 58 individual practices.
   
    """
    with open("APP_350_v1_1/features.yml", "r") as stream:
        try:
            features_yml = (json_normalize(yaml.safe_load(stream)))
        except yaml.YAMLError as exc:
            print(exc)
    
    data_types = json_normalize(features_yml['data_types'])
    
    list_of_practice_groups = []
    practice_groups = range(len(data_types.columns))
    
    for i in practice_groups:
        practices_and_features = json_normalize(data_types[i])

        list_of_practice_groups.extend(practices_and_features['practices'])
    
    print(f"{len(list_of_practice_groups)} different groups of practices returned, containing {len([practice for practice_group in list_of_practice_groups for practice in practice_group])} individual practices.")
    
    return list_of_practice_groups