
# import core ds libraries
import numpy as np
import pandas as pd
from pandas import json_normalize
import yaml

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer


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
    # copy over some other columns
    initial_segment_df["source_policy_number"] = all_policies_df.loc[0,"policy_id"]
    initial_segment_df["policy_type"] = all_policies_df.loc[0,"policy_type"]
    initial_segment_df["contains_synthetic"] = all_policies_df.loc[0,"contains_synthetic"]

    segment_df = initial_segment_df
    
    for i in all_policies_df.index: # loop through each policy
        
        this_segment = all_policies_df.loc[i,"segments"]
        this_segment_df = json_normalize(this_segment) # normalise it
        this_segment_df.set_index('segment_id', inplace=True)
        this_segment_df["source_policy_number"] = all_policies_df.loc[i,"policy_id"]
        this_segment_df["policy_type"] = all_policies_df.loc[i,"policy_type"]
        this_segment_df["contains_synthetic"] = all_policies_df.loc[i,"contains_synthetic"]

        segment_df = pd.concat( [segment_df, this_segment_df], axis=0 ) 
    
    # Tidy indexes and columns
    segment_df['policy_segment_id'] = segment_df.index
    segment_df.reset_index(drop=True, inplace=True)
    segment_df.index.names = ['segment_id']
    segment_df = segment_df[['source_policy_number', 'policy_type', 'contains_synthetic', 'policy_segment_id', 'segment_text', 'annotations', 'sentences']]
    
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


###


def add_empty_annotation_columns(dataframe, list_of_practices):
    # append empty annotation columns to the dataframe specified and returns a new dataframe.

    empty_annotations = pd.DataFrame( data = 0, columns = list_of_practices, index = range(len(dataframe)) ) 
    # make the list of annotations into columns

    dataframe_with_empty_annots = pd.concat([dataframe, empty_annotations], axis=1) 
    # put the columns onto the segment dataframe

    print(f"The shape of the returned dataframe is {dataframe_with_empty_annots.shape}") # Verify

    return dataframe_with_empty_annots



###



# code taken from https://app.neptune.ai/neptune-ai/eda-nlp-tools/n/0-0-eda-nlp-million-headlines-f42d5ffd-0a6c-47f5-8cb8-bcddc93bf5e7
def get_top_ngrams(corpus, n=None, top_n=10):
    """
    Inputs: 
    - corpus : column from your df with all your text. Each row can be a sentence.
    - n=None : n of ngrams to return, e.g. n=2 for bi-grams.
    - top_n = 10 : how many of the most frequently occuring n-grams would you like to return? e.g. top_n = 5 just to get the top 5
    Example: 
    top_n_bigrams = get_top_ngram(segment_df['segment_text'], 2, top_n=15)
    Then you can rearrange it to put it into a graph as so:
    x, y = map(list, zip(*top_n_bigrams))
    """
    vec = CountVectorizer(ngram_range = (n, n)).fit(corpus) # set up the CountVectorizer w.r.t. n
    
    bag_of_words = vec.transform(corpus) # sparse matrix of headlines & n-grams
    
    sum_words = bag_of_words.sum( axis = 0 ) # for each n-gram, sum of rows it appears in(?)
    
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()] 
    # vocabulary_.items() is a dict of bigram : feature index
    # sum_words is in the same order so we can use vocabulary_.items() index
    # to access the sum_words object and grab the total occurrences
    # Thus words_freq lists each word and the total occurrences
    
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True) # sorting by frequency
    return words_freq[:top_n]