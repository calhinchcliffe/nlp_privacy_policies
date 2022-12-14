
# import core ds libraries and every library required for the functions
import numpy as np
import pandas as pd
from pandas import json_normalize
import yaml

import matplotlib.pyplot as plt
import seaborn as sns

import sys
import time
from collections import defaultdict
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# modelling
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# modelling pipeline
from tempfile import mkdtemp
# from sklearn.pipeline import Pipeline # not required since using imbalanced learn
from imblearn.pipeline import Pipeline 
    # Using the pipeline from the imbalanced learn library since it allows for sampling
from imblearn import FunctionSampler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV


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




###-------------------------------------------------------------------------------------###




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




###-------------------------------------------------------------------------------------###




def generate_segment_df(all_policies_df):
    
    """
    This function takes in a dataframe with all the policies and returns a new dataframe featuring each segment. 
    """
    
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



###-------------------------------------------------------------------------------------###




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




###-------------------------------------------------------------------------------------###




def add_empty_annotation_columns(dataframe, list_of_columns):
    """
    Append empty annotation columns to the dataframe specified and returns a new dataframe.
    Inputs: 
    - dataframe to append to
    - list of empty column names to add
    Returns a new dataframe.
    """

    empty_annotations = pd.DataFrame( data = 0, columns = list_of_columns, index = range(len(dataframe)) ) 
    # make the list of annotations into columns

    dataframe_with_empty_annots = pd.concat([dataframe, empty_annotations], axis=1) 
    # put the columns onto the segment dataframe

    print(f"The shape of the returned dataframe is {dataframe_with_empty_annots.shape}") # Verify

    return dataframe_with_empty_annots



###-------------------------------------------------------------------------------------###



def get_features_for_practices():
    
    """
    Inputs: none
    Returns a list of lists of crafted features (each list corresponds to the features for a different practice)
    """

    with open("APP_350_v1_1/features.yml", "r") as stream:
        try:
            features_yml = (json_normalize(yaml.safe_load(stream)))
        except yaml.YAMLError as exc:
            print(exc)

    data_types = json_normalize(features_yml['data_types'])

    list_of_practice_features = []
    practice_groups = range(len(data_types.columns))

    for i in practice_groups:
        practices_and_features = json_normalize(data_types[i])

        list_of_practice_features.extend(practices_and_features['features'])

    print(f"{len(list_of_practice_features)} different groups of features returned, containing {len([practice for practice_group in list_of_practice_features for practice in practice_group])} individual features.")

    return list_of_practice_features




###-------------------------------------------------------------------------------------###



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







###-------------------------------------------------------------------------------------###



def get_annots_for_each_sentence(segment_annotations = None, list_of_practices = None):
    """
    This function creates a DataFrame containing every sentence from every privacy policy (one sentence per row), 
    with columns for each annotation.
    
    Steps in this function:
    1. Create an initial DataFrame with just one row, consisting of one sentence plus metadata columns
    2. Append rows to it until every sentence has been added
    3. Add and populate the annotation columns
    
    Inputs:
    - segment_annotations: DataFrame created in the previous notebook, where each row is a segment
    - list_of_practices: A list of every different concatenated annotation
    (annotation of the privacy practice and whether it is 1st party or 3rd party, e.g. 'Contact_E_Mail_Address_1stParty' and 'Location_GPS_1stParty')
    
    Outputs:
    all_annot_sentence_df_annots: resulting DataFrame containing: 
    - a sentence from a policy, 
    - metadata about where the sentence came from, and 
    - that sentence's annotations.
    This only has the sentences that were annotated.
    """
    
    if not isinstance(segment_annotations, pd.DataFrame):
        segment_annotations = pd.read_pickle('objects/segment_annotations.pkl')
    if list_of_practices == None:
        list_of_practice_groups = get_list_of_practice_groups()
        list_of_practices = [practice for practice_group in list_of_practice_groups for practice in practice_group]
    
    # 1. Create an initial DataFrame with just one row, consisting of one sentence plus metadata columns
    
    row = 2 # initialise by starting at row 2, which is the first row with annotated sentences

    # Creating a df at the sentence level by expanding the sentence annotations within the segment annotations df. 
    # This creates a DataFrame with two columns: sentence, and annotations (a dictionary of annotations for the sentence)
    these_sentences = json_normalize(segment_annotations.at[row, 'sentences']) 
    
    # Now appending all policy meta data to the sentences
    these_sentences["source_policy_number"] = segment_annotations.at[row,"source_policy_number"] 
    these_sentences["policy_type"] = segment_annotations.at[row,"policy_type"]
    these_sentences["contains_synthetic"] = segment_annotations.at[row,"contains_synthetic"]
    these_sentences["policy_segment_id"] = segment_annotations.at[row,"policy_segment_id"]
    # re-ordering the columns
    these_sentences = these_sentences[['source_policy_number', 'policy_type', 'contains_synthetic', 'policy_segment_id', 'sentence_text', 'annotations']]
    
    # 2. Append rows until every sentence has been added
    
    all_annot_sentence_df = these_sentences.copy()
    
    for row in range(len(segment_annotations)):
        next_sentences = 0
        if segment_annotations.loc[row, 'sentences'] != []:
            next_sentences = json_normalize(segment_annotations.at[row, 'sentences'])
            next_sentences["source_policy_number"] = segment_annotations.at[row,"source_policy_number"]
            next_sentences["policy_type"] = segment_annotations.at[row,"policy_type"]
            next_sentences["contains_synthetic"] = segment_annotations.at[row,"contains_synthetic"]
            next_sentences["policy_segment_id"] = segment_annotations.at[row,"policy_segment_id"]
            next_sentences = next_sentences[['source_policy_number', 'policy_type', 'contains_synthetic', 'policy_segment_id', 'sentence_text', 'annotations']]
            all_annot_sentence_df = pd.concat([all_annot_sentence_df, next_sentences], axis = 0)
    all_annot_sentence_df.reset_index(drop=True, inplace=True)
    print(f"The shape of the DataFrame with annotated sentences is {all_annot_sentence_df.shape}")
    
    # 3. Add and populate the annotation columns
    
    # add the empty annotation columns
    all_annot_sentence_df_annots = add_empty_annotation_columns(all_annot_sentence_df, list_of_practices)

    # populate the columns with the annotations
    for index in range(len(all_annot_sentence_df_annots)):
        practices_dictionaries = all_annot_sentence_df_annots.loc[index, 'annotations']
        for each_practice in practices_dictionaries:
            all_annot_sentence_df_annots.loc[index, each_practice['practice']] += 1
    
    return all_annot_sentence_df_annots







###-------------------------------------------------------------------------------------###






def sentence_filtering(X, y, df_filter, sf_filter, balanced_downsize_filter):
    """
    Filter the X and y data using Sentence Filtering, 
    or if this leaves too few data, filter using balanced downsize filtering.
    
    Inputs: 
        X: X data
        y: y data
        df_filter:  a filter (boolean series) to use to filter the data. 
                    Intended to be sf_filter (sentence filtering)
                    or balanced_downzise_filter (all the positive cases plus an equally sized random sample of negative cases)
        sf_filter:  the sentence filtering filter for the classifier being trained
                    Intended to be sf_filter = ((X[classifier_features] > 0).sum(axis=1) > 0 )
        balanced_downsize_filter: For filtering to all the positive cases plus an equally sized random sample of negative cases.
                    Intended to be balanced_downzise_filter = (
                        positive_rows |
                        negative_rows.where(negative_rows == True).dropna().sample(n=positive_rows.sum(), replace=False))
    Outputs:
        X2: filtered X data
        y2: filtered y data
    
    """
    
        # filter y
    y2 = y.loc[df_filter].copy()
    y2.reset_index(inplace=True, drop=True)
    
        # filter X
    X2 = X.loc[df_filter].copy()
    X2.reset_index(inplace=True, drop=True)
    
        # check whether this sentence filtering leaves enough data (arbitrary > 75)
        # if not, use balanced downsizing instead:
    if df_filter.equals(sf_filter) & (
        ( y2.value_counts().get(1, 0) < 75 ) or ( y2.value_counts().get(0, 0) < 75 )
    ):
        X2, y2 = sentence_filtering(X, y, balanced_downsize_filter, sf_filter, balanced_downsize_filter)
    
    return X2, y2








###-------------------------------------------------------------------------------------###





def full_modelling_pipeline(classifier, df_for_pipelining, df_for_evaluation, clean_annotation_features,
                            model_results_series, col_transform_unigrams, col_transform_withbigrams):
    
    """
    Inputs:
    
        classifier: name (exact string) of the target column to predict
        
        df_for_pipelining: Dataframe with text segments, classifiers and crafted features
        
        model_results_series:   The empty series to save the results (classifier-objects and predictions) to. 
                                A pickle file of the same name will be saved with the results. 
        
        col_transform_unigrams:    for the column transformer, column transformation with column to apply to
        
        col_transform_withbigrams: as above
    """
    
    # Step 1 ??? declare the classifier
    
    print(f"Running for classifier: {classifier}")
    start_code_time = time.time()
    
    # Step 2 ??? Separate into X and y
    
    # the crafted features columns happen to be all those after and including 'contact info', so I 
    # use every column after and including the first crafted feature, which happens to be 'contact info'
    X = pd.concat(
        [df_for_pipelining['segment_text'], 
         df_for_pipelining.loc[:,'contact info':]],
        axis=1
    ).copy()

    y = df_for_pipelining[classifier]
    
    # Step 3 ??? filter for sentence filtering to use in Pipeline
    
    # filtering the crafted features to get the list of features from the same row that lists the classifier
    classifier_features = clean_annotation_features[ clean_annotation_features['annotation'] == classifier ]     \
                            .reset_index().at[0,'features']
    
    # true/false boolean series for sentence filtering:
    sf_filter = ((X[classifier_features] > 0)\
                     .sum(axis=1) > 0 )
    
    # true/false boolean series for balanced downsizing filter:
    positive_rows = (y == 1)
    negative_rows = (y == 0)
    balanced_downsize_filter = (
        positive_rows |
        negative_rows.where(negative_rows == True).dropna().sample(n=positive_rows.sum(), replace=False)
    )
    
    print("Ready for grid search.")
    
    # Step 4 ??? CV Grid Search
    
    fitted_search = model_pipeline_step_4(X, y, sf_filter, balanced_downsize_filter, 
                                          col_transform_unigrams, col_transform_withbigrams)

    print("Ready for evaluation")
    
    # Step 5 ??? separate test (evaluation) data into X and y, run the model on them and save the results
    
    X_eval = pd.concat(
        [df_for_evaluation['segment_text'], 
         df_for_evaluation.loc[:,'contact info':]],
        axis=1
    ).copy()

    y_eval = df_for_evaluation[classifier]
    
    # Running the model
    classifier_prediction = fitted_search.predict(X_eval)

    # saving the model results for future use
    model_results_series[classifier] = [fitted_search, y_eval, classifier_prediction]
    model_results_series.to_pickle("objects/model_results.pkl")
    
    # Tests
    if type(model_results_series[classifier]) == int:
        print("Model results not saved.")
        raise NotSavedError("Check model results")
    
    print(f"The runtime for {classifier} was {round(time.time() - start_code_time, 5)}")
    print()








###-------------------------------------------------------------------------------------###





def model_pipeline_step_4(X, y, sf_filter, balanced_downsize_filter, 
                          col_transform_unigrams, col_transform_withbigrams):

#     cachedir = mkdtemp() # Optional memory dump to help with processing

    sf_kw_args = {'df_filter': sf_filter, 'sf_filter': sf_filter, 'balanced_downsize_filter': balanced_downsize_filter}
    
    # In the pipeline_sequences list, the syntax requires that each step is named and some value is provided,
    # but that value will be changed as each different option is looped through in the parameter grid.
    pipeline_sequences = [
        ('sentence_filtering', FunctionSampler(func=sentence_filtering, validate=False, kw_args=sf_kw_args)),
        ('tfidf', ColumnTransformer(col_transform_withbigrams, remainder='passthrough')), 
        ('model', LogisticRegression(random_state=1, max_iter=1000))
    ]

    pipe = Pipeline(pipeline_sequences) #, memory = cachedir)

    param_grid = [
        {
            'model': [LogisticRegression(random_state=1, max_iter=1000)],
            'sentence_filtering': [FunctionSampler(func=sentence_filtering, validate=False, kw_args=sf_kw_args), 
                                   None],
            'tfidf': [ColumnTransformer(col_transform_withbigrams, remainder='passthrough'),
                      ColumnTransformer(col_transform_unigrams, remainder='passthrough')]
        },
        {
            'model': [SVC(kernel='linear', class_weight='balanced', random_state=1)],
            'model__C': [0.1, 1, 10],
            'sentence_filtering': [FunctionSampler(func=sentence_filtering, validate=False, kw_args=sf_kw_args), 
                                   None],
            'tfidf': [ColumnTransformer(col_transform_withbigrams, remainder='passthrough'),
                      ColumnTransformer(col_transform_unigrams, remainder='passthrough')]
        }
    ]

    # Create grid search object
    grid_search_object = GridSearchCV(estimator=pipe, param_grid = param_grid, cv = 5, verbose=1, n_jobs=-1, scoring='f1')
    
    fitted_search = grid_search_object.fit(X, y)
    
    return fitted_search 