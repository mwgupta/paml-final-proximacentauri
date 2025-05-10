import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
import os
from sklearn.decomposition import PCA

# All pages
def fetch_dataset():
    """
    This function renders the file uploader that fetches the dataset either from local machine

    Input:
        - page: the string represents which page the uploader will be rendered on
    Output: None
    """
    # Check stored data
    df = None
    dataset_filename = './datasets/Amazon Product Reviews I.csv'
    if 'data' in st.session_state:
        df = st.session_state['data']
    else:
        if os.path.exists(dataset_filename):
            st.write("Loading dataset file: {}".format(dataset_filename))
            df = pd.read_csv(dataset_filename)
        else:
            st.write("File does not exist: {}".format(dataset_filename))
    if df is not None:
        st.session_state['data'] = df
    return df

# Page A
def clean_data(df):
    """
    This function removes all feature but 'reviews.text', 'reviews.title', and 'reviews.rating'
        - Then, it remove Nan values and resets the dataframe indexes

    Input: 
        - df: the pandas dataframe
    Output: 
        - df: updated dataframe
        - data_cleaned (bool): True if data cleaned; else false
    """
    data_cleaned = False
    # Simplify relevant columns names
    if ('reviews.rating' in df.columns):
        df['rating'] = df['reviews.rating']
        df.drop(['reviews.rating'], axis=1, inplace=True)

    if ('reviews.text' in df.columns):
        df['reviews'] = df['reviews.text']
        df.drop(['reviews.text'], axis=1, inplace=True)

    if ('reviews.title' in df.columns):
        df['title'] = df['reviews.title']
        df.drop(['reviews.title'], axis=1, inplace=True)

    # Drop irrelevant columns
    relevant_cols = ['reviews', 'rating', 'title']
    df = df.loc[:, relevant_cols]

    # Drop Nana
    df.dropna(subset=['rating', 'reviews', 'title'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    data_cleaned = True

    # Store new features in st.session_state
    st.session_state['data'] = df
    return df, data_cleaned

# Page B
def reduce_feature_dimensionality(X_train_sentiment, X_val_sentiment):
    """
        Reduce dimensions of training and test dataset features X_train_sentiment, X_val_sentiment
        Inputs:
            - X_train_sentiment (pandas dataframe): training dataset features 
            - X_val_sentiment (pandas dataframe): test dataset features 
        Outputs:
            - X_train_sentiment (numpy array): training dataset features with smaller dimensions (number of columns)
            - X_val_sentiment (numpy array): test dataset features with smaller dimensions (number of columns)
        """
    # Convert input from pandas dataframe to to numpy arrays
    X_train_sentiment, X_val_sentiment = np.array(X_train_sentiment), np.array(X_val_sentiment)
    if len(X_val_sentiment) > 0:
        num_examples, num_features = X_val_sentiment.shape
    else:
        num_examples, num_features = 0, 0
    # User input target number of output features
    num_comps = st.slider(
        f'Select number of features for dimensionality reduction using PCA',
        1, min(num_examples, num_features), 64
    )
    # Use PCA to reduce feature dimensions of training and test dataset features
    if len(X_val_sentiment) > 0:
        pca = PCA(n_components=num_comps)
        pca.fit(X_train_sentiment)
        X_train_sentiment = pca.transform(X_train_sentiment)

        pca.fit(X_val_sentiment)
        X_val_sentiment = pca.transform(X_val_sentiment)
    return X_train_sentiment, X_val_sentiment

# Page A
def remove_review(X, remove_idx):
    """
    This function drops selected feature(s)

    Input: 
        - X: the pandas dataframe
        - remove_idx: the index of review to be removed
    Output: 
        - X: the updated dataframe
    """

    X = X.drop(index=remove_idx)
    return X

# Page A

# Checkpoint 4
def display_review_keyword(df, keyword, n_reviews=5):
    """
    This function shows n_reviews reviews 

    Input: 
        - df: the pandas dataframe
        - keyword: keyword to search in reviews
        - n_reviews: number of review to display
    Output: 
        - None
    """
    keyword_df = df['reviews'].str.contains(keyword)
    filtered_df = df[keyword_df]#.head(n_reviews)

    return filtered_df

# Page B
def set_pos_neg_reviews(df, negative_upper_bound):
    """
    This function updates df with a column called 'sentiment' and sets the positive and negative review sentiment as either -1 or +1

    Input:
        - df: dataframe containing the dataset
        - negative_upper_bound: tuple with upper and lower range of ratings from positive reviews
        - negative_upper_bound: upper bound of negative reviews
    Output:
        - df: dataframe with 'sentiment' column of +1 and -1 for review sentiment
    """
    df = df[df['rating'] != negative_upper_bound].copy()

    # Create a new feature called 'sentiment' and store in df with negative sentiment < up_bound
    df.loc[:, 'sentiment'] = df['rating'].apply(lambda r: +1 if r > negative_upper_bound else -1)

    # Summarize positibve and negative example counts
    st.write('Number of positive examples: {}'.format(
        len(df[df['sentiment'] == 1])))
    st.write('Number of negative examples: {}'.format(
        len(df[df['sentiment'] == -1])))

    # Save updated df st.session_state
    st.session_state['data'] = df
    return df