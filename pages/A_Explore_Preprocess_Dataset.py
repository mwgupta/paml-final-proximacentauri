import streamlit as st
import pandas as pd
import plotly.express as px
from pandas.plotting import scatter_matrix
import os
from itertools import combinations
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

#############################################

# st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### PAML Final - Predicting Mortgage Propensity Scores Using Regression")

#############################################

st.markdown('# Explore/Preprocess Dataset')

#############################################

st.markdown('### Import Dataset')

# Helper Function
def load_dataset(filepath):
    """
    This function uses the filepath (string) a .csv file locally on a computer 
    to import a dataset with pandas read_csv() function. Then, store the 
    dataset in session_state.

    Input: data is the filename or path to file (string)
    Output: pandas dataframe df
    """
    data = pd.read_csv(filepath)
    st.session_state['house_df'] = data
    return data

# Helper function
def sidebar_filter(df, chart_type, x=None, y=None):
    """
    This function renders the feature selection sidebar 

    Input: 
        - df: pandas dataframe containing dataset
        - chart_type: the type of selected chart
        - x: features
        - y: targets
    Output: 
        - list of sidebar filters on features
    """
    df=df.dropna()
    side_bar_data = []

    select_columns = []
    if (x is not None):
        select_columns.append(x)
    if (y is not None):
        select_columns.append(y)
    if (x is None and y is None):
        select_columns = list(df.select_dtypes(include='number').columns)

    for idx, feature in enumerate(select_columns):
        try:
            f = st.sidebar.slider(
                str(feature),
                float(df[str(feature)].min()),
                float(df[str(feature)].max()),
                (float(df[str(feature)].min()), float(df[str(feature)].max())),
                key=chart_type+str(idx)
            )
        except Exception as e:
            print(e)
        side_bar_data.append(f)
    return side_bar_data

# Helper Function
def compute_correlation(df, features):
    """
    This function computes pair-wise correlation coefficents of X and render summary strings

    Input: 
        - df: pandas dataframe 
        - features: a list of feature name (string), e.g. ['age','height']
    Output: 
        - correlation: correlation coefficients between one or more features
        - summary statements: a list of summary strings where each of it is in the format: 
            '- Features X and Y are {strongly/weakly} {positively/negatively} correlated: {correlation value}'
    """
    correlation = df[features].corr()
    feature_pairs = combinations(features, 2)
    cor_summary_statements = []

    for f1, f2 in feature_pairs:
        cor = correlation[f1][f2]
        summary = '- Features %s and %s are %s %s correlated: %.2f' % (
            f1, f2, 'strongly' if cor > 0.5 else 'weakly', 'positively' if cor > 0 else 'negatively', cor)
        st.write(summary)
        cor_summary_statements.append(summary)

    return correlation, cor_summary_statements

def summarize_missing_data(df, top_n=3):
    """
    This function summarizes missing values in the dataset

    Input: 
        - df: the pandas dataframe
        - top_n: top n features with missing values, default value is 3
    Output: 
        - a dictionary containing the following keys and values: 
            - 'num_categories': counts the number of features that have missing values
            - 'average_per_category': counts the average number of missing values across features
            - 'total_missing_values': counts the total number of missing values in the dataframe
            - 'top_missing_categories': lists the top n features with missing values
    """
    out_dict = {'num_categories': 0,
                'average_per_category': 0,
                'total_missing_values': 0,
                'top_missing_categories': []}

    # Used for top categories with missing data
    missing_column_counts = df[df.columns[df.isnull().any()]].isnull().sum()
    max_idxs = np.argsort(missing_column_counts.to_numpy())[::-1][:top_n]

    # Compute missing statistics
    out_dict['num_categories'] = df.isna().any(axis=0).sum()
    out_dict['average_per_category'] = df.isna().sum().sum()/len(df.columns)
    out_dict['total_missing_values'] = df.isna().sum().sum()
    out_dict['top_missing_categories'] = df.columns[max_idxs[:top_n]].to_numpy()

    # Display missing statistics
    st.markdown('Number of categories with missing values: {0:.2f}'.format(
        out_dict['num_categories']))
    st.markdown('Average number of missing values per category: {0:.2f}'.format(
        out_dict['average_per_category']))
    st.markdown('Total number of missing values: {0:.2f}'.format(
        out_dict['total_missing_values']))
    st.markdown('Top {} categories with most missing values: {}'.format(
        top_n, out_dict['top_missing_categories']))
    return out_dict

def remove_features(df,removed_features):
    """
    Remove the features in removed_features (list) from the input pandas dataframe df. 

    Input: df is dataset in pandas dataframe
    Output: pandas dataframe df
    """
    X = df.copy()
    X  = X.drop(removed_features, axis=1)
    st.session_state['house_df'] = X
    return X

def remove_nans(df):
    """
    This function removes all NaN values in the dataframe

    Input: 
        - df: pandas dataframe
    Output: 
        - df: updated df with no Nan observations
    """
    # Remove obs with nan values
    df = df.dropna()
    # df.to_csv('remove_nans.csv',index= False)
    st.session_state['house_df'] = df
    return df

def impute_dataset(df, impute_method):
    """
    Impute the dataset df with imputation method impute_method 
    including mean, median, zero values or drop Nan values in 
    the dataset (all numeric and string columns).

    Input: 
    - df is dataset in pandas dataframe
    - impute_method = {'Zero', 'Mean', 'Median','DropNans'}
    Output: pandas dataframe df
    """
    df=df.dropna()
    X = df.copy()
    nan_colns = X.columns[X.isna().any()].tolist()
    numeric_columns = list(X.select_dtypes(['float','int']).columns)
    if impute_method == 'Zero':
        # X = X.fillna(0)
        for col in nan_colns: 
            if(col in numeric_columns):
                X[col].fillna(0, inplace=True)
    elif impute_method == 'Mean':
        for col in nan_colns: 
            if(col in numeric_columns):
                X[col].fillna(value=X[col].mean(), inplace=True)
    elif impute_method == 'Median':
        for col in nan_colns:
            if(col in numeric_columns):
                X[col].fillna(value=X[col].median(), inplace=True)
    elif impute_method == 'DropNans':
        data_size1 = X.size
        X = X.dropna()
        data_size2 = X.size
        st.write('%d values removed from the dataset' %(np.abs(data_size2-data_size1)))
    st.session_state['house_df'] = X
    return X

def remove_outliers(df, features, outlier_removal_method=None):
    """
    This function removes the outliers of the given feature(s)

    Input: 
        - df: pandas dataframe
        - feature: the feature(s) to remove outliers
    Output: 
        - dataset: the updated data that has outliers removed
        - lower_bound: the lower 25th percentile of the data
        - upper_bound: the upper 25th percentile of the data
    """
    df=df.dropna()
    dataset = df.copy()

    for feature in features:
        lower_bound = dataset[feature].max()
        upper_bound = dataset[feature].min()

        if(outlier_removal_method =='IQR'): # IQR method
            if (feature in df.columns):
                dataset = dataset.dropna()
                Q1 = np.percentile(dataset[feature], 25, axis=0)
                Q3 = np.percentile(dataset[feature], 75, axis=0)
                IQR = Q3 - Q1
                upper_bound = Q3 + 1.5*IQR
                lower_bound = Q1 - 1.5*IQR
        else: # Standard deviation methods
            upper_bound = dataset[feature].mean() + 3* dataset[feature].std() #mean + 3*std
            lower_bound = dataset[feature].mean() - 3* dataset[feature].std() #mean - 3*std
        dataset_size1 = dataset.size
        dataset = dataset[dataset[feature] > lower_bound]
        dataset = dataset[dataset[feature] < upper_bound]
        dataset_size2 = dataset.size
        st.write('%s: %d outliers were removed from feature %s in the dataset' % (outlier_removal_method,dataset_size1-dataset_size2, feature))

    st.session_state['house_df'] = dataset
    return dataset

def one_hot_encode_feature(df, features):
    """
    This function performs one-hot encoding on the given features,
    removes the original columns, and ensures one-hot columns are integers (0/1).

    Input: 
        - df: the pandas dataframe
        - features: the feature(s) to perform one-hot-encoding
    Output: 
        - df: dataframe with one-hot-encoded features as integers
    """
    df = df.dropna()

    # One-hot encode the selected features
    dummies = pd.get_dummies(df[features], drop_first=False).astype(int)

    # Drop original categorical columns
    df = df.drop(columns=features)

    # Concatenate encoded columns back into DataFrame
    df = pd.concat([df, dummies], axis=1)

    st.write(f'Features {features} have been one-hot encoded.')
    st.session_state['house_df'] = df
    return df

def integer_encode_feature(df, features):
    """
    This function performs integer-encoding on the given features and drops the originals.

    Input: 
        - df: the pandas dataframe
        - features: the feature(s) to perform integer-encoding
    Output: 
        - df: dataframe with integer-encoded feature(s)
    """
    enc = OrdinalEncoder()
    df[features] = enc.fit_transform(df[features])
    
    st.write(f'Features {features} have been integer encoded.')
    st.session_state['house_df'] = df
    return df

def create_feature(df, math_select, math_feature_select, new_feature_name):
    """
    Create a new feature with name new_feature_name in dataset df with the 
    mathematical operation math_select (string) on features math_feature_select (list). 

    Input: 
        - df: the pandas dataframe
        - math_select: the math operation to perform on the selected features
        - math_feature_select: the features to be performed on
        - new_feature_name: the name for the new feature
    Output: 
        - df: the udpated dataframe
    """
    df = df.dropna()
    if (len(math_feature_select) == 1): 
        if(math_select == 'square root'):  # sqrt
            df[new_feature_name] = np.sqrt(df[math_feature_select])
        if(math_select == 'ceil'):  # ceil
            df[new_feature_name] = np.ceil(df[math_feature_select])
        if(math_select == 'floor'):  # floor
            df[new_feature_name] = np.floor(df[math_feature_select])
    else:
        if (math_select == 'add'):
            df[new_feature_name] = df[math_feature_select[0]] + df[math_feature_select[1]]
        elif (math_select == 'subtract'):
            df[new_feature_name] = df[math_feature_select[0]] - df[math_feature_select[1]]
        elif (math_select == 'multiply'):
            df[new_feature_name] = df[math_feature_select[0]] * df[math_feature_select[1]]
        elif (math_select == 'divide'):
            df[new_feature_name] = df[math_feature_select[0]] / df[math_feature_select[1]]
    st.session_state['house_df'] = df
    return df

def compute_descriptive_stats(df, stats_feature_select, stats_select):
    """
    Compute descriptive statistics stats_select on a feature stats_feature_select 
    in df. Statistics stats_select include mean, median, max, and min. Return 
    the results in an output string out_str and dictionary out_dict (dictionary).

    Input: 
    - df: the pandas dataframe
    - stats_feature_select: list of feaures to computer statistics on
    - stats_select: list of mathematical opations
    Output: 
    - output_str: string used to display feature statistics
    - out_dict: dictionary of feature statistics
    """
    output_str=''
    out_dict = {
        'mean': None,
        'median': None,
        'max': None,
        'min': None
    }
    df=df.dropna()
    X = df.copy()
    for f in stats_feature_select:
        output_str = str(f)
        for s in stats_select:
            if(s=='Mean'):
                mean = round(X[f].mean(), 2)
                output_str = output_str + ' mean: {0:.2f}    |'.format(mean)
                out_dict['mean'] = mean
            elif(s=='Median'):
                median = round(X[f].median(), 2)
                output_str = output_str + ' median: {0:.2f}    |'.format(median)
                out_dict['median'] = median
            elif(s=='Max'):
                max = round(X[f].max(), 2)
                output_str = output_str + ' max: {0:.2f}    |'.format(max)
                out_dict['max'] = max
            elif(s=='Min'):
                min = round(X[f].min(), 2)
                output_str = output_str + ' min: {0:.2f}    |'.format(min)
                out_dict['min'] = min
        st.write(output_str)
    return output_str, out_dict

def scale_features(df, features, scaling_method): 
    """
    Use the scaling_method to transform numerical features in the dataset df. 

    Input: 
        - df: the pandas dataframe
        - features: list of features
        - scaling method is a string; Options include {'Standardarization', 'Normalization', 'Log'}
    Output: 
        - Standarization: X_new = (X - mean)/Std
        - Normalization: X_new = (X - X_min)/(X_max - X_min)
        - Log: X_log = log(X)
    """
    df = df.dropna()
    X = df.copy()
    for f in features:
        if(scaling_method == 'Standardarization'):
            X[f+'_std'] = (X[f] - X[f].mean()) / X[f].std()
            st.write('Feature {} is scaled using {}'.format(f, scaling_method))
        elif(scaling_method == 'Normalization'):
            X[f+'_norm'] = (X[f] - X[f].min()) / (X[f].max() - X[f].min())  
            st.write('Feature {} is scaled using {}'.format(f, scaling_method))
        elif(scaling_method == 'Log'):
            X[f+'_log'] = np.log2(X[f])
            X[X[f+'_log']<0] = 0 # Check for -inf
            st.write('Feature {} is scaled using {}'.format(f, scaling_method))
        else:
            st.write('scaling_method is invalid.')

    st.session_state['house_df'] = X
    return X

###################### FETCH DATASET #######################
df = None
if('house_df' in st.session_state):
    df = st.session_state['house_df']
else:
    filepath = st.file_uploader('Upload a Dataset', type=['csv', 'txt'])
    if(filepath):
        df = load_dataset(filepath)

######################### MAIN BODY #########################

######################### EXPLORE DATASET #########################

if df is not None:
    st.markdown('### 1. Explore Dataset Features')

    # Restore dataset if already in memory
    st.session_state['house_df'] = df

    # Display dataframe as table
    st.dataframe(df.describe())

    ###################### VISUALIZE DATASET #######################
    st.markdown('### 2. Visualize Features')

    numeric_columns = list(df.select_dtypes(include='number').columns)
    #numeric_columns = list(df.select_dtypes(['float','int']).columns)    
    # Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    # Collect user plot selection
    st.sidebar.header('Select type of chart')
    chart_select = st.sidebar.selectbox(
        label='Type of chart',
        options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot']
    )

    # Draw plots
    if chart_select == 'Scatterplots':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            side_bar_data = sidebar_filter(
                df, chart_select, x=x_values, y=y_values)
            plot = px.scatter(data_frame=df,
                              x=x_values, y=y_values,
                              range_x=[side_bar_data[0][0],
                                       side_bar_data[0][1]],
                              range_y=[side_bar_data[1][0],
                                       side_bar_data[1][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Histogram':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.histogram(data_frame=df,
                                x=x_values,
                                range_x=[side_bar_data[0][0],
                                         side_bar_data[0][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Lineplots':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            side_bar_data = sidebar_filter(
                df, chart_select, x=x_values, y=y_values)
            plot = px.line(df,
                           x=x_values,
                           y=y_values,
                           range_x=[side_bar_data[0][0],
                                    side_bar_data[0][1]],
                           range_y=[side_bar_data[1][0],
                                    side_bar_data[1][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Boxplot':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.box(df,
                          x=x_values,
                          range_x=[side_bar_data[0][0],
                                   side_bar_data[0][1]])
            st.write(plot)
        except Exception as e:
            print(e)

    # Display original dataframe
    st.markdown('## 3. View initial data with missing values or invalid inputs')
    st.dataframe(df)

    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    # Show summary of missing values including 
    missing_data_summary = summarize_missing_data(df)

    # Remove param
    st.markdown('### 4. Remove irrelevant/useless features')
    removed_features = st.multiselect(
        'Select features',
        df.columns,
    )
    df = remove_features(df, removed_features)

    ########
    # Display updated dataframe
    st.dataframe(df)

    # Impute features
    st.markdown('### 5. Impute data')
    st.markdown('Transform missing values to 0, mean, or median')

    # Use selectbox to provide impute options {'Zero', 'Mean', 'Median'}
    impute_method = st.selectbox(
        'Select imputation method',
        ('Zero', 'Mean', 'Median','DropNans')
    )

    # Call impute_dataset function to resolve data handling/cleaning problems
    df = impute_dataset(df, impute_method)
    
    # Display updated dataframe
    st.markdown('### Result of the imputed dataframe')
    st.dataframe(df)

############################################# PREPROCESS DATA #############################################
    # Handling Text and Categorical Attributes
    st.markdown('### 6. Handling Text and Categorical Attributes')
    string_columns = list(set(df.columns) - {'session_id', 'pages_viewed', 'time_on_site_sec', 'propensity_score'})

    int_col, one_hot_col = st.columns(2)

    # Perform Integer Encoding
    with int_col:
        text_feature_select_int = st.multiselect(
            'Select text features for Integer encoding',
            ['ALL'] + string_columns,
        )
        if text_feature_select_int:
            if 'ALL' in text_feature_select_int:
                selected_cols = string_columns
            else:
                selected_cols = text_feature_select_int
            if st.button('Integer Encode feature'):
                df = integer_encode_feature(df, selected_cols)

    # Perform One-hot Encoding
    with one_hot_col:
        text_feature_select_onehot = st.multiselect(
            'Select text features for One-hot encoding',
            ['ALL'] + string_columns,
        )
        if text_feature_select_onehot:
            if 'ALL' in text_feature_select_onehot:
                selected_cols = string_columns
            else:
                selected_cols = text_feature_select_onehot
            if st.button('One-hot Encode feature'):
                df = one_hot_encode_feature(df, selected_cols)

    # Show updated dataset
    st.write(st.session_state['house_df']) # TODO: not saving and printing correctly

    # Sacling features
    st.markdown('### 7. Feature Scaling')
    st.markdown('Use standardarization or normalization to scale features')

    # Use selectbox to provide impute options {'Standardarization', 'Normalization', 'Log'}
    scaling_method = st.selectbox(
        'Select feature scaling method',
        ('Standardarization', 'Normalization', 'Log')
    )

    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    scale_features_select = st.multiselect(
        'Select features to scale',
        numeric_columns,
    )

    if (st.button('Scale Features')):
        # Call scale_features function to scale features
        if(scaling_method and scale_features_select):
            df = scale_features(df, scale_features_select, scaling_method)

    # Display updated dataframe
    # st.dataframe(df)
    st.write(df)

    # Create New Features
    st.markdown('## 8. Create New Features')
    st.markdown(
        'Create new features by selecting two features below and selecting a mathematical operator to combine them.')
    math_select = st.selectbox(
        'Select a mathematical operation',
        ['add', 'subtract', 'multiply', 'divide', 'square root', 'ceil', 'floor'],
    )

    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    if (math_select):
        if (math_select == 'square root' or math_select == 'ceil' or math_select == 'floor'):
            math_feature_select = st.multiselect(
                'Select features for feature creation',
                numeric_columns,
            )
            sqrt = np.sqrt(df[math_feature_select])
            if (math_feature_select):
                new_feature_name = st.text_input('Enter new feature name')
                if (st.button('Create new feature')):
                    if (new_feature_name):
                        df = create_feature(
                            df, math_select, math_feature_select, new_feature_name)
                        st.write(df)
        else:
            math_feature_select1 = st.selectbox(
                'Select feature 1 for feature creation',
                numeric_columns,
            )
            math_feature_select2 = st.selectbox(
                'Select feature 2 for feature creation',
                numeric_columns,
            )
            if (math_feature_select1 and math_feature_select2):
                new_feature_name = st.text_input('Enter new feature name')
                if (st.button('Create new feature')):
                    df = create_feature(df, math_select, [
                                        math_feature_select1, math_feature_select2], new_feature_name)
                    st.write(df)

    st.markdown('### 9. Inspect Features for outliers')
    outlier_feature_select = None
    numeric_columns = list(df.select_dtypes(include='number').columns)

    outlier_method_select = st.selectbox(
        'Select statistics to display',
        ['IQR', 'STD']
    )

    outlier_feature_select = st.multiselect(
        'Select a feature for outlier removal',
        numeric_columns,
    )
    if (outlier_feature_select and st.button('Remove Outliers')):
        df = remove_outliers(df, outlier_feature_select, outlier_method_select)
        st.write(df)

    # Descriptive Statistics 
    st.markdown('### 10. Summary of Descriptive Statistics')

    stats_numeric_columns = list(df.select_dtypes(['float','int']).columns)
    stats_feature_select = st.multiselect(
        'Select features for statistics',
        stats_numeric_columns,
    )

    stats_select = st.multiselect(
        'Select statistics to display',
        ['Mean', 'Median','Max','Min']
    )
            
    # Compute Descriptive Statistics including mean, median, min, max
    display_stats, _ = compute_descriptive_stats(df, stats_feature_select, stats_select)

    ###################### CORRELATION ANALYSIS #######################
    st.markdown("### 11. Correlation Analysis")
    # Collect features for correlation analysis using multiselect
    numeric_columns = list(df.select_dtypes(['float','int']).columns)


    select_features_for_correlation = st.multiselect(
        'Select features for visualizing the correlation analysis (up to 4 recommended)',
        numeric_columns,
    )

    # Compute correlation between selected features
    correlation, correlation_summary = compute_correlation(
        df, select_features_for_correlation)
    st.write(correlation)

    # Display correlation of all feature pairs
    if select_features_for_correlation:
        try:
            fig = scatter_matrix(
                df[select_features_for_correlation], figsize=(12, 8))
            st.pyplot(fig[0][0].get_figure())
        except Exception as e:
            print(e)

    # st.markdown('#### Continue to Preprocess Data')