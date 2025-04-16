import numpy as np                  
import pandas as pd                  
import streamlit as st                 
from pages.B_Train_Model import split_dataset
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### PAML Final - Predicting Mortgage Propensity Scores Using Regression")

#############################################

st.title('Test Model')

#############################################
# Checkpoint 10
def rmse(y_true, y_pred):
    """
    This function computes the root mean squared error by measuring the 
    difference between predicted and actual values using Euclidean distance.

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - error: root mean squared error
    """
    error=0
    
    # Add code here.
    num_examples = y_true.shape[0]
    error = np.sqrt(np.sum(np.power(y_true-y_pred,2))/num_examples)
    return error

# Checkpoint 11
def mae(y_true, y_pred):
    """
    Measures the mean absolute error (MAE) between predicted and actual values.

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - error: mean absolute error
    """
    error=0
    
    # Add code here.
    num_examples = y_true.shape[0]
    error = (np.sum(np.abs(y_true-y_pred))) / num_examples
    return error

# Checkpoint 12
def r2(y_true, y_pred):
    """
    Compute the coefficient of determination (R2 score) which 
    represents the proportion of variance in predicted values 
    that can be explained by the input features.

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - error: r2 score
    """
    error=0
    
    # Add code here.
    tss = np.sum(np.power(y_true - np.mean(y_true),2))
    rss = np.sum(np.power(y_true - y_pred,2))
    error = 1 - (rss/tss)
    return error

# Used to access model performance in dictionaries
METRICS_MAP = {
    'mean_absolute_error': mae,
    'root_mean_squared_error': rmse,
    'r2_score': r2
}

# Helper function
def compute_eval_metrics(X, y_true, model, metrics):
    """
    This function checks the metrics of the models

    Input:
        - X: pandas dataframe with training features
        - y_true: pandas dataframe with true targets
        - model: the model to evaluate
        - metrics: the metrics to evlauate performance 
    Output:
        - metric_dict: a dictionary contains the computed metrics of the selected model, with the following structure:
            - {metric1: value1, metric2: value2, ...}
    """
    metric_dict = {}
    y_pred = model.predict(X)
    for metric in metrics:
        metric_dict[metric] = METRICS_MAP[metric](y_true, y_pred)
    return metric_dict


def plot_learning_curve(X_train, X_val, y_train, y_val, trained_model, metrics, model_name):
    """
    This function plots the learning curve. Note that the learning curve is calculated using 
    increasing sizes of the training samples
    Input:
        - X_train: training features
        - X_val: validation/test features
        - y_train: training targets
        - y_val: validation/test targets
        - trained_model: the trained model to be calculated learning curve on
        - metrics: a list of metrics to be computed
        - model_name: the name of the model being checked
    Output:
        - fig: the plotted figure
        - df: a dataframe containing the train and validation errors, with the following keys:
            - df[metric_fn.__name__ + " Training Set"] = train_errors
            - df[metric_fn.__name__ + " Validation Set"] = val_errors
    """
    fig = make_subplots(rows=len(metrics), cols=1,
                        shared_xaxes=True, vertical_spacing=0.1)
    df = pd.DataFrame()
    for i, metric in enumerate(metrics):
        metric_fn = METRICS_MAP[metric]
        train_errors, val_errors = [], []
        for m in range(500, len(X_train)+1, 500):
            trained_model.fit(X_train[:m], y_train[:m])
            y_train_predict = trained_model.predict(X_train[:m])
            y_val_predict = trained_model.predict(X_val)
            train_errors.append(metric_fn(y_train[:m], y_train_predict))
            val_errors.append(metric_fn(y_val, y_val_predict))

        fig.add_trace(go.Scatter(x=np.arange(500, len(X_train)+1, 500),
                      y=train_errors, name=metric_fn.__name__+" Train"), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(500, len(X_train)+1, 500),
                      y=val_errors, name=metric_fn.__name__+" Val"), row=i+1, col=1)

        fig.update_xaxes(title_text="Training Set Size")
        fig.update_yaxes(title_text=metric_fn.__name__, row=i+1, col=1)
        fig.update_layout(title=model_name)

        df[metric_fn.__name__ + " Training Set"] = train_errors
        df[metric_fn.__name__ + " Validation Set"] = val_errors
    return fig, df

# Helper function
def restore_data(df):
    """
    This function restores the training and validation/test datasets from the training page using st.session_state
                Note: if the datasets do not exist, re-split using the input df

    Input: 
        - df: the pandas dataframe
    Output: 
        - X_train: the training features
        - X_val: the validation/test features
        - y_train: the training targets
        - y_val: the validation/test targets
    """
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    # Restore train/test dataset
    if ('X_train' in st.session_state):
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        st.write('Restored train data ...')
    if ('X_val' in st.session_state):
        X_val = st.session_state['X_val']
        y_val = st.session_state['y_val']
        st.write('Restored test data ...')
    if('target' in st.session_state):
        feature_predict_select = st.session_state['target']
        st.write('Restored target ...')
    if('feature' in st.session_state):
        feature_input_select = st.session_state['feature']
        st.write('Restored feature input ...')
        
    if (X_train is None):
        # Select variable to explore
        numeric_columns = list(df.select_dtypes(include='number').columns)
        # Select variable to predict
        feature_predict_select = st.selectbox(
            label='Select variable to predict',
            options=list(df.select_dtypes(include='number').columns),
            key='feature_selectbox',
            index=8
        )

        st.session_state['target'] = feature_predict_select

        # Select input features
        feature_input_select = st.multiselect(
            label='Select features for regression input',
            options=[f for f in list(df.select_dtypes(
                include='number').columns) if f != feature_predict_select],
            key='feature_multiselect'
        )

        st.session_state['feature'] = feature_input_select

        st.write('You selected input {} and output {}'.format(
            feature_input_select, feature_predict_select))

        df = df.dropna()
        X = df.loc[:, df.columns.isin(feature_input_select)]
        Y = df.loc[:, df.columns.isin([feature_predict_select])]

        # Convert to numpy arrays
        X = np.asarray(X.values.tolist()) 
        Y = np.asarray(Y.values.tolist()) 

        # Split train/test
        st.markdown(
            '### Enter the percentage of test data to use for training the model')
        number = st.number_input(
            label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

        X_train, X_val, y_train, y_val = split_dataset(X, Y, number)
        st.write('Restored training and test data ...')
    return X_train, X_val, y_train, y_val

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

random.seed(10)
###################### FETCH DATASET #######################
df = None
if('house_df' in st.session_state):
    df = st.session_state['house_df']
else:
    filepath = st.file_uploader('Upload a Dataset', type=['csv', 'txt'])
    if(filepath):
        df = load_dataset(filepath)

if df is not None:
    # Restore dataset splits
    X_train, X_val, y_train, y_val = restore_data(df)

    st.markdown("## Get Performance Metrics")
    metric_options = ['mean_absolute_error',
                      'root_mean_squared_error', 'r2_score']
    
    # Select multiple metrics for evaluation
    metric_select = st.multiselect(
        label='Select metrics for regression model evaluation',
        options=metric_options,
    )
    if (metric_select):
        st.session_state['metric_select'] = metric_select
        st.write('You selected the following metrics: {}'.format(metric_select))

    regression_methods_options = ['Multiple Linear Regression',
                                  'Polynomial Regression', 'Ridge Regression']
    trained_models = [
        model for model in regression_methods_options if model in st.session_state]
    st.session_state['trained_models'] = trained_models

    # Select a trained regression model for evaluation
    model_select = st.multiselect(
        label='Select trained regression models for evaluation',
        options=trained_models
    )

    plot_options = ['Learning Curve', 'Metric Results']

    review_plot = st.multiselect(
        label='Select plot option(s)',
        options=plot_options
    )
    
    st.write('You selected the following models for evaluation: {}'.format(model_select))
    if (model_select and review_plot):

        eval_button = st.button('Evaluate your selected regression models')

        if eval_button:
            st.session_state['eval_button_clicked'] = eval_button

        #if 'eval_button_clicked' in st.session_state and st.session_state['eval_button_clicked']:

            if 'Learning Curve' in review_plot:
                for model in model_select:
                    trained_model = st.session_state[model]
                    fig, df = plot_learning_curve(
                        X_train, X_val, y_train, y_val, trained_model, metric_select, model)
                    st.plotly_chart(fig)

            if 'Metric Results' in review_plot:
                models = [st.session_state[model]
                          for model in model_select]

                train_result_dict = {}
                val_result_dict = {}
                for idx, model in enumerate(models):
                    train_result_dict[model_select[idx]] = compute_eval_metrics(
                        X_train, y_train, model, metric_select)
                    val_result_dict[model_select[idx]] = compute_eval_metrics(
                        X_val, y_val, model, metric_select)

                st.markdown('### Predictions on the training dataset')
                st.dataframe(train_result_dict)

                st.markdown('### Predictions on the validation dataset')
                st.dataframe(val_result_dict)

    # Deploy Model

    # Select a model to deploy from the trained models
    #st.markdown("## Choose your Deployment Model")
    #model_select = st.selectbox(
    #    label='Select the model you want to deploy',
    #    options=st.session_state['trained_models'],
    #)

    #if (model_select):
    #    st.write('You selected the model: {}'.format(model_select))
    #    st.session_state['deploy_model'] = st.session_state[model_select]

    #st.write('Continue to Deploy Model')