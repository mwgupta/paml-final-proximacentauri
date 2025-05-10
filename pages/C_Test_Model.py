import numpy as np                   
import pandas as pd               
import streamlit as st                 
import random
import plotly.graph_objects as go
from helper_functions import fetch_dataset
from pages.B_Train_Model import split_dataset
from sklearn.metrics import confusion_matrix

random.seed(10)
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown(
    "### Homework 2 - Predicting Product Review Sentiment Using Classification")

#############################################

st.title('Test Model')

#############################################

# Checkpoint 19
def compute_accuracy(prediction_labels, true_labels):    
    """
    Compute classification accuracy
    Input
        - prediction_labels (numpy): predicted product sentiment
        - true_labels (numpy): true product sentiment
    Output
        - accuracy (float): accuracy percentage (0-100%)
    """
    accuracy=None
    try:
        # Add code here
                
        corr_predictions = np.sum(prediction_labels == true_labels)
        tot_predictions = len(true_labels)
        accuracy = corr_predictions / tot_predictions

        st.write('compute_accuracy function has not been completed. Remove this statement upon completion.')
    except ValueError as err:
        st.write({str(err)})
    return accuracy

# Checkpoint 20
def compute_precison_recall(prediction_labels, true_labels, print_summary=False):
    """
    Compute precision and recall 
    Input
        - prediction_labels (numpy): predicted product sentiment
        - true_labels (numpy): true product sentiment
    Output
        - precision (float): precision score = TP/TP+FP
        - recall (float): recall score = TP/TP+FN
    """
    precision=None
    recall=None
    try:
        # Add code here
        
        pred_labels = np.array(prediction_labels)
        actual_labels = np.array(true_labels)
        
        TP = np.sum((prediction_labels == 1) & (true_labels == 1))
        FP = np.sum((prediction_labels == 1) & (true_labels == 0))
        FN = np.sum((prediction_labels == 0) & (true_labels == 1))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        if print_summary:
            st.write(f"True Positives: {TP}")
            st.write(f"False Positives: {FP}")
            st.write(f"False Negatives: {FN}")
            st.write(f"Precision: {precision:.3f}")
            st.write(f"Recall: {recall:.3f}")

        st.write('compute_precison_recall function has not been completed. Remove this statement upon completion.')
    except ValueError as err:
        st.write({str(err)})

    return precision, recall

# Helper Function
def compute_eval_metrics(X, y_true, model, metrics, print_summary=False):
    """
    This function computes one or more metrics (precision, recall, accuracy) using the model

    Inputs:
        - X: pandas dataframe with training features
        - y_true: pandas dataframe with true targets
        - model: the model to evaluate
        - metrics: the metrics to evaluate performance (string); 'precision', 'recall', 'accuracy'
    Outputs:
        - metric_dict: a dictionary contains the computed metrics of the selected model, with the following structure:
            - {metric1: value1, metric2: value2, ...}
    """
    metric_dict = {'precision': -1,
                   'recall': -1,
                   'accuracy': -1}
    try:
        # Predict the product sentiment using the input model and data X
        y_pred = model.predict(X)

        # Compute the evaluation metrics in 'metrics = ['precision', 'recall', 'accuracy']' using the predicted sentiment
        precision, recall = compute_precison_recall(y_pred, y_true.to_numpy())
        accuracy = compute_accuracy(y_pred, y_true.to_numpy())

        if 'precision' in metrics:
            metric_dict['precision'] = precision
        if 'recall' in metrics:
            metric_dict['recall'] = recall
        if 'accuracy' in metrics:
            metric_dict['accuracy'] = accuracy
    except ValueError as err:
        st.write({str(err)})
    return metric_dict


# Helper Function
def plot_decision_boundary(X_2d, y, models):
    try:
        st.markdown('### Visualizing the Classification Decision Boundary')
        st.write(
            "To better understand the trained classifiers, we\'ll visualize the features where the classifier switches from predicting a positive review to a negative review."
            " This is called the decision boundary.\n\n"
        )
        
        num_comps = X_2d.shape[-1]
        # pick two dimensions to plot
        feat_plot0 = st.slider(
            'First feature to plot',
            0, num_comps-1, 7
        )
        feat_plot1 = st.slider(
            'Second feature to plot',
            0, num_comps-1, 40
        )

        st.write(f'You selected: features {feat_plot0} and {feat_plot1} to plot.')
        

        for model in models:
            if model.model_name == 'Naive Bayes':
                # features must be positive for naive bayes
                X_plot = X_2d - X_2d.min()
            else:
                X_plot = X_2d
            boundary = model.decision_boundary([feat_plot0, feat_plot1])

            # Add scatter plot for data points
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=X_plot[:, feat_plot0][y == 1], y=X_plot[:, feat_plot1][y == 1],
                    mode='markers',
                    marker=dict(color='blue'),
                    name="y = 1"  # Legend name for this class
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=X_plot[:, feat_plot0][y == -1], y=X_plot[:, feat_plot1][y == -1],
                    mode='markers',
                    marker=dict(color='red'),
                    name="y = -1"  # Legend name for this class
                )
            )
            # Add decision boundary as a line plot
            fig.add_trace(
                go.Scatter(
                    x=[boundary[0], min(X_plot[:, feat_plot0])],
                    y=[min(X_plot[:, feat_plot1]), boundary[1]],
                    mode='lines+markers',
                    name='Decision Boundary',
                    marker=dict(color='green')
                ),
            )

            fig.update_layout(
                title=f"Decision boundary for {model.model_name}",
                xaxis_title=f"PCA Feature {feat_plot0}",
                yaxis_title=f"PCA Feature {feat_plot1}"
            )

            st.plotly_chart(fig)
    except ValueError as err:
        st.write({str(err)})

# Helper function
def restore_data_splits(df):
    """
    This function restores the training and validation/test datasets from the training page using st.session_state
                Note: if the datasets do not exist, re-split using the input df

    Inputs: 
        - df: the pandas dataframe
    Outputs: 
        - X_train: the training features
        - X_val: the validation/test features
        - y_train: the training targets
        - y_val: the validation/test targets
    """
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    try:
        # Restore train/test dataset
        if ('X_train' in st.session_state):
            X_train = st.session_state['X_train']
            y_train = st.session_state['y_train']
            st.write('Restored train data ...')
        if ('X_val' in st.session_state):
            X_val = st.session_state['X_val']
            y_val = st.session_state['y_val']
            st.write('Restored test data ...')
        if (X_train is None):
            # Select variable to explore
            numeric_columns = list(df.select_dtypes(include='number').columns)
            feature_select = st.selectbox(
                label='Select variable to predict',
                options=numeric_columns,
            )
            
            # Split train/test
            st.markdown(
                '### Enter the percentage of test data to use for training the model')
            number = st.number_input(
                label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

            X_train, X_val, y_train, y_val = split_dataset(df, number, feature_select, 'TF-IDF')
            st.write('Restored training and test data ...')
    except ValueError as err:
        st.write({str(err)})
    return X_train, X_val, y_train, y_val

###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:
    # Restore dataset splits
    X_train, X_val, y_train, y_val = restore_data_splits(df)

    st.markdown("## Get Performance Metrics")
    metric_options = ['precision', 'recall', 'accuracy']

    classification_methods_options = ['Logistic Regression',
                                      'Naive Bayes',
                                      'Support Vector Machine']

    trained_models = [
        model for model in classification_methods_options if model in st.session_state]
    st.session_state['trained_models'] = trained_models

    # Select a trained classification model for evaluation
    model_select = st.multiselect(
        label='Select trained classification models for evaluation',
        options=trained_models
    )
    if (model_select):
        st.write(
            'You selected the following models for evaluation: {}'.format(model_select))

        eval_button = st.button('Evaluate your selected classification models')

        if eval_button:
            st.session_state['eval_button_clicked'] = eval_button

        if 'eval_button_clicked' in st.session_state and st.session_state['eval_button_clicked']:
            st.markdown('## Review Classification Model Performance')

            plot_options = ['Metric Results', 'Decision Boundary']

            review_plot = st.multiselect(
                label='Select plot option(s)',
                options=plot_options
            )

            ############## Task 11: Plot ROC Curves
            if 'Metric Results' in review_plot:
                models = [st.session_state[model]
                          for model in model_select]

                train_result_dict = {}
                val_result_dict = {}

                # Select multiple metrics for evaluation
                metric_select = st.multiselect(
                    label='Select metrics for classification model evaluation',
                    options=metric_options,
                )
                if (metric_select):
                    st.session_state['metric_select'] = metric_select
                    st.write(
                        'You selected the following metrics: {}'.format(metric_select))

                    for idx, model in enumerate(models):
                        train_result_dict[model_select[idx]] = compute_eval_metrics(
                            X_train, y_train, model, metric_select, print_summary=True)
                        val_result_dict[model_select[idx]] = compute_eval_metrics(
                            X_val, y_val, model, metric_select, print_summary=True)

                    st.markdown('### Predictions on the training dataset')
                    st.dataframe(train_result_dict)

                    st.markdown('### Predictions on the validation dataset')
                    st.dataframe(val_result_dict)
        
            ############## Task 12: Plot Decision Boundary
            if 'Decision Boundary' in review_plot:
                models = [st.session_state[model] for model in model_select]
                plot_decision_boundary(np.array(X_train), np.ravel(y_train), models)