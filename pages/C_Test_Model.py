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

    except ValueError as err:
        st.write({str(err)})

    return precision, recall

def compute_eval_metrics(X, y_true, model):
    y_pred = model.predict(X)

    precision, recall = compute_precison_recall(y_pred, y_true)
    accuracy = compute_accuracy(y_pred, y_true)

    st.write(f"Precision: {precision:.3f}")
    st.write(f"Recall:    {recall:.3f}")
    st.write(f"Accuracy:  {accuracy:.3f}")

    return precision, recall, accuracy


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
        if 'X_train' in st.session_state:
            X_train = st.session_state['X_train']
            y_train = st.session_state['y_train']
            st.write('Restored train data ...')
        if 'X_val' in st.session_state:
            X_val = st.session_state['X_val']
            y_val = st.session_state['y_val']
            st.write('Restored test data ...')

        if X_train is None or X_val is None:

            # Define sensible defaults
            number = 30  # default test size percent
            random_state = 42
            target_name = 'Default'
            features_name = 'All'

            from sklearn.model_selection import StratifiedShuffleSplit

            if features_name == 'All':
                features_name = df.columns[df.columns != target_name].tolist()

            X, y = df[features_name], df[target_name]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=number / 100, random_state=random_state)
            train_idx, test_idx = next(sss.split(X, y))

            X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_val = y.iloc[train_idx].values, y.iloc[test_idx].values

            # convert to numpy arrays
            X_train = np.array(X_train)
            X_val = np.array(X_val)
            y_train = np.array(y_train)
            y_val = np.array(y_val)

            st.dataframe(df)

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
                                      'Support Vector Machine',
                                      'Artificial Neural Network']

    trained_models = [
        model for model in classification_methods_options if model in st.session_state]
    st.session_state['trained_models'] = trained_models

    if len(trained_models) == 0:
        st.write('No trained classification models found. Please go back to B Train Model.')

    # Select a trained classification model for evaluation
    model_select = st.multiselect(
        label='Select trained classification models for evaluation',
        options=trained_models
    )
    if (model_select):
        model_name = model_select[0]
        st.write('You selected the following models for evaluation: {}'.format(model_name))

        # Metrics
        model = st.session_state[model_name]
        st.markdown('**Train Metrics**')
        compute_eval_metrics(X_train, y_train, model)

        st.markdown('**Test Metrics**')
        compute_eval_metrics(X_val, y_val, model)

        # # Decision Boundary
        # models = [st.session_state[model] for model in model_select]
        # plot_decision_boundary(np.array(X_train), np.ravel(y_train), models)