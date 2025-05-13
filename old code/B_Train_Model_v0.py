from ctypes import sizeof
import numpy as np                    
from sklearn.model_selection import train_test_split
import streamlit as st                  
import random
from helper_functions import fetch_dataset, set_pos_neg_reviews, reduce_feature_dimensionality
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed=10 to produce consistent results
random.seed(10)

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown(
    "### Homework 2 - Predicting Product Review Sentiment Using Classification")

#############################################

st.title('Train Model')

#############################################

# Checkpoint 4
def split_dataset_v0(df, number, target, feature_encoding, random_state=42):
    """
    Splits the dataset into training and test sets based on a given ratio
    Inputs:
        - df: DataFrame containing dataset
        - number: Percentage of data to be used for testing
        - target: Name of the target column ('rating')
        - feature_encoding: Feature encoding type ('Word Count' or 'TF-IDF')
        - random_state: Seed for reproducibility
    Outputs:
        - X_train_sentiment: Training feature set (encoded)
        - X_val_sentiment: Test feature set (encoded)
        - y_train: Training target labels
        - y_val: Test target labels
    """
    X_train = []
    X_val = []
    y_train = []
    y_val = []

    X_train_sentiment, X_val_sentiment = [], []
    try:
        # Add code here
        # st.write('split_dataset function has not been completed. Remove this statement upon completion.')
        y = df[target]  
        
        # Split data into training and test sets first
        X_train, X_val, y_train, y_val = train_test_split(
            df, y, test_size=number / 100, random_state=random_state
        )

        # Select features after splitting
        if feature_encoding == "TF-IDF":
            X_train_sentiment = X_train.loc[:, X_train.columns.str.startswith("tf_idf_word_count_")]
            X_val_sentiment = X_val.loc[:, X_val.columns.str.startswith("tf_idf_word_count_")]
        elif feature_encoding == "Word Count":
            X_train_sentiment = X_train.loc[:, X_train.columns.str.startswith("word_count_")]
            X_val_sentiment = X_val.loc[:, X_val.columns.str.startswith("word_count_")]
        else:
            raise ValueError("Invalid feature encoding type. Choose 'TF-IDF' or 'Word Count'.")


        # print("X_val_sentiment", X_val_sentiment)
        train_percentage = (len(X_train) /
                            (len(X_train)+len(X_val)))*100
        test_percentage = (len(X_val) /
                           (len(X_train)+len(X_val)))*100

        # Print dataset split result
        st.markdown('The training dataset contains {0:.2f} observations ({1:.2f}%) and the test dataset contains {2:.2f} observations ({3:.2f}%).'.format(len(X_train),
                                                                                                                                                          train_percentage,
                                                                                                                                                          len(
                                                                                                                                                              X_val),
                                                                                                                                                          test_percentage))
    except:
        print('Exception thrown; testing test size to 0')

    st.write(
            " First, we'll simplify the features using Principal Components Analysis (PCA). This will help identify important patterns in the features"
            " by projecting them to a smaller number of dimensions."
        )
    # Reduce feature dimensionality
    if(len(X_train_sentiment) and len(X_val_sentiment)):
        X_train_sentiment, X_val_sentiment = reduce_feature_dimensionality(X_train_sentiment, X_val_sentiment)
    return X_train_sentiment, X_val_sentiment, y_train, y_val

def split_dataset(df, number, target_name, features_name, random_state=42):
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=number, random_state=random_state)

    # st.write(features_name)
    # st.write(target_name)
    # st.write(df.columns)

    if features_name == 'All':
        features_name = df.columns[df.columns != target_name].tolist()

    X, y = df[features_name], df[target_name]
    train_idx, test_idx = next(sss.split(X, y))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values

    return X_train, X_test, y_train, y_test

# 3.1 LogisticRegression Class
class LogisticRegression(object):
    def __init__(self, learning_rate=0.001, num_iterations=500): 
        self.model_name = 'Logistic Regression'
        self.learning_rate = learning_rate 
        self.num_iterations = num_iterations 
        self.likelihood_history=[]

    # helper
    def stable_sigmoid(self, z):
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )
    
    # Checkpoint 5
    def predict_probability(self, X):
        """
        Produces probabilistic estimate for P(y_i = +1 | x_i, w)
        Inputs:
            - X: Input feature matrix
        Outputs:
            - y_pred: Probability of positive class (range: 0 to 1)
        """
        y_pred=None
        try:
            # Add code here
            # st.write('predict_probability function has not been completed. Remove this statement upon completion.')
            try:
                X = np.array(X, dtype=float)  # Force conversion to float
                z = np.dot(X, self.W) + self.b
                y_pred = self.stable_sigmoid(z)
            except Exception as err:
                st.write("Error in predict_probability:")
                st.write("X type:", type(X))
                st.write("X dtype:", X.dtype)
                st.write("W type:", type(self.W))
                st.write("W dtype:", self.W.dtype)
                st.write("Exception:", str(err))
                raise

            z = np.dot(X, self.W) + self.b 
            # y_pred = 1 / (1 + np.exp(-z))
            y_pred = self.stable_sigmoid(z)
        except ValueError as err:
            st.write({str(err)})
        return y_pred
    
    # Checkpoint 6
    def compute_avg_log_likelihood(self, X, Y, W):
        """
        Compute the average log-likelihood of logistic regression coefficients
        Inputs:
            - X: Feature subset
            - Y: Ground truth labels
            - W: Model weights
        Outputs:
            - lp: Log-likelihood estimate
        """
        lp=None
        try:
            # Add code here
            # indicator = (Y==+1)
            # scores = np.dot(X, W)

            # # logexp = np.log(1. + np.exp(-scores))
            # # # Simple check to prevent overflow
            # # mask = np.isinf(logexp)
            # # logexp[mask] = -scores[mask]

            # logexp = np.where(
            #     scores > 0,
            #     np.log(1 + np.exp(-scores)),
            #     -scores + np.log(1 + np.exp(scores))
            # )

            # lp = np.sum((indicator-1)*scores - logexp)/len(X)

            z = np.dot(X, W)
            p = self.stable_sigmoid(z)
            log_likelihood = Y * np.log(p + 1e-15) + (1 - Y) * np.log(1 - p + 1e-15)
            lp = np.mean(log_likelihood)

        except ValueError as err:
            st.write({str(err)})
        return lp
    
     # Checkpoint 7
    def update_weights(self):      
        """
        Compute the logistic regression derivative using 
            gradient ascent and update weights self.W and bias self.b
        Inputs:
            - None
        Outputs:
            - None
        """
        try:
            y_pred = self.predict(self.X)
            
            dW = (1/self.num_examples)*self.X.T.dot(self.Y - y_pred) # - 2 * 0.1 * self.W # added L2 regularization
            db = (1/self.num_examples)*np.sum(self.Y - y_pred)
            
            self.W += self.learning_rate * dW
            self.b += self.learning_rate * db
        except ValueError as err:
            st.write({str(err)})
        return self

    # Checkpoint 8
    def predict(self, X):
        """
        Predicts class labels based on logistic regression decision boundary using X, W, and b
        Inputs: 
            - X: Input feature matrix
        Outputs:
            - y_pred: List of predicted class labels (-1 or +1)
        """
        y_pred=None
        try:
            Z = self.predict_probability(X)
            y_pred = [-1 if z <= 0.5 else +1 for z in Z]
            # st.write('predict function has not been completed. Remove this statement upon completion.')
        except ValueError as err:
                st.write({str(err)})
        return y_pred 
    
    # Checkpoint 9
    def fit(self, X, Y):   
        """
        Initialize input features self.X, targets self.Y, weights self.W, biases self.b, and log likelihood self.likelihood_history
            Fits the logistic regression model to the data using gradient ascent
        Inputs:
            - X: Feature matrix
            - Y: Target sentiment labels
        Outputs:
            - self: Trained model instance
            - self.W: fitted weights
            - self.b: fitted bias
            - self.likelihood_history: history of log likelihood
        """    
        # Initialization: weights, features, target, bias, likelihood_history          
        self.X = X
        self.Y = Y
        self.b = 0
        self.likelihood_history=[]
        # Compute gradient ascent learning 
        try:
            # Add code here
            self.W = np.zeros(self.X.shape[1])
            self.num_examples = self.X.shape[0]
            for i in range(self.num_iterations):
                self.update_weights()
                lg = self.compute_avg_log_likelihood(self.X, self.Y, self.W)
                self.likelihood_history.append(lg)
        except ValueError as err:
                st.write({str(err)})
        return self

    # Helper Function
    def get_weights(self):
        """
        Retrieve and print the coefficients of the trained logistic regression model
        Inputs:
            - None
        Outputs:
            - weights: List of model parameters
        """
        weights=[]
        try:
            # set weight for given model_name
            W = np.array([f for f in self.W])
            # Print Coefficients
            st.write('-------------------------')
            st.write('Model Coefficients for '+ self.model_name)
            num_positive_weights = np.sum(W >= 0)
            num_negative_weights = np.sum(W < 0)
            st.write('* Number of positive weights: {}'.format(num_positive_weights))
            st.write('* Number of negative weights: {}'.format(num_negative_weights))
            weights = [self.W]
        except ValueError as err:
            st.write({str(err)})
        return weights
    
    # Helper Function
    def decision_boundary(self, feat_ids):
        """
        Compute decision boundary values where P(y_i = +1 | x_i) equals P(y_i = -1 | x_i)
        Inputs:
            - feat_ids: Array of feature indices to compute the decision boundary
        Outputs:
            - boundary: Array containing feature values corresponding to the decision boundary
        """
        boundary=[]
        try:
            # Extract relevant weight components
            W_ks = self.W[feat_ids]
            # Compute boundary values
            boundary = - self.b / W_ks
            # Handle edge cases
            boundary = np.nan_to_num(boundary, posinf=0, neginf=0)
        except ValueError as err:
            st.write({str(err)})
        return boundary

# 3.2 NaiveBayes Class
class NaiveBayes(object):
    def __init__(self, classes, alpha=1):
        self.model_name = 'Naive Bayes'
        self.classes = classes
        if not isinstance(self.classes, np.ndarray):
            self.classes = np.array(self.classes)
        self.num_classes = len(self.classes)
        mapping = {i: k for i, k in enumerate(self.classes)}
        self.idx_to_class = np.vectorize(mapping.get)
        self.likelihood_history=[]
        # smoothing parameter to avoid zero probability
        self.alpha = alpha
    
    # Checkpoint 10
    def predict_logprob(self, X):
        """
        Computes the log probability of each class given input features
        Inputs:
            - X: Input features
        Outputs:
            - y_pred: log probability of positive product review
        """
        y_pred=None
        try:
            # Add code here
            y_pred = np.dot(X, np.log(self.W.T))
            y_pred += np.log(self.W_prior)
        except ValueError as err:
            st.write({str(err)})
        return y_pred
    
    # Checkpoint 11
    def predict_probability(self, X):
        """
        Produces probabilistic estimate for P(y_i = +1 | x_i)
        Inputs:
            - X: Input features
        Outputs:
            - y_pred: probability of positive product review (range: 0 to 1)
        """
        y_pred=None
        try:
            # Add code here
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            # Identify index for positive class
            pos_class_idx = np.where(self.classes == 1)[0][0]
            # Get log probabilities
            y_pred = self.predict_logprob(X)
            # Convert log probabilities to probabilities using the softmax function
            probs = np.exp(np.array(y_pred))
            probs = np.exp(probs) / np.sum(np.exp(probs),axis=1)[:, None]
            # Extract probability of positive class
            y_pred = probs[:, pos_class_idx]

        except ValueError as err:
            st.write({str(err)})
        return y_pred

    # Checkpoint 12
    def predict(self, X):
        """
        Predicts the class label for each input sample
        Inputs: 
            - X: Input features
        Outputs:
            - y_pred: List of predicted class labels
        """
        y_pred=None
        try:
            y_pred = self.predict_logprob(X)
            y_pred = np.argmax(y_pred, axis=1)
            mapping = {i: k for i, k in enumerate(self.classes)}
            idx_to_class = np.vectorize(mapping.get)
            y_pred = idx_to_class(y_pred)
        except ValueError as err:
            st.write({str(err)})
        return y_pred
    
    # Checkpoint 13
    def fit(self, X, Y):
        """
        Initialize self.num_examples, self.num_features, weights self.W, prior probabilities self.W_prior 
            Fits the Naive Bayes classifier using a closed-form solution
        Inputs: 
            - X: Input features
            - Y: list of actual product sentiment classes 
        Outputs:
            - self: The trained Naive Bayes model
            - self.W: fitted model weights
            - self.likelihood_history: history of log likelihood
        """
        try:
                # Number of examples, Number of features
            num_examples, num_features = X.shape
            classes = np.unique(np.ravel(Y))
            num_classes = len(classes)
            # Initialization: weights, prior, likelihood
            self.W = np.zeros((num_classes, num_features))
            self.W_prior = np.zeros(num_classes)
            self.likelihood_history=[]

            # closed-form solution for model parameters
            # Compute class-conditional probabilities and class priors
            for ind, class_k in enumerate(classes):
                # Select samples belonging to class_k
                X_class_k = X[Y == class_k]
                # Compute likelihood using Laplace smoothing
                self.W[ind] = (np.sum(X_class_k, axis=0) + self.alpha)
                self.W[ind] /= (np.sum(X_class_k) + (self.alpha * X_class_k.shape[-1]))
                # Compute prior
                self.W_prior[ind] = X_class_k.shape[0] / num_examples
            
            # Compute and store log likelihood history
            log_likelihood = np.log(self.predict_probability(X)).mean()
            self.likelihood_history.append(log_likelihood)
        except ValueError as err:
            st.write({str(err)})
        return self

    # Helper Function
    def get_weights(self):
        """
        Retrieves and prints the trained naive bayes model coefficients
        Inputs:
            - None
        Outputs:
            - weights: List of model parameters
        """
        weights=None
        try:
            # set weight for given model_name
            # Print Coefficients
            st.write('-------------------------')
            st.write('Model Coefficients for '+ self.model_name)
            num_positive_weights = np.sum(self.W >= 0) + np.sum(self.W_prior >= 0)
            num_negative_weights = np.sum(self.W < 0) + np.sum(self.W_prior < 0)
            st.write('* Number of positive weights: {}'.format(num_positive_weights))
            st.write('* Number of negative weights: {}'.format(num_negative_weights))
            weights = [self.W, self.W_prior]
        except ValueError as err:
            st.write({str(err)})
        return weights

    # Helper Function
    def decision_boundary(self, feat_ids):
        """
        Compute decision boundary values where P(y_i = +1 | x_i) equals P(y_i = -1 | x_i).
        Inputs:
            - feat_ids: Array of feature indices to compute the decision boundary.
        Outputs:
            - boundary: Array containing feature values corresponding to the decision boundary.
        """
        boundary=None
        try:
            # Extract relevant weight components
            W_ks = self.W[:, feat_ids]
            W_prior_ks = self.W_prior
            # Compute boundary values
            boundary = np.log(W_prior_ks[0]) - np.log(W_prior_ks[1])
            boundary /= np.log(W_ks[1]) - np.log(W_ks[0])
            # Handle edge cases
            boundary = np.nan_to_num(boundary, posinf=0, neginf=0)
        except ValueError as err:
            st.write({str(err)})
        return boundary

# 3.3 Support Vector Machine (SVM) Class
class SVM(object):
    def __init__(self, learning_rate=0.001, num_iterations=500, lambda_param=0.01):
        self.model_name = 'Support Vector Machine'
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.likelihood_history = []
    
    # Checkpoint 14
    def predict_score(self, X):
        """
        Produces raw decision values before thresholding
        Inputs:
            - X: Input features
        Outputs:
            - scores: Raw SVM decision values
        """
        scores=None
        try:
            # Add code here
            scores = X.dot(self.W) + self.b
        except ValueError as err:
            st.write({str(err)})
        return scores
    
    # Checkpoint 15
    def compute_hinge_loss(self, X, Y):
        """
        Compute the hinge loss for SVM using X, Y, and self.W
        Inputs:
            - X: Input features
            - Y: Ground truth labels
        Outputs:
            - loss: Computed hinge loss
        """
        loss=None
        try:
            margins = 1 - Y * self.predict_score(X)
            margins = np.maximum(0, margins)
            loss = np.mean(margins) + (self.lambda_param / 2) * np.sum(self.W ** 2)
        except ValueError as err:
            st.write({str(err)})
        return loss
    
    # Checkpoint 16
    def update_weights(self):
        """
        Compute SVM derivative using gradient descent and update weights
        Inputs:
            - None
        Outputs:
            - self: The trained SVM model
            - self.W: Weight vector updated based on gradient descent
            - self.b: Bias term updated based on gradient descent
            - self.likelihood_history: history of log likelihood
        """
        try:
            # Add code here
            scores = self.predict_score(self.X)
            indicator = (self.Y * scores) < 1
            
            num_examples = self.X.shape[0]
            dW = (-np.dot(self.X.T, (self.Y * indicator)) + 2 * self.lambda_param * self.W ) / num_examples
            db = -np.sum(self.Y * indicator) / num_examples
            
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            
            loss = -self.compute_hinge_loss(self.X, self.Y)
            self.likelihood_history.append(loss)
        except ValueError as err:
            st.write({str(err)})
        return self
    
    # Checkpoint 17
    def predict(self, X):
        """
        Predicts class labels using the trained SVM model
        Inputs:
            - X: Input features
        Outputs:
            - y_pred: List of predicted classes (-1 or +1)
        """
        y_pred=None
        try:
            # Add code here
            scores = self.predict_score(X)
            y_pred = np.where(scores >= 0, +1, -1)
        except ValueError as err:
            st.write({str(err)})
        return y_pred
    
    # Checkpoint 18
    def fit(self, X, Y):
        """
        Train SVM using gradient descent.
        Inputs:
            - X: Input features
            - Y: True class labels (-1 or +1)
        Outputs:
            - self: Trained SVM model
        """
        try:
            # Add code here

            self.X = X
            self.Y = Y

            self.W = np.zeros(X.shape[1])
            self.b = 0
            self.loss_history = []

            for i in range(self.num_iterations):
                self.update_weights()
                loss = self.compute_hinge_loss(X, Y)
                self.loss_history.append(-loss)

            st.write('fit function has not been completed. Remove this statement upon completion.')
        except ValueError as err:
            st.write({str(err)})
        return self
    
    # Helper Function
    def get_weights(self):
        """
        Retrieve and print the coefficients of the trained SVM model
        Inputs:
            - None
        Outputs:
            - weights: List of model parameters
        """
        try:
            # Print Coefficients
            st.write('-------------------------')
            st.write('Model Coefficients for ' + self.model_name)
            num_positive_weights = np.sum(self.W >= 0)
            num_negative_weights = np.sum(self.W < 0)
            st.write('* Number of positive weights: {}'.format(num_positive_weights))
            st.write('* Number of negative weights: {}'.format(num_negative_weights))
            weights = [self.W]
        except ValueError as err:
            st.write({str(err)})
        return weights

    # Helper Function
    def decision_boundary(self, feat_ids):
        """
        Compute decision boundary values where P(y_i = +1 | x_i) equals P(y_i = -1 | x_i).
        Inputs:
            - feat_ids: Array of feature indices to compute the decision boundary.
        Outputs:
            - boundary: Array containing feature values corresponding to the decision boundary.
        """
        boundary=[]
        try:
            # Extract relevant weight components
            W_ks = self.W[feat_ids]
            # Compute boundary values
            boundary = - self.b / W_ks
            # Handle edge cases
            boundary = np.nan_to_num(boundary, posinf=0, neginf=0)
        except ValueError as err:
            st.write({str(err)})
        return boundary

###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:

    # Display dataframe as table
    st.dataframe(df)

    # Select variable to predict
    feature_predict_select = 'Default'

    st.session_state['target'] = feature_predict_select

    # drop down menu of df coluns
    feature_input_select = st.selectbox(
        label='Select X features to predict whether mortgage loan defaults (Default)',
        options=['All'] + list(df.columns),
        index=0,
        key='feature_input_select'
    )

    st.session_state['feature'] = feature_input_select

    # Task 4: Split train/test
    st.markdown('## Split dataset into Train/Test sets')
    st.markdown(
        '### Enter the percentage of test data to use for training the model')
    number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

    X_train, X_val, y_train, y_val = [], [], [], []
    # Compute the percentage of test and training data
    if (feature_predict_select in df.columns):
        X_train, X_val, y_train, y_val = split_dataset(
            df, number, feature_predict_select, feature_input_select)
        
        # Save train and test split to st.session_state
        st.session_state['X_train'] = X_train
        st.session_state['X_val'] = X_val
        st.session_state['y_train'] = y_train
        st.session_state['y_val'] = y_val

    classification_methods_options = ['Logistic Regression',
                                      'Naive Bayes', 
                                      'Support Vector Machine']
    
    # Collect ML Models of interests
    classification_model_select = st.multiselect(
        label='Select regression model for prediction',
        options=classification_methods_options,
    )
    st.write('You selected the follow models: {}'.format(
        classification_model_select))

    # Add parameter options to each regression method

    # Task 5: Logistic Regression
    if (classification_methods_options[0] in classification_model_select):
        st.markdown('#### ' + classification_methods_options[0])

        lg_col1, lg_col2 = st.columns(2)

        with (lg_col1):
            lg_learning_rate_input = st.text_input(
                label='Input learning rate 👇',
                value='0.0001',
                key='lg_learning_rate_textinput'
            )
            st.write('You select the following learning rate value(s): {}'.format(lg_learning_rate_input))

        with (lg_col2):
            # Maximum iterations to run the LG until convergence
            lg_num_iterations = st.number_input(
                label='Enter the number of maximum iterations on training data',
                min_value=1,
                max_value=5000,
                value=500,
                step=100,
                key='lg_max_iter_numberinput'
            )
            st.write('You set the maximum iterations to: {}'.format(lg_num_iterations))

        lg_params = {
            'num_iterations': lg_num_iterations,
            'learning_rate': [float(val) for val in lg_learning_rate_input.split(',')],
        }
        if st.button('Logistic Regression Model'):
            try:
                lg_model = LogisticRegression(num_iterations=lg_params['num_iterations'], 
                                            learning_rate=lg_params['learning_rate'][0])
                lg_model.fit(X_train.to_numpy(), np.ravel(y_train))
                # lg_model.fit(X_train, np.ravel(y_train))
                st.session_state[classification_methods_options[0]] = lg_model
            except ValueError as err:
                st.write({str(err)})
        
        if classification_methods_options[0] not in st.session_state:
            st.write('Logistic Regression Model is untrained')
        else:
            st.write('Logistic Regression Model trained')

    # Task 6: Naive Bayes
    if (classification_methods_options[1] in classification_model_select):
        st.markdown('#### ' + classification_methods_options[1])
        if st.button('Naive Bayes'):
            try:
                nb_model = NaiveBayes(np.unique(np.ravel(y_train)))
                #nb_model.fit(X_train.to_numpy(), np.ravel(y_train))
                nb_model.fit(X_train, np.ravel(y_train))
                st.session_state[classification_methods_options[1]] = nb_model
            except ValueError as err:
                st.write({str(err)})
        
        if classification_methods_options[1] not in st.session_state:
            st.write('Naive Bayes Model is untrained')
        else:
            st.write('Naive Bayes Model trained')

    # Task 7: Support Vector Machine
    if (classification_methods_options[2] in classification_model_select):
        st.markdown('#### Support Vector Machine')

        svm_col1, svm_col2 = st.columns(2)

        with svm_col1:
            svm_learning_rate_input = st.text_input(
                label='Input learning rate 👇',
                value='0.001',
                key='svm_learning_rate_textinput'
            )
            st.write('You selected the following learning rate value(s): {}'.format(svm_learning_rate_input))

        with svm_col2:
            svm_num_iterations = st.number_input(
                label='Enter the number of maximum iterations on training data',
                min_value=1,
                max_value=5000,
                value=500,
                step=100,
                key='svm_max_iter_numberinput'
            )
            st.write('You set the maximum iterations to: {}'.format(svm_num_iterations))

        svm_params = {
            'num_iterations': svm_num_iterations,
            'learning_rate': [float(val) for val in svm_learning_rate_input.split(',')],
        }

        if st.button('Train SVM Model'):
            try:
                svm_model = SVM(num_iterations=svm_params['num_iterations'], 
                                learning_rate=svm_params['learning_rate'][0])
                svm_model.fit(X_train, np.ravel(y_train))
                st.session_state[classification_methods_options[2]] = svm_model
            except ValueError as err:
                st.write({str(err)})

        if 'Support Vector Machine' not in st.session_state:
            st.write('SVM Model is untrained')
        else:
            st.write('SVM Model trained')
    
    # Store models in dict
    trained_models={}
    for model_name in classification_methods_options:
        if(model_name in st.session_state):
            trained_models[model_name] = st.session_state[model_name]


    # Task 9: Inspect classification coefficients
    st.markdown('## Inspect model coefficients')

    # Select multiple models to inspect
    inspect_models = st.multiselect(
        label='Select features for classification input',
        options=trained_models,
        key='inspect_multiselect'
    )
    st.write('You selected the {} models'.format(inspect_models))

    models = {}
    weights_dict = {}
    if(inspect_models):
        for model_name in inspect_models:
            if model_name in trained_models:
                weights_dict[model_name] = trained_models[model_name].get_weights()

    # Inspect model likelihood
    st.markdown('## Inspect model likelihood')

    # Select multiple models to inspect
    inspect_model_likelihood = st.selectbox(
        label='Select model',
        options=trained_models,
        key='inspect_cost_multiselect'
    )

    st.write('You selected the {} model'.format(inspect_model_likelihood))

    if inspect_model_likelihood and inspect_model_likelihood in trained_models:
        try:
            fig = make_subplots(rows=1, cols=1,
                shared_xaxes=True, vertical_spacing=0.1)
            cost_history=trained_models[inspect_model_likelihood].likelihood_history

            x_range = st.slider("Select x range:",
                                    value=(0, len(cost_history)))
            st.write("You selected : %d - %d"%(x_range[0],x_range[1]))
            cost_history_tmp = cost_history[x_range[0]:x_range[1]]
            
            fig.add_trace(go.Line(x=np.arange(x_range[0],x_range[1],1),
                        y=cost_history_tmp, mode='lines+markers', name=inspect_model_likelihood), row=1, col=1)

            fig.update_xaxes(title_text="Training Iterations")
            fig.update_yaxes(title_text='Log Likelihood', row=1, col=1)
            fig.update_layout(title=inspect_model_likelihood)
            st.plotly_chart(fig)
        except Exception as e:
            print(e)

    st.write('Continue to Test Model')