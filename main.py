"""
We need to determine with ML whether given patient has ASD or not.
ASD - Autism Spectrum Disorder (Autyzm)
What is ASD?
Autism spectrum disorder (ASD) is a developmental disorder that affects communication and behavior.
People with ASD may repeat certain behaviors, have difficulty with social interactions, and have unusual
responses to sensory experiences.

We have 19 features and 1 target column. We need to predict target column.
We need to use ML to predict target column.
But first lets check if we have any missing values in all boolean, discrete and categorical columns.
"""

import joblib
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from helper import *

# widen the display
pd.set_option('display.max_columns', 100)


def remove_punctuation(inDF, columns):
    """
    Remove punctuation from columns in DataFrame
    :param inDF: dataframe
    :param columns: list of columns
    :return: dataframe
    """
    regex_pattern = r"[^\w\s]"
    for c in columns:
        inDF[c] = inDF[c].str.replace(regex_pattern, "").str.strip().str.lower()

    return inDF


def check_missing_values(inDF):
    """
    Check if there is any missing values in columns and plot them
    :param inDF: dataframe
    :return: None
    """
    null_df = inDF.isnull().sum().reset_index()
    null_df.columns = ["features", "null counts"]
    # now plot it
    fig = px.bar(null_df, x="features", y="null counts")
    # change the orientation of complete plot
    fig.update_layout(barmode="group", xaxis={"categoryorder": "total descending"},
                      title="Null count in each feature",
                      width=700, height=300, titlefont={"size": 14},
                      font_family="Courier New",
                      title_x=0.5, showlegend=False)
    # fig.show()
    # Save the plot as html
    fig.write_html("plots/null_count.html")


def fill_missing_values(inDF):
    """
    Fill missing values in columns
    :param inDF: dataframe
    :return: dataframe
    """
    procDF = inDF.copy()

    # Fill missing values in columns
    # now we need to impute the null values in relation and Ethnicity column
    procDF['relation'].fillna('others', inplace=True)
    procDF['ethnicity'].fillna('others', inplace=True)

    # Fill outliers in age column
    temp_df = procDF.copy()
    procDF.loc[procDF['age'] > 100, 'age'] = procDF['age'].median()
    # replace missing values in age column with median
    procDF['age'] = procDF['age'].fillna(procDF['age'].median())

    # Plot the box plot to check outliers
    fig = go.Figure(data=[go.Box(y=temp_df['age'], name="Before imputation"),
                          go.Box(y=procDF['age'], name="After imputation")])
    # fig.show()
    #  reduce font size of title
    # move title to the bottom
    fig.update_layout(title={"text": "Age feature before and after imputation with Median",
                             "x"   : 0.5},
                      titlefont={"size": 14},
                      width=500, height=300,
                      font_family="Courier New",
                      showlegend=False)
    # Save the plot as html
    fig.write_html("plots/age_boxplot.html")

    # Now lets check the distribution of age column with ASD
    fig = px.histogram(procDF, x="age", color="asd", marginal="rug",
                       hover_data=procDF.columns)

    fig.update_layout(title={"text": "Age feature distribution with ASD",
                             "x"   : 0.5},
                      titlefont={"size": 14},
                      width=800, height=300,
                      font_family="Courier New",
                      showlegend=True)
    # move legend to the top
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    # Save the plot as html
    fig.write_html("plots/age_distribution.html")

    # Now let us replace missing value for 'family', 'used_app_before' and 'gender' columns with
    # 'not sure', 'nan' and 'unknown' respectively
    procDF = procDF.fillna({'family'         : 'not sure',
                            'used_app_before': 'na',
                            'gender'         : 'unknown',
                            'age_desc'       : 'na',
                            'country_of_res' : 'others',
                            'jaundice'       : 'na'})

    return procDF


def convert_to_integer(inDF):
    """
    Convert column to integer
    :param inDF: dataframe
    :return: dataframe
    """
    for col in inDF.columns:
        if col.endswith("_score"):
            inDF[col] = inDF[col].astype("bool").astype("int")

    return inDF


def load_and_preprocess_data(filename, save_plots=True):
    """
    Load data and preprocess it
    :return: dataframe
    """
    inDF = pd.read_csv(filename)

    # lower all column names
    inDF.columns = inDF.columns.str.lower()

    # change column name jundice to jaundice
    inDF = inDF.rename(columns={"jundice": "jaundice"})

    # Print shape and data types
    print("Shape of data: {}".format(inDF.shape))
    # print(df.dtypes)

    # Convert columns to integer
    procDF = convert_to_integer(inDF)

    # Remove punctuation from columns
    cols = ['country_of_res', 'ethnicity', 'relation']
    procDF = remove_punctuation(procDF, cols)

    # Check for missing values
    if save_plots:
        check_missing_values(procDF)

    # Fill missing values
    outDF = fill_missing_values(procDF)

    # drop duplicate rows
    outDF = outDF.drop_duplicates()

    return outDF


# Model building
def encode_data_and_split(inDF, target_col, split=True, test_size=0.3, seed=42):
    """
    Encode data
    :param inDF: dataframe
    :return: X, y
    """
    # drop duplicate rows
    procDF = inDF.drop_duplicates()

    cols_to_be_encoded = ['ethnicity', 'relation', 'country_of_res', 'used_app_before',
                          'family', 'gender', 'jaundice']
    cols_to_be_drop = ['age_desc']

    # Drop unwanted columns like age_desc
    procDF = procDF.drop(cols_to_be_drop, axis=1)
    procDF[target_col] = procDF[target_col].map({'YES': 1, 'NO': 0})

    if split:
        # Label encode the categorical columns
        for i, col in enumerate(cols_to_be_encoded):
            le = LabelEncoder()
            procDF[col] = le.fit_transform(procDF[col])

            # save the label encoder
            joblib.dump(le, "models/encoder/label_encoder_{}.pkl".format(i))
    else:
        # Load the label encoder
        for i, col in enumerate(cols_to_be_encoded):
            le = joblib.load("models/encoder/label_encoder_{}.pkl".format(i))
            procDF[col] = le.transform(procDF[col])

    # Split data into X and y
    X = procDF.drop(target_col, axis=1)
    y = procDF[target_col]

    if split:
        # Split the data into train and test
        return train_test_split(X, y, test_size=test_size,
                                random_state=seed), \
            X.columns
    else:
        return X, y, X.columns


def plot_shap_values(shap_values, X, feature_names, class_names, figure_name):
    """
    Plot shap values
    :param shap_values: shap values
    :param X: X
    :param feature_names: feature names
    :param model_type: model type
    :return: None
    """
    # compute the shap values for every feature along with the expected value and class names
    # plot the shap values
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      class_names=class_names, show=False)
    # Save the plot as html
    plt.savefig(figure_name, dpi=500, bbox_inches='tight')
    plt.close()


def train_model(inDF):
    """
    Train model
    :param inDF: dataframe
    :return: model
    """
    # Now let us encode the data and split it into train and test
    (X_train, X_test, y_train, y_test), feature_names = encode_data_and_split(inDF, 'asd')

    # defining parameter range from SVC
    param_grid = {'C'     : [0.1, 1, 10, 100, 1000],
                  'gamma' : [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'poly', 'sigmoid']}

    # create grid search object
    svc_grid = GridSearchCV(SVC(), param_grid, cv=5, refit=True, verbose=3)

    # fit the model for grid search
    svc_grid.fit(X_train, y_train)

    # print best parameter after tuning
    print("Best parameters from SVC: {}".format(svc_grid.best_params_))

    # print how our model looks after hyper-parameter tuning
    print("Best Estimator from SVC: {}".format(svc_grid.best_estimator_))

    # print the score
    print("Best score from SVC: {}".format(svc_grid.score(X_test, y_test)))

    # save the model
    joblib.dump(svc_grid, "models/svc_model.pkl")

    # Now lets build a model of Logistic Regression with pipeline and GridSearchCV
    # Create a pipeline including all the functions we need
    pipe = Pipeline([('lr', LogisticRegression())])

    # Create a grid search object
    grid = {'lr__C'      : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'lr__penalty': ['l1', 'l2'],
            'lr__solver' : ['liblinear', 'saga']}

    # Create a gridsearchcv object
    gridsearch = GridSearchCV(pipe, grid, cv=5, scoring='roc_auc', n_jobs=-1)

    # Fit the model
    gridsearch.fit(X_train, y_train)

    # Get the best parameters
    print(f"Best parameters from Logistic regression: {gridsearch.best_params_}")

    # Get the best score
    print(f"Best score from Logistic regression: {gridsearch.best_score_}")

    # Get the best estimator
    print(f"Best estimator from Logistic Regression: {gridsearch.best_estimator_}")

    # Save the best estimator
    joblib.dump(gridsearch.best_estimator_, 'models/ASD_model.pkl')

    # store all results in a dataframe from both svc and logistic regression
    results = pd.DataFrame({'Model'          : ['SVC', 'Logistic Regression'],
                            'Score'          : [svc_grid.score(X_test, y_test),
                                                gridsearch.score(X_test, y_test)],
                            'Best Parameters': [svc_grid.best_params_,
                                                gridsearch.best_params_]})

    # save the results
    results.to_csv('models/results.csv', index=False)

    # plot the results for both training set and test set
    plotly_obj.plot_results(svc_grid.best_estimator_,
                            X_train, X_test, y_train, y_test, 'SVC')
    plotly_obj.plot_results(gridsearch.best_estimator_,
                            X_train, X_test, y_train, y_test, 'Logistic Regression')

    return X_test, y_test, feature_names


def predict_and_evaluate_model(X, y, feature_names, model_type='SVC'):
    # Load the model
    if model_type == 'SVC':
        model = joblib.load("models/svc_model.pkl")
    else:
        model = joblib.load("models/ASD_model.pkl")

    # Get the predictions
    y_pred = model.predict(X)

    # plot the confusion matrix with plotly
    plotly_obj.plot_confusion_matrix(y, y_pred, title="Confusion Matrix from {}".format(model_type),
                                     filename="plots/confusion_matrix_{}.html".format(model_type))

    # Plot ROC curve
    plotly_obj.plot_roc_curve(y, y_pred,
                              "ROC curve for {} model".format(model_type),
                              "plots/{}_model_ROC_curve.html".format(model_type))

    # Get the classification report
    # print(classification_report(y, y_pred))
    print("Accuracy of {} on Test Set: {}".format(model_type,
                                                  accuracy_score(y, y_pred)))

    if model_type != 'SVC':
        # Get the feature importance from logistic regression
        feature_importance = model.named_steps['lr'].coef_[0]

        # Create a dataframe
        feature_importance_df = pd.DataFrame({'feature'   : feature_names,
                                              'importance': feature_importance})

        # Sort the dataframe by importance
        feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)

        # Plot the feature importance
        fig = px.bar(feature_importance_df, x='feature', y='importance',
                     title="Feature Importance", color='importance',
                     color_continuous_scale=px.colors.sequential.RdBu)
        fig.update_layout(xaxis={'categoryorder': 'total descending'},
                          yaxis_title="Importance", xaxis_title="Features",
                          template='plotly_dark',
                          title_font_size=14, title_font_family='Rockwell',
                          height=400, width=600)

        # fig.show()

        # Save the plot as html
        fig.write_html("plots/feature_importance_from_{}.html".format(model_type))

        # lets us use shap to explain the model and class names
        explainer = shap.LinearExplainer(model.named_steps['lr'], X,
                                         feature_perturbation="dependent")
        shap_values = explainer.shap_values(X)

        # get the class names
        class_names = model.classes_

        # plot the shap values with plotly
        plot_shap_values(shap_values, X, feature_names, class_names,
                         "plots/{}_model_SHAP_values.png".format(model_type))


def validate_model(filename, model_name="models/ASD_model.pkl"):
    """
    We will use the model to predict the values for the unseen data
    :return: accuracy
    """
    # load the unseen data
    unseen_data = load_and_preprocess_data(filename, save_plots=False)

    # encode the unseen data
    X_unseen, y_unseen, feature_names = encode_data_and_split(unseen_data, 'asd', split=False)

    # load the model
    model = joblib.load(model_name)

    # get the predictions
    y_pred = model.predict(X_unseen)

    # get the accuracy
    accuracy = accuracy_score(y_unseen, y_pred)

    return accuracy


# start with main function
if __name__ == '__main__':
    # Load data and preprocess it
    df = load_and_preprocess_data("data/Autism-Adult-Data.csv")

    # Create object of PlotlyHelper class
    plotly_obj = PlotlyHelper(df)

    # now let us plot whether ASD is frequent in any country
    plotly_obj.plot_stacked_histogram("country_of_res", "asd",
                                      "ASD is frequent in any country",
                                      "plots/ASD_frequent_in_country.html")

    # Lets check with which ethnicity ASD is more frequent
    plotly_obj.plot_stacked_histogram("ethnicity", "asd",
                                      "ASD against ethnicity",
                                      "plots/ASD_against_ethnicity.html")

    # Lets check with Jaundice is more frequent in any country
    plotly_obj.plot_stacked_histogram("jaundice", "asd",
                                      "ASD against jaundice",
                                      "plots/ASD_against_jaundice.html")

    # let us plot jaundice vs age with patients with ASD
    fig = px.histogram(df[df['asd'] == 'YES'], x='age', color='jaundice',
                          title="Jaundice vs Age for ASD patients",
                            color_discrete_map={'yes': 'red', 'no': 'blue'},
                            template='plotly_white',
                            height=400, width=750)
    fig.update_layout(xaxis_title="Age", yaxis_title="Count",
                      title_font_size=14, font_family='Courier New, monospace',
                      title_x=0.5)

    # move legend to the top
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    # fig.show()
    fig.write_html("plots/jaundice_vs_age_with_ASD.html")

    # Lets get count of patients country wise
    patients_country_wise = pd.DataFrame(df[df['asd'] == 'YES']['country_of_res'].value_counts()).rename(
        {'country_of_res':
             'patient_counts'},
        axis=1)
    plotly_obj.plot_pie(patients_country_wise['patient_counts'],
                        patients_country_wise.index,
                        "Country wise ASD patients",
                        "plots/country_wise_ASD_patients.html")

    # Lets create a model
    X_test, y_test, features = train_model(df)

    # test the model
    predict_and_evaluate_model(X_test, y_test, features)

    # check the model on unseen data
    accuracy = validate_model(filename="data/unseen_data.csv", model_name="models/svc_model.pkl")
    print(f"Accuracy of the model on unseen data: {accuracy}")
