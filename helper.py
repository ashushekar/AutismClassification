import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


# Let us write plotly class function for all plots
class PlotlyHelper:
    def __init__(self, inDF):
        self.inDF = inDF

    def plot_bar(self, x, y, title, filename):
        """
        Plot bar plot
        :param x: x column
        :param y: y column
        :param title: title of plot
        :param filename: filename to save plot
        :return: None
        """
        fig = px.bar(self.inDF, x=x, y=y)
        fig.update_layout(title=title, title_x=0.5,
                          width=700, height=400,
                          font_family="Courier New")
        # fig.show()
        # Save the plot as html
        fig.write_html(filename)

    def plot_stacked_histogram(self, x, y, title, filename):
        """
        Plot stacked histogram
        :param x: x column
        :param y: y column
        :param title: title of plot
        :param filename: filename to save plot
        :return: None
        """
        fig = px.histogram(self.inDF, x=x, color=y, barmode="stack")
        # remove xaxis label
        fig.update_xaxes(title_text="")
        fig.update_layout(xaxis={"categoryorder": "total descending"},
                          title=title, title_x=0.5,
                          width=700, height=350,
                          font_family="Courier New")
        # move legend to the top
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        # fig.show()
        # Save the plot as html
        fig.write_html(filename)

    def plot_pie(self, values, names, title, filename):
        """
        Plot pie chart
        :param values: values column
        :param names: names column
        :param title: title of plot
        :param filename: filename to save plot
        :return: None
        """
        fig = go.Figure(data=[go.Pie(labels=names, values=values,
                                     rotation=90, hole=.3)])
        fig.update_traces(textposition='inside', textinfo='percent+label',
                          marker=dict(colors=px.colors.sequential.RdBu,
                                      line=dict(color='#000000', width=2)))
        fig.update_layout(title=title, title_x=0.5, title_font=dict(size=14),
                          template="plotly_white", font_family="Courier New",
                          width=600, height=600, showlegend=False)
        # fig.show()
        # Save the plot as html
        fig.write_html(filename)

    # plot AUC-ROC curve for the model on test set with plotly
    def plot_roc_curve(self, y_test, y_pred, title, filename):
        """
        Plot ROC curve for the model on test set and print AUC score with plotly
        :param y_test: test labels
        :param y_pred: predicted labels
        :param title: title of plot
        :param filename: filename to save plot
        :return: None
        """
        # calculate AUC
        auc = roc_auc_score(y_test, y_pred)
        print('AUC: %.3f' % auc)
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # plot the roc curve for the model
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                 mode='lines',
                                 name='ROC'))
        # Show AUC score in plot
        fig.add_annotation(x=0.5, y=0.5,
                           text="AUC: {}".format(auc),
                           showarrow=False,
                           font=dict(size=12))
        fig.update_layout(title=title, title_x=0.5,
                          title_font=dict(size=13),
                          font_family="Courier New",
                          width=600, height=600)
        # axis labels
        fig.update_xaxes(title_text='False Positive Rate')
        fig.update_yaxes(title_text='True Positive Rate')

        # show the plot
        # fig.show()
        # Save the plot as html
        fig.write_html(filename)

    def plot_confusion_matrix(self, y_test, y_pred, title, filename):
        # Get the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = ff.create_annotated_heatmap(z=cm, x=['Predicted NO', 'Predicted YES'],
                                          y=['Actual NO', 'Actual YES'])
        fig.update_layout(title=title,
                          title_x=0.5, title_font=dict(size=12),
                          font_family="Courier New",
                          width=500, height=500)
        # fig.show()
        # Save the plot as html
        fig.write_html(filename)

    def plot_results(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Plot the results
        :param grid: grid search object
        :param X_train: train data
        :param X_test: test data
        :param y_train: train target
        :param y_test: test target
        :param model_name: name of the model
        :return: None
        """
        # plot histogram of the results showing f1 score, accuracy and precision for both train and test
        X_train_pred = model.predict(X_train)
        train_f1 = f1_score(y_train, X_train_pred)
        train_acc = accuracy_score(y_train, X_train_pred)
        train_precision = precision_score(y_train, X_train_pred)

        X_test_pred = model.predict(X_test)
        test_f1 = f1_score(y_test, X_test_pred)
        test_acc = accuracy_score(y_test, X_test_pred)
        test_precision = precision_score(y_test, X_test_pred)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Train', 'Test'], y=[train_f1, test_f1],
                             name='F1 score', marker_color='indianred'))
        fig.add_trace(go.Bar(x=['Train', 'Test'], y=[train_acc, test_acc],
                             name='Accuracy', marker_color='lightsalmon'))
        fig.add_trace(go.Bar(x=['Train', 'Test'], y=[train_precision, test_precision],
                             name='Precision', marker_color='darkorange'))
        fig.update_layout(title='Results for {}'.format(model_name),
                          yaxis_title='Score',
                          barmode='group',
                          bargap=0.15,
                          title_x=0.5,
                          width=600,
                          height=400, font_family="Courier New, monospace")

        # move legend to the top
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))

        fig.write_html("plots/results_{}.html".format(model_name))
