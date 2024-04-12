import os
from darts.metrics.metrics import mape, mase, mse, rmse, smape, mae
import plotly.graph_objects as go
import plotly.io as pio
from darts.models.forecasting.arima import ARIMA
import pandas as pd
import gc
from darts import TimeSeries

def load_model_sarima_model(model_name=None):
    model = ARIMA.load(model_name)
    return model


def get_data(df=None,
             site_name=None,
             aqs_param=None,
             city=None):

    site_level_1 = df[(df['Site Name'] == site_name) & (df['COUNTY'] == city) & (df['AQS_PARAMETER_DESC'] == aqs_param)]
    grouped = site_level_1[['Date', 'DAILY_AQI_VALUE']]
    grouped.reset_index(inplace=True, drop=True)
    return grouped


# Function to calculate evaluation metrics for time series forecasts
def Evaluation_matrics(actual, predicted, insample):
    # Calculate the evaluation metrics
    return {
        'mae': mae(actual, predicted),
        'sampe': smape(actual, predicted),
        "mse": mse(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
        "mase": mase(actual, predicted, insample)

    }


def plot_testing_training(Horizon=None,
                          predicted=None,
                          testing_sc=None):
    parent_experiments_image_plot = 'plots/experiments_plots/image'
    parent_experiments_html_plot = 'plots/experiments_plots/html'

    os.makedirs(f'{parent_experiments_image_plot}', exist_ok=True)
    os.makedirs(f'{parent_experiments_html_plot}', exist_ok=True)

    # Creating visualization of predicted and actual AQI values
    fig = go.Figure()

    # Adding a trace for predicted AQI values
    fig.add_trace(go.Scatter(
        x=predicted.pd_series().index,  # x axis values set to index of predicted series
        y=predicted.pd_series().values,  # y axis values set to values of predicted series
        name='predicted'))

    # Adding a trace for actual AQI values
    fig.add_trace(go.Scatter(
        x=testing_sc[0:Horizon].pd_series().index,  # x axis values set to index of actual series
        y=testing_sc[0:Horizon].pd_series().values,  # y axis values set to values of actual series
        name='actual'))

    # Updating the layout of the visualization
    fig.update_traces(hoverinfo='text+name',
                      mode='lines+markers')  # Adding hover info to show text and name of each trace
    fig.update_layout(
        legend=dict(y=0.1, traceorder='reversed', font_size=16),  # Updating the legend of the visualization
        title="Horizon (Days) : " + str(Horizon),  # Updating the title of the visualization,
        width=1200
    )

    # Updating the x-axis of the visualization to have rangeslider and ranges elector
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([dict(step="all")])
        ),
    )

    # Showing the final visualization
    # fig.show()
    pio.write_html(fig,
                   os.path.join(parent_experiments_html_plot, f"{Horizon}.html"),
                   auto_open=False
                   )
    pio.write_image(fig,
                    os.path.join(parent_experiments_image_plot, f"{Horizon}.png"),
                    format='png')
    fig.show()

def train_test_predicted_plot(df_train, df_test, x_feature, y_feature, predicted, model_name, filename,metrics):
    """
    Plots the training data, actual values, and forecasted values using Plotly.

    Args:
        train (pd.Series): The training data.
        test (pd.Series): The actual values.
        predicted (pd.Series): The forecasted values.
        model_name (str): The name of the forecasting model.

    Returns:
        None
    """

    # Create a subplot with two rows and one column
    try:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_train[x_feature],
                y=df_train[y_feature],
                name='Input Series',
                mode='lines+markers'
            ))

        # Add a trace for actual values
        fig.add_trace(
            go.Scatter(
                x=df_test[x_feature],
                y=df_test[y_feature],
                name='Actual Values',
                mode='lines+markers'
            )
        )

        # Add a trace for forecasted values
        fig.add_trace(
            go.Scatter(
                x=df_test[x_feature],
                y=predicted[y_feature],
                name=f'Sarima Model Prediction',
                mode='lines+markers'
            )
        )

        # Update xaxis properties
        fig.update_xaxes(title_text='Time')

        # Update yaxis properties
        fig.update_yaxes(title_text=y_feature)

        # Update title and height
        title = f'Forecasting using {model_name}\ninput window size :{df_train.shape[0]}\n Horizan : {df_test.shape[0]}\n Metrics : {metrics}'
        fig.update_layout(
            title=title,
            height=500,
            width=1500,
            legend=dict(x=0, y=1, traceorder='normal', orientation='h')
        )

        # Save the plot as an HTML file
        # fig.show()
        parent_path = os.path.join('..','Plots','ARIMAModel', 'Experiments')
        os.makedirs(parent_path,exist_ok=True)
        # print('parent path : ',parent_path)
        fig.write_html(f'{parent_path}/forecasting_using_{model_name}_combination_{filename}'+'.html')
        fig.write_image(f'{parent_path}/forecasting_using_{model_name}_combination_{filename}' + '.png')
        fig.show()
    except Exception as e:
        print("Error ", e)
def plot_visualization(predictions=None,
                       ts_train=None,
                       ts_test=None,
                       filename=None,
                       metrics =  None
                       ):
    try:

        # Convert train_series into a pandas dataframe and reset index
        df_train = ts_train.pd_dataframe().reset_index()

        # Convert test_series into a pandas dataframe and reset index
        df_test = ts_test.pd_dataframe().reset_index()

        # Convert prediction into a pandas dataframe and reset index
        forecast = predictions.pd_dataframe().reset_index()

        x_feature = 'Date'
        y_feature = 'MedianSoldPrice_AllHomes'
        filename = filename
        train_test_predicted_plot(df_train, df_test, x_feature, y_feature, forecast, 'ARIMA-Prediction',
                                  filename=filename,metrics=metrics)
    except Exception as e:
        print('Error occurred in plot_visualization function : ', e)


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def qq_plot(data, distribution='norm', line=True):
    """
    Create a QQ plot to compare the given data distribution with the specified theoretical distribution.
    
    Parameters:
        data (array-like): The data to be plotted.
        distribution (str, optional): The theoretical distribution to compare against. Default is 'norm' (normal distribution).
        line (bool, optional): If True, a line representing perfect fit will be drawn. Default is True.
    
    Returns:
        None
    """
  

    # Generate the theoretical quantiles based on the specified distribution
    if distribution == 'norm':
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
    elif distribution == 'uniform':
        theoretical_quantiles = stats.uniform.ppf(np.linspace(0.01, 0.99, len(data)))
    else:
        raise ValueError("Unsupported distribution. Please choose 'norm' or 'uniform'.")
    
    # Sort the data and calculate empirical quantiles
    sorted_data = np.sort(data)
    empirical_quantiles = np.quantile(sorted_data, np.linspace(0.01, 0.99, len(data)))
    
    # Plot the QQ plot
    plt.figure(figsize=(8, 6))
    plt.scatter(theoretical_quantiles, empirical_quantiles, color='blue', edgecolors='k')
    plt.xlabel('Theoretical Quantiles(Data Quantile)')
    plt.ylabel('Empirical Quantiles(Normal Qantile)')
    plt.title('QQ Plot')
    
    # Draw a line representing perfect fit if line parameter is True
    if line:
        min_val = min(theoretical_quantiles[0], empirical_quantiles[0])
        max_val = max(theoretical_quantiles[-1], empirical_quantiles[-1])
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    
    plt.grid(True)
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_empirical_and_normal(data):
    # Plot empirical distribution (histogram)
    plt.hist(data, bins=20, density=True, alpha=0.6, color='g', label='Empirical Distribution')

    # Fit normal distribution
    mu, std = norm.fit(data)

    # Plot the PDF of the fitted normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')

    # Add labels and legend
    plt.xlabel('Data')
    plt.ylabel('Probability Density')
    plt.title('Empirical and Fitted Normal Distribution')
    plt.legend()

    plt.show()

def evaluation_of_model_sarima_Model(
        model_name=None,
        ts_test=None,
        ts_train=None):
    try:

        # Load the pre-trained model
        model = load_model_sarima_model(model_name)

        # List to store evaluation results for different configurations
        evaluation_dict_list = []

         # Loop over different window sizes
        input_window = [7, 7 * 2, 7 * 3, 7 * 4, 7 * 5,7 * 6]
        for window_size in input_window:

            # Extract a subset of the test data based on the window size and current position
            subset_input = ts_test[:window_size]
            actual_val = ts_test[window_size:]


            actual_val_pd_series = actual_val.pd_series()
            actual_val_pd_series[actual_val_pd_series.values ==0] = 1
            actual_val = TimeSeries.from_series(actual_val_pd_series)




            # Calculate the forecast horizon
            horizon = len(actual_val)
            print(f'Evaluation of input window : {window_size} & Horizon : {horizon}')

            # Make predictions using the model
            forecasted_series = model.predict(horizon, series=subset_input)

            forecasted_pd_series = forecasted_series.pd_series()
            forecasted_pd_series[forecasted_pd_series.values <0] = 0
            forecasted_series = TimeSeries.from_series(forecasted_pd_series)

            # Generate a filename based on the current configuration
            filename = f"mode_name_{model_name.split('/')[-1].split('.')[0]}_input_window_{window_size}_output_window_{horizon}"

            # Inverse transform the forecasted series

            # Calculate evaluation metrics
            metrics = calculate_metrics(actual_val, forecasted_series)

            # Display the calculated metrics
            print('metrics:', metrics)

            # Store the evaluation results in a dictionary
            evaluation_dict = {
                'input_window_in_days': window_size,
                'output_window_in_days': horizon,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE'],
                'MSE': metrics['MSE']
            }

            # Append the evaluation dictionary to the list
            evaluation_dict_list.append(evaluation_dict)

            # Uncomment the following lines if you want to plot visualizations
            plot_visualization(forecasted_series,
                                ts_train=subset_input,
                                ts_test=actual_val,
                                filename=filename,
                                  metrics=metrics)

            # Optional: Break the inner loop for testing with fewer iterations
                # Clear intermediate variables
            del subset_input, actual_val, forecasted_series, metrics

        # Create a DataFrame from the list of evaluation dictionaries
        evaluation_df = pd.DataFrame(evaluation_dict_list)

        # Perform garbage collection to free up memory
        gc.collect()
        return evaluation_df
    except Exception as e:
        print('[Error] Occurred in evaluation_of_model() : ',e)

def calculate_metrics(actual, predicted):
    try:

        # Convert inputs to numpy arrays for easier calculations
        actual = np.array(actual.values())
        predicted = np.array(predicted.values())

        # Calculate individual metrics
        mae = np.mean(np.abs(predicted - actual))
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        mape = np.mean(np.abs((predicted - actual) / actual)) * 100
        mse = np.mean((predicted - actual) ** 2)

        metrics = {
            "MAE": np.round(mae, 2),
            "RMSE": np.round(rmse, 2),
            "MAPE": np.round(mape, 2),
            "MSE": np.round(mse, 2),
        }
        return metrics

    except Exception as e:
        print('Error in calculate_metrics() : ',e)


