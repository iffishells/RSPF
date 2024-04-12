import os

from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from darts.utils.statistics import check_seasonality



def inspect_seasonality(ts):
    try:

        seasonal_period = []
        for m in range(2, 25):
            is_seasonal, period = check_seasonality(ts, m=m, alpha=0.05)
            if is_seasonal:
                print("There is seasonality of order {}.".format(period))
                seasonal_period.append(period)

        print(f'There is seasonality of order : {seasonal_period}')
        return seasonal_period
    except Exception as e:
        print(f'Error : {e}')
def adfuller_test(shot_count):
    # Perform the AD Fuller test
    result = adfuller(shot_count)

    # Print the test results
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))

    # Interpret the test results
    if result[1] <= 0.05:
        print(
            "strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


from darts.utils.statistics import plot_acf, plot_pacf


def plotting_acf_plot(ts=None,
                      m=None,
                      max_lag=100,
                      parent_dir_name_for_saving_plots=None,
                      series_type: str = None,
                      filename=None
                      ):
    # data_name = filename.split('__')[0]
    # plot_file_dir = os.path.join(parent_dir_name_for_saving_plots,
    #                              filename)
    # os.makedirs(plot_file_dir,exist_ok=True)
    #
    # plot_filename = os.path.join(plot_file_dir,f'series_type_{series_type}_{data_name}_acf_plot.png')



    plot_acf(ts, m=m, max_lag=max_lag, fig_size=(10, 5), axis=None, default_formatting=True)
    plt.xlabel('lags')
    plt.ylabel('correlation')
    plt.title(f'Auto Correlation Plot of {filename}')
    plt.tight_layout()
    # plt.savefig(plot_filename)
    plt.show()
    # plt.close()


def plotting_pacf_plot(ts=None,
                      m=None,
                      max_lag=100,
                      parent_dir_name_for_saving_plots=None,
                      series_type: str = None,
                      filename=None
                      ):
    # data_name = filename.split('__')[0]
    # plot_file_dir = os.path.join(parent_dir_name_for_saving_plots,
    #                              filename)
    # os.makedirs(plot_file_dir,exist_ok=True)

    # plot_filename = os.path.join(plot_file_dir,f'series_type_{series_type}_{data_name}_pacf_plot.png')



    plot_pacf(ts, m=m, max_lag=max_lag, fig_size=(10, 5), axis=None, default_formatting=True)
    plt.xlabel('lags')
    plt.ylabel('correlation')
    plt.title(f'Partial Auto Correlation Plot of {filename}')
    plt.tight_layout()
    # plt.savefig(plot_filename)
    plt.show()
    # plt.close()
