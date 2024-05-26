import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def load_dataset(lookback, path_to_data='./data/globa_superstore.csv'):
    data = pd.read_csv(path_to_data, encoding='unicode_escape')

    data.drop('Unnamed: 0', axis=1, inplace=True)

    data_arima = data.copy(deep=True)

    data_acc = data_arima.loc[data_arima['Sub-Category'] == 'Tables']

    data_sales_acc = data_acc[['Order Date', 'Sales']].copy(deep=True)

    add_data_acc = data_sales_acc.groupby(['Order Date'])

    add_data_acc = add_data_acc.mean().round(2).reset_index()

    dates = pd.date_range(start='2011-01-03', end='2014-12-31', freq='D')
    dates = pd.DataFrame(dates)
    dates = dates.rename(columns={0: 'Order Date'})
    add_data_acc["Order Date"] = pd.to_datetime(add_data_acc["Order Date"])
    dates = pd.merge(dates, add_data_acc, on='Order Date', how="left")
    dates = dates.fillna(0)
    dates.set_index('Order Date', inplace=True)
    datesm = dates.resample('M').mean()

    timeseries = datesm[['Sales']].values.astype('float32')

    train_size = int(len(timeseries) * 0.67)
    train, test = timeseries[:train_size], timeseries[train_size:]
    x_train, y_train = create_dataset(train, lookback=lookback)
    x_test, y_test = create_dataset(test, lookback=lookback)
    return timeseries, x_train, y_train, x_test, y_test


def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


def create_plot(model, timeseries, x_train, x_test, lookback):
    train_size = int(len(timeseries) * 0.67)
    with torch.no_grad():
        train_plot = np.ones_like(timeseries) * np.nan
        y_pred = model(x_train)
        y_pred = y_pred[:, -1, :]
        train_plot[lookback:train_size] = model(x_train)[:, -1, :]
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[train_size + lookback:len(timeseries)] = model(x_test)[:, -1, :]

    plt.plot(timeseries)
    plt.plot(train_plot, c='r')
    plt.plot(test_plot, c='g')
    plt.show()
