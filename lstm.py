import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the scriptname (in argv[0])

import backtrader as bt

from bokeh.plotting import figure, show, output_file, output_notebook, save
import aspose.words as aw



def flask_html(Symbol):
        #get data on this ticker
    tickerData = yf.Ticker(Symbol)

    #get the historical prices for this ticker
    data = tickerData.history( start='2015-1-1', end='2021-7-6')
    datatest=tickerData.history( start='2021-02-10', end='2021-7-6')
    #see your data
    data
    datatest.to_csv('data_test.csv')
    #data
    pd.read_csv('data_test.csv')
    datatest

    # data['Close'].plot()
    # plt.title('Intraday Times Series for the '+f'{Symbol}'+' stock')
    # plt.show()

    len(data.iloc[:,0:1].values)

    Keep_Days=100
    training_set=data.iloc[:-Keep_Days,0:1].values
    len(training_set)

    # 3. Feature engineering

    ### Feature Scaling

    #sc = StandardScaler()
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    ### Creating a data structure with 60 timesteps and 1 output

    n=60

    X_train = []
    y_train = []
    for i in range(n, len(training_set)):
        X_train.append(training_set_scaled[i-n:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    ### Reshaping

    X_train.shape

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    X_train.shape

    # 4. Model building

    ## Building and Training the LSTM

    ### Initialising the LSTM

    regressor = Sequential()

    ### Adding the first LSTM layer 

    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    #regressor.add(Dropout(0.2))

    ### Adding a second LSTM layer 

    regressor.add(LSTM(units = 50, return_sequences = True))
    #regressor.add(Dropout(0.2))

    ### Adding a third LSTM layer 

    regressor.add(LSTM(units = 50, return_sequences = True))
    #regressor.add(Dropout(0.2))
        
    ### Adding a fourth LSTM layer 

    regressor.add(LSTM(units = 50, return_sequences = True))
    #regressor.add(Dropout(0.2))

    ### Adding a fifth LSTM layer 

    regressor.add(LSTM(units = 50))
    #regressor.add(Dropout(0.2))

    ### Adding the output layer

    regressor.add(Dense(units = 1))

    # 5. Compiling the LSTM

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    ### Fitting the LSTM to the Training set

    regressor.fit(X_train, y_train, epochs = 40, batch_size = 32,verbose=0)

    # 6. Making the predictions

    real_stock_price = data.iloc[-Keep_Days:, 0:1].values

    len(real_stock_price)

    ### Getting the predicted stock price

    #dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    inputs = data.iloc[len(data)- len(real_stock_price) -n:,0:1].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(n, n+len(real_stock_price)):
        X_test.append(inputs[i-n:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    len(predicted_stock_price)

    def f(l):
        l=np.array(l)
        l=l.reshape(-1,1)
        l=sc.transform(l)
        y= regressor.predict(np.array([l]))
        y=sc.inverse_transform(y)
        return y[0][0]

    X_test.shape

    # 7. Visualising the results

    # plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
    # plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
    # plt.title('Stock Price Prediction')
    # plt.xlabel('Time')
    # plt.ylabel(Symbol+' Stock Price')
    # plt.legend()
    # plt.show()

    # 8. Backtesing

    # Create a Stratey
    class TestStrategy(bt.Strategy):

        def log(self, txt, dt=None):
            ''' Logging function fot this strategy'''
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))


        # params = (
        #     ("period", 2),
        #     ("af", 0.02),
        #     ("afmax", 0.2)
        #     )
        def __init__(self):
            # Keep a reference to the "close" line in the data[0] dataseries
            self.dataclose = self.datas[0].close
            # self.psar = bt.indicators.ParabolicSAR(period=self.p.period, af=self.p.af, afmax=self.p.afmax)




            # To keep track of pending orders and buy price/commission
            self.order = None
            self.buyprice = None
            self.buycomm = None

        def notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]:
                # Buy/Sell order submitted/accepted to/by broker - Nothing to do
                return

            # Check if an order has been completed
            # Attention: broker could reject order if not enough cash
            if order.status in [order.Completed]:
                if order.isbuy():
                    self.log(
                        'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                        (order.executed.price,
                         order.executed.value,
                         order.executed.comm))

                    self.buyprice = order.executed.price
                    self.buycomm = order.executed.comm
                else:  # Sell
                    self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                             (order.executed.price,
                              order.executed.value,
                              order.executed.comm))

                self.bar_executed = len(self)

            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log('Order Canceled/Margin/Rejected')

            self.order = None

        def notify_trade(self, trade):
            if not trade.isclosed:
                return

            self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                     (trade.pnl, trade.pnlcomm))

        def next(self):
            # Simply log the closing price of the series from the reference
            #############################################self.log('Close, %.2f' % self.dataclose[0])

            # Check if an order is pending ... if yes, we cannot send a 2nd one
            if self.order:
                return

            # Check if we are in the market
            if not self.position:

                X1=[self.dataclose[i] for i in range(-n,0)]
                X2=[self.dataclose[i] for i in range(-n+1,1)]
                # Not yet ... we MIGHT BUY if ...
                if f(X1) < f(X2):# and self.psar<self.dataclose[0]:
                        # current close less than previous close

                        #if self.dataclose[-1] < self.dataclose[-2]:
                            # previous close less than the previous close

                            # BUY, BUY, BUY!!! (with default parameters)
                    self.log('LONG CREATE, %.2f' % self.dataclose[0])

                        # Keep track of the created order to avoid a 2nd order
                    self.order = self.buy()

                # elif  f(X2) < f(X1):# and self.dataclose[0]<self.psar:
                #     self.log('SHORT CREATE, %.2f' % self.dataclose[0])
                #     self.order = self.sell()

            else:

                X1=[self.dataclose[i] for i in range(-n,0)]
                X2=[self.dataclose[i] for i in range(-n+1,1)]
                # Already in the market ... we might sell
                if (self.position.size)>0 and f(X2) < f(X1) :
                    # SELL, SELL, SELL!!! (with all possible default parameters)
                    self.log('CLOSE CREATE, %.2f' % self.dataclose[0])

                    # Keep track of the created order to avoid a 2nd order
                    self.order = self.close()
                elif  (self.position.size)<0 and f(X1) < f(X2):
                    self.log('CLOSE CREATE, %.2f' % self.dataclose[0])
                    self.order = self.close()

    cerebro = bt.Cerebro()

    cerebro.addstrategy(TestStrategy)

    data = bt.feeds.GenericCSVData(
        dataname='data_test.csv',

        fromdate=datetime.datetime(2021, 2, 10),
        todate=datetime.datetime(2021, 7, 6),

        nullvalue=0.0,

        dtformat=('%Y-%m-%d'),

        datetime=0,
        high=2,
        low=3,
        open=1,
        close=4,
        volume=5,
        openinterest=-1
    )

    #data_backtest = bt.feeds.YahooFinanceData(dataname = Symbol, fromdate = datetime.datetime(2020,10,2), todate = datetime.datetime(2021, 2, 26),reverse=False)

    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    cerebro.addsizer(bt.sizers.PercentSizer, percents = 98)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    plt.ion()
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams.update({'font.size': 10}) 
    
    figure = cerebro.plot()[0][0]
    plt.close(figure)
    
    return figure.savefig('static/chart.png')
