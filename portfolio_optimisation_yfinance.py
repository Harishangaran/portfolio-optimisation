# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:49:08 2020

@author: harishangaran

The following scripts allows you to call dynamic stock prices
from yahoo finance and run monte carlo simulations model to
get the optimised portfolio weightings for your desired stocks
in your portfolio.

All you have to do is call the class and input your list of stocks
in your portfolio, the period you want to lookback and the number of
iterations you want to run.

Call many stocks as you want. For best results run atleast 10000 sims.

The model will output the best weightings for your stocks in the
portfolio, the maximum return and the volatility and the best Sharpe
ratio.

Calling function:
    optimisePortfolio(['AAPL','AMZN','IBM','MSFT'],'1000d',iterations=10000)

# Call list of stocks from yahoo finance. Check yahoo finance for ticker
    symbols
    
# 'd' is for daily data.
    
# iterations is the number of simulations

Bonus: the model will output the efficient frontier curve with all the
sim points along with the point with best Sharpe (large red dot).

You are welcome to edit and improve the efficiency of the model.

If you find the object oriented code below difficult to understand
just request for functional code.

"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.optimize import minimize


class optimisePortfolio:
    def __init__(self,stocklist,period,iterations=5000):
        self.stocklist = stocklist
        self.period = period
        self.iterations = iterations
        self.callPrices()
        self.calculateLogReturn()
        self.runSimulation()
        self.createCurveVal()
        self.plotEfficientFrontier()
        
    def callPrices(self):
        
        # Calling the price from yahoo finance
        listOfStocks = [yf.Ticker(i).history(period=self.period
                                             ) for i in self.stocklist]
        
        # Zipping the stock names as keys of a dictionary
        self.listOfStocksDict = dict(zip(self.stocklist,listOfStocks ))
        
        # Appending all close prices to a list of series
        listOfClosePrices = []
        for i in range(len(self.stocklist)):
            listOfClosePrices.append(self.listOfStocksDict[
                self.stocklist[i]]['Close'])
            
        # Concating the list of series to a pandas dataframe
        self.portfolio = pd.concat(listOfClosePrices,axis=1)
        self.portfolio.columns = self.stocklist
        self.portfolio.dropna(inplace=True)
        return self.portfolio


    def calculateLogReturn(self):
        self.df = self.portfolio
        
        #calculating log returns of the stock close prices
        self.logReturn = np.log(self.df/self.df.shift(1)) #log returns
        return self.logReturn
        
    def runSimulation(self):
        #using np.random.seed(101) to create same random numbers in
        #order every time you simulate
        np.random.seed(101)
        
        #creating empty arrays for weights,returns,volume and sharpe ratio
        self.allWeights = np.zeros(((self.iterations),len(self.df.columns))) #np.zeros((5000,4))
        self.returnArray = np.zeros(self.iterations)
        self.volumeArray = np.zeros(self.iterations)
        self.sharpeArray = np.zeros(self.iterations)

        for ind in range(self.iterations):
            
            #weights
            weights = np.array(np.random.random(len(self.df.columns)))
            weights = weights/np.sum(weights) #make weights = sum of 1
            
            #save weights
            self.allWeights[ind,:] = weights
            
            #get expected portfolio return and save it to array
            self.returnArray[ind] = np.sum((self.logReturn.mean() * weights) * 252)
            
            #get expected volatility and save it to array
            #since using 1000s of iteration you need to use 
            #linear algebra to speed up cal
            self.volumeArray[ind] = np.sqrt(np.dot(weights.T,np.dot(self.logReturn.cov() * 252,weights)))
            
            self.sharpeArray[ind] = self.returnArray[ind]/self.volumeArray[ind]

        #get max sharpe
        self.sharpeArray.max()
        
        #get loc of max sharpe
        self.sharpeArray.argmax() #use this to find optimum weight
                
        #get the optimum weight information
        self.weightinfo = self.allWeights[self.sharpeArray.argmax(),:] * 100
        
        #zip name to values to a dict
        self.weightList = dict(zip(self.df.columns,self.weightinfo))
        
        self.listtoprint = list(self.weightList.values())
        self.listtoprint = [ '%.2f' % elem for elem in self.listtoprint ]
        #print(self.listtoprint)
        
        #print weight information
        #check console
        for item in self.df.columns:
            print("The weight % for {} is {}".format(item,self.weightList[item]))
        
        #print max return and the vol
        #check console
        print("The maximum return is {} % where the volatility was {} % with a Sharpe of {}".format(
            self.returnArray[self.sharpeArray.argmax()],self.volumeArray[self.sharpeArray.argmax()],self.sharpeArray.max()))
        return self.listtoprint
    #section to get the efficient frontier curve values
    #getting the max return for volatility
    def getRetVolSharpeArrays(self,weights):
        weights = np.array(weights)
        ret = np.sum((self.logReturn.mean() * weights) * 252)
        vol = np.sqrt(np.dot(weights.T,np.dot(self.logReturn.cov() * 252,weights)))
        sr = ret/vol
        return np.array([ret,vol,sr])

    def checkSum(self,weights):
        # return 0 if the sum of the weights is 1
        return np.sum(weights) - 1

    def minimizeVolatility(self,weights):
        return self.getRetVolSharpeArrays(weights)[1] #grabbing vol col
    
    #creating the effcient frontier curve values
    #that is getting the maximum return for each volatility point
    def createCurveVal(self):
        
        self.frontier_volatility = []
        self.bounds = [(0,1) for i in range(len(self.df.columns))]
        self.initial_guess = np.repeat(1/len(self.df.columns),len(self.df.columns))
        
        #check efficient frontier - best y val for x val
        self.frontier_y = np.linspace(0,self.returnArray.max(),100) #values are the return range - check graph
                                                                    #100 points
        for possible_return in self.frontier_y:
            cons = ({'type':'eq','fun':self.checkSum},
                {'type':'eq','fun':lambda w: self.getRetVolSharpeArrays(w)[0] - possible_return})
            result = minimize(self.minimizeVolatility,self.initial_guess,method='SLSQP',bounds=self.bounds,constraints=cons)
            self.frontier_volatility.append(result['fun'])

        return self.frontier_volatility,self.frontier_y
    
    ''' Printing Efficient Frontier Graph '''    
    def plotEfficientFrontier(self):
        
        #plot simulation points
        plt.figure(figsize=(12,8))
        
        #try different cmap styles - check the ones below
        #inferno,magma,viridis,plasma
        plt.scatter(self.volumeArray,self.returnArray,c=self.sharpeArray,cmap='viridis',alpha=0.7) 
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        
        #plot the optimum point
        plt.scatter(self.volumeArray[self.sharpeArray.argmax()],self.returnArray[self.sharpeArray.argmax()],c='red',s=50,edgecolor='black')
        #plot frontier curve
        plt.plot(self.frontier_volatility,self.frontier_y,'r--',linewidth=3) 
        plt.table(cellText=[self.stocklist],cellLoc='center',rowLabels=['Assets '],loc='bottom',bbox=[0.0,-0.2,1,0.1])
        plt.table(cellText=[self.listtoprint],cellLoc='center',rowLabels=['Weight'],loc='bottom',bbox=[0.0,-0.3,1,0.1])
        plt.title('Efficient Frontier Curve from {} simulations - author Harishangaran'.format(self.iterations))
        #plt.subplots_adjust(left=0.2, bottom=0.2)
        
        
optimisePortfolio(['AAPL','AMZN','IBM','MSFT'],'1000d',iterations=100000)