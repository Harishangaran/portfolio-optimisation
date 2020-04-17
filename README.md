# Portfolio-optimisation
Optimise your portfolio using Monte Carlo simulation and plot your efficient frontier curve

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

#Calling function:
    
    optimisePortfolio(['AAPL','AMZN','IBM','MSFT'],'1000d',iterations=10000)

    Call list of stocks from yahoo finance. Check yahoo finance for ticker
    symbols
    
    'd' is for daily data.
    
    iterations is the number of simulations

Bonus: the model will output the efficient frontier curve with all the
sim points along with the point with best Sharpe (large red dot).

You are welcome to edit and improve the efficiency of the model.

If you find the object oriented code below difficult to understand
just request for functional code.
