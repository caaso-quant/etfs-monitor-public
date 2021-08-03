import numpy as np
import datetime as dt
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import cov
import scipy
import pandas as pd
import plotly.graph_objects as go

class EfficientFrontier:
    def get_data(self, stock_data, start_date):
        """Filters returns with start_date and returns mean and covariation matrix.

        :param stock_data: Dataframe containing the assets historical returns
        :type stock_data: pd.DataFrame
        :param start_date: Datetime to filter the historical returns.
        :type start_date: datetime
        :return: Mean returns and covariation matrix.
        :rtype: tuple(pd.DataFrame, pd.DataFrame)
        """
        stock_data = stock_data[stock_data.index > start_date]
        mean_returns = stock_data.mean()
        cov_matrix = stock_data.cov()

        return mean_returns, cov_matrix

    def portfolio_performance(self, weights, mean_returns, cov_matrix):
        """Calculates portfolio performance and standard deviation given \
            assets weights, mean returns and covariation matrix.

        :param weights: Weights of assets in portfolio.
        :type weights: list
        :param mean_returns: Mean returns of the assets.
        :type mean_returns: pd.Series
        :param cov_matrix: Covariation matrix of the assets.
        :type cov_matrix: pd.DataFrame
        :return: Portfolio return and standard deviation.
        :rtype: tuple(float, float)
        """
        returns = np.sum(mean_returns*weights)*12
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)
        return returns, std

    def negative_sr(self, weights, mean_returns, cov_matrix, risk_free_rate):
        """Calculates negative Sharpe Ratio.

        :param weights: Weights of assets in portfolio.
        :type weights: list
        :param mean_returns: Mean returns of the assets.
        :type mean_returns: pd.Series
        :param cov_matrix: Covariation matrix of the assets.
        :type cov_matrix: pd.DataFrame
        :param risk_free_rate: Risk free rate.
        :type risk_free_rate: float
        :return: Sharpe ratio multiplied by -1.
        :rtype: float
        """
        p_returns, p_std = self.portfolio_performance(weights, mean_returns, cov_matrix)
        return - (p_returns - risk_free_rate)/p_std

    def adjust_weights(self, weights):
        """Adjust weights for sum to be 1.

        :param weights: Weights of assets in portfolio.
        :type weights: list
        :return: Adjusted weights.
        :rtype: list
        """
        sum_weights = np.sum(np.array(weights))
        return [weight / sum_weights for weight in weights]

    def portfolio_variance(self, weights, mean_returns, cov_matrix):
        """Calculate portfolio variance (standard deviation).

        :param weights: Weights of assets in portfolio.
        :type weights: list
        :param mean_returns: Mean returns of the assets.
        :type mean_returns: pd.Series
        :param cov_matrix: Covariation matrix of the assets.
        :type cov_matrix: pd.DataFrame
        :return: Portfolio standard deviation.
        :rtype: float
        """
        return self.portfolio_performance(weights, mean_returns, cov_matrix)[1]

    def portfolio_return(self, weights, mean_returns, cov_matrix):
        """Calculate portfolio return.

        :param weights: Weights of assets in portfolio.
        :type weights: list
        :param mean_returns: Mean returns of the assets.
        :type mean_returns: pd.Series
        :param cov_matrix: Covariation matrix of the assets.
        :type cov_matrix: pd.DataFrame
        :return: Portfolio return.
        :rtype: float
        """
        return self.portfolio_performance(weights, mean_returns, cov_matrix)[0]

    def maximize_sr(self, mean_returns, cov_matrix, risk_free_rate, constraint_set=(0, 1)):
        """Calculate weights that maximizes portfolio sharpe ratio.

        :param mean_returns: Mean returns of the assets.
        :type mean_returns: pd.Series
        :param cov_matrix: Covariation matrix of the assets.
        :type cov_matrix: pd.DataFrame
        :param risk_free_rate: Risk free rate.
        :type risk_free_rate: float
        :param constraint_set: Constraint set for solver, defaults to (0, 1)
        :type constraint_set: tuple, optional
        :return: Scipy solver result.
        :rtype: scipy.optimize.OptimizeResult
        """
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix, risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constraint_set
        bounds = tuple(bound for asset in range(num_assets))
        result = scipy.optimize.minimize(self.negative_sr, num_assets*[1./num_assets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def minimize_variance(self, mean_returns, cov_matrix, risk_free_rate, constraint_set=(0, 1)):
        """Calculate weights that minimizes portfolio variance.

        :param mean_returns: Mean returns of the assets.
        :type mean_returns: pd.Series
        :param cov_matrix: Covariation matrix of the assets.
        :type cov_matrix: pd.DataFrame
        :param risk_free_rate: Risk free rate.
        :type risk_free_rate: float
        :param constraint_set: Constraint set for solver, defaults to (0, 1)
        :type constraint_set: tuple, optional
        :return: Scipy solver result.
        :rtype: scipy.optimize.OptimizeResult
        """
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constraint_set
        bounds = tuple(bound for asset in range(num_assets))
        result = scipy.optimize.minimize(self.portfolio_variance, num_assets*[1./num_assets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result 

    def get_results(self, func, mean_returns, cov_matrix, risk_free_rate, constraint_set=(0, 1)):
        """Calculate portfolio returns, standard deviation and weights, \
            given a solver weights function.

        :param func: Solver function that returns portfolio weights.
        :type func: func
        :param mean_returns: Mean returns of the assets.
        :type mean_returns: pd.Series
        :param cov_matrix: Covariation matrix of the assets.
        :type cov_matrix: pd.DataFrame
        :param risk_free_rate: Risk free rate.
        :type risk_free_rate: float
        :param constraint_set: Constraint set for solver, defaults to (0, 1)
        :type constraint_set: tuple, optional
        :return: Portfolio return, standard deviation and assets weights.
        :rtype: tuple(float, float, list)
        """
        portfolio = func(mean_returns, cov_matrix, risk_free_rate)
        returns, std = self.portfolio_performance(portfolio['x'], mean_returns, cov_matrix)
        allocation = pd.DataFrame(portfolio['x'], index = mean_returns.index, columns=['allocation'])
        return returns, std, allocation

    def efficient_optimization(self, mean_returns, cov_matrix, return_target, constraint_set=(0, 1)):
        """Optimizes for minimum variance given a target return.

        :param mean_returns: Mean returns of the assets.
        :type mean_returns: pd.Series
        :param cov_matrix: Covariation matrix of the assets.
        :type cov_matrix: pd.DataFrame
        :param return_target: Target portfolio return.
        :type return_target: float
        :param constraint_set: Constraint set for solver, defaults to (0, 1)
        :type constraint_set: tuple, optional
        :return: Scipy solver result.
        :rtype: scipy.optimize.OptimizeResult
        """
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix)
        constraints = (
            {'type': 'eq', 'fun': lambda x: self.portfolio_return(x, mean_returns, cov_matrix) - return_target},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        bound = constraint_set
        bounds = tuple(bound for asset in range(num_assets))
        eff_opt = scipy.optimize.minimize(self.portfolio_variance, num_assets*[1./num_assets], args=args, method="SLSQP", bounds=bounds, constraints=constraints)
        return eff_opt

    def calculated_results(self, mean_returns, cov_matrix, risk_free_rate, constraint_set=(0,1)):
        """Calculates returns and standard deviation for maximum sharpe ration, minimum variance and efficient frontier.

        :param mean_returns: Mean returns of the assets.
        :type mean_returns: pd.Series
        :param cov_matrix: Covariation matrix of the assets.
        :type cov_matrix: pd.DataFrame
        :param risk_free_rate: Risk free rate.
        :type risk_free_rate: float
        :param constraint_set: Constraint set for solver, defaults to (0, 1)
        :type constraint_set: tuple, optional
        :return: Portfolio return, standard deviation and weights for maximum sharpe \
            ratio and minimum variance. Also the efficient frontier list and target returns.
        :rtype: tuple(float, float, list, float, float, list, list, list)
        """
        max_sr_returns, max_sr_std, max_sr_allocation = self.get_results(self.maximize_sr, mean_returns, cov_matrix, risk_free_rate, constraint_set)
        min_vol_returns, min_vol_std, min_vol_allocation = self.get_results(self.minimize_variance, mean_returns, cov_matrix, risk_free_rate, constraint_set)

        target_returns = np.linspace(min_vol_returns, max_sr_returns, 20)
        efficient_list = []
        for target in target_returns:
            efficient_list.append(self.efficient_optimization(mean_returns, cov_matrix, target)['fun'])

        max_sr_returns, max_sr_std = round(max_sr_returns*100, 2), round(max_sr_std*100, 2)
        min_vol_returns, min_vol_std = round(min_vol_returns*100, 2), round(min_vol_std*100, 2)

        return max_sr_returns, max_sr_std, max_sr_allocation, min_vol_returns, min_vol_std, min_vol_allocation, efficient_list, target_returns

    def ef_graph(self, mean_returns, cov_matrix, returns, std, risk_free_rate, constraint_set=(0, 1)):
        """Creates efficient frontier chart.

        :param mean_returns: Mean returns of the assets.
        :type mean_returns: pd.Series
        :param cov_matrix: Covariation matrix of the assets.
        :type cov_matrix: pd.DataFrame
        :param returns: Current portfolio return.
        :type returns: float
        :param std: Current portfolio standard deviation.
        :type std: float
        :param risk_free_rate: Risk free rate.
        :type risk_free_rate: float
        :param constraint_set: Constraint set for solver, defaults to (0, 1)
        :type constraint_set: tuple, optional
        :return: Efficient frontier chart figure.
        :rtype: go.Figure
        """
        max_sr_returns, max_sr_std, max_sr_allocation, min_vol_returns, min_vol_std, min_vol_allocation, efficient_list, target_returns = self.calculated_results(mean_returns, cov_matrix, risk_free_rate, constraint_set)

        max_sharpe_ratio = go.Scatter(
            name='Maximum Sharpe Ratio',
            mode='markers',
            x=[max_sr_std],
            y=[max_sr_returns],
            marker=dict(color='red', size=14, line=dict(width=3, color='black'))
        )

        min_vol_ratio = go.Scatter(
            name='Minimum Volatility',
            mode='markers',
            x=[min_vol_std],
            y=[min_vol_returns],
            marker=dict(color='green', size=14, line=dict(width=3, color='black'))
        )

        current_portfolio = go.Scatter(
            name='Current Portfolio',
            mode='markers',
            x=[std],
            y=[returns],
            marker=dict(color='blue', size=14, line=dict(width=3, color='black'))
        )

        ef_curve = go.Scatter(
            name='Efficient Frontier',
            mode='lines',
            x=[round(ef_std*100, 2) for ef_std in efficient_list],
            y=[round(target*100, 2) for target in target_returns],
            line=dict(color='white', width=4, dash='dashdot')
        )

        #data = [max_sharpe_ratio, min_vol_ratio, ef_curve, current_portfolio]
        data = [max_sharpe_ratio, min_vol_ratio, ef_curve]
        layout = go.Layout(
            title = 'Portfolio optitmisation',
            yaxis = dict(title='Annualised Return (%)'),
            xaxis = dict(title= 'Annualised Volatility (%)'),
            showlegend = True,
            legend = dict(
                x = 0.75, y = 0, traceorder = 'normal',
                bgcolor='black',
                bordercolor='white',
                borderwidth=2
            ),
            
            width=800,
            height=600
        )
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            dict(
                paper_bgcolor = 'black',
                plot_bgcolor = '#404040',
                font=dict(color='white')
            )
        )
        return fig


    def trace_by_time(self):
        """Creates efficient frontier chart.

        :return: Efficient frontier chart figure.
        :rtype: go.Figure
        """
        weights = self.adjust_weights(self.weights)
        weights = np.array(self.weights)
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days = 365 * self.years)
        mean_returns, cov_matrix = self.get_data(self.stocks, start_date)

        returns, std = self.portfolio_performance(weights, mean_returns, cov_matrix) 
        returns, std = round(returns*100, 2), round(std*100, 2)

        return self.ef_graph(mean_returns, cov_matrix, returns, std, 0.0425)
    
    def __init__ (self, stocks, weights, years = 1):
        """Efficient frontier constructor

        :param stocks: Dataframe containing the assets historical returns
        :type stocks: pd.DataFrame
        :param weights: Current porfolio weights.
        :type weights: list
        :param years: Amount of years to calculate, defaults to 1
        :type years: int, optional
        """
        self.stocks = stocks.div(100)
        self.weights = weights 
        self.years = years