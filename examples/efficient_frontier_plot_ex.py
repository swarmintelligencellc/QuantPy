# Example plotting effienct frontier.

import quantpy as qp
import datetime as dt
from pylab import cumsum, figure, legend, show
from numpy import full

# end = dt.datetime.now()
# start = end - dt.timedelta(days=300)

tickers = [\
  'GLD', 'SRUUF',\
  'NERD', 'PEJ',\
  'V', 'PLTR', 'BLOK', 'NVDA', 'XSD', 'SMH', 'MSFT', 'META', 'GOOGL',\
  'SBIO', 'XBI', 'FBT',\
  'SPY', 'QQQ', 'VYM', 'FLOW',\
  'GRID', 'RNEW', 'ICLN', 'EVX',\
  'MLPX', 'XLE', 'VDE', 'IXC', 'COAL', 'UNG',\
  'SMR', 'NUKZ', 'NLR',\
  'WMT', 'AMZN', 'VDC', 'TAGS', 'FTXG', 'XLP',\
  ]

# Grap portfolio
P = qp.Portfolio(tickers, start='2024-02-21', end='2024-12-16', bench='GLD')

# figure()
# # Make plots of normalized returns.
# for ticker in tickers:
#   P.nplot(ticker)
# legend(loc='best', shadow=True, fancybox=True)
# show()


figure()
# Calculate the returns buying 1 share of everything.
returns_equal_investments = P.ret_for_w(full(len(tickers), 1./len(tickers)))
cumsum(returns_equal_investments).plot(color='r',label='Buy and Hold Equally.')
final_return_equal = cumsum(returns_equal_investments)[-1]

# Find the optimal weighting that yields the same return with minimum variance.
minimized_risk_weights = P.min_var_w_ret(final_return_equal)
returns_minimized_risk = P.ret_for_w(minimized_risk_weights)
cumsum(returns_minimized_risk).plot(label='Same return but min variance.')

# Find the optimal weighting that yields maximum return to risk ratio (sharpe)
efficient_frontier, max_sharpe, risk_at_max_return, max_sharpe_weights, max_return_weights = P.efficient_frontier()

print(('Max Sharpe Ratio: ', round(max_sharpe, 2)))
print(max_sharpe_weights)
returns_max_sharpe = P.ret_for_w(max_sharpe_weights)
cumsum(returns_max_sharpe).plot(label='Max Sharpe Ratio.')
print(returns_max_sharpe)

print(('Risk at Max Returns: ', risk_at_max_return))
print(max_return_weights)
returns_max_returns = P.ret_for_w(max_return_weights)
cumsum(returns_max_returns).plot(label='Max Returns.')

legend(loc='best',shadow=True, fancybox=True)
show()

# Plot effiecent frontier
# P.efficient_frontier_plot()
