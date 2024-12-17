import yfinance as yf
from pylab import legend, xlabel, ylabel, sqrt, ylim,\
    sqrt, mean, std, plot, show, figure, cumsum
from numpy import array, zeros, matrix, ones, shape, linspace, hstack, cov, where, full
from pandas import Series, DataFrame
from numpy.linalg import inv

# Portfolio class.
class Portfolio:
    def __init__(self, symbols, start=None, end=None, bench='GLD'):

        # Make sure input is a list
        if type(symbols) != list:
            symbols = [symbols]

        # Create distionary to hold assets.
        self.asset = {}

        # Retrieve assets from data source (IE. Yahoo)
        for symbol in symbols:
            try:
                self.asset[symbol] = yf.download(symbol, start, end)
            except:
                print("Asset " + str(symbol) + " not found!")

        # Get Benchmark asset.
        self.benchmark = yf.download(bench, start, end)
        self.benchmark['Return'] = self.benchmark['Adj Close'].diff()

        # Get returns, beta, alpha, and sharp ratio.
        for symbol in symbols:
            # Get returns.
            self.asset[symbol]['Return'] = self.asset[symbol]['Adj Close'].diff()
            print(symbol)
            # Get Beta.
            A = self.asset[symbol]['Return'].fillna(0)
            B = self.benchmark['Return'].fillna(0)
            self.asset[symbol]['Beta'] = cov(A, B)[0, 1] / cov(A, B)[1, 1]
            # Get Alpha
            self.asset[symbol]['Alpha'] = self.asset[symbol]['Return'] - \
                self.asset[symbol]['Beta'] * self.benchmark['Return']

            # Get Sharpe Ratio
            tmp = self.asset[symbol]['Return']
            self.asset[symbol]['Sharpe'] = \
                sqrt(len(tmp)) * mean(tmp.fillna(0)) / std(tmp.fillna(0))

    def nplot(self, symbol, color='b', nval=0):
        tmp = (self.benchmark if symbol == 'GLD' else self.asset[symbol])['Adj Close']
        tmp = tmp / tmp.iloc[nval:].sum()
        plot(tmp, color=color, label=symbol)

    def betas(self):
        betas = [v['Beta'][0] for k, v in self.asset.items()]
        return Series(betas, index=self.asset.keys())

    def returns(self):
        returns = [v['Return'].dropna() for k, v in self.asset.items()]
        return Series(returns, index=self.asset.keys())

    def cov(self):
        keys, values = self.returns().keys(), self.returns().values.tolist()
        return DataFrame(
            cov(values), index=keys, columns=keys)

    def get_w(self, kind='sharpe'):
        V = self.cov()
        iV = matrix(inv(V))

        if kind == 'characteristic':
            e = matrix(ones(len(self.asset.keys()))).T
        elif kind == 'sharpe':
            suml = [ self.returns()[symbol].sum() for symbol in self.asset.keys()]
            e = matrix(suml).T
        else:
            print('\n  *Error: There is no weighting for kind ' + kind)
            return

        num = iV * e
        denom = e.T * iV * e
        w = array(num / denom).flatten()
        return Series(w, index=self.asset.keys())

    def efficient_frontier_w(self, fp):
        wc = self.get_w(kind='characteristic')
        wq = self.get_w(kind='sharpe')

        fc = self.ret_for_w(wc).sum()
        fq = self.ret_for_w(wq).sum()

        denom = fq - fc
        w = (fq - fp) * wc + (fp - fc) * wq
        return Series(w / denom, index=self.asset.keys())

    def efficient_frontier(self, xi=0.01, xf=4, npts=100, scale=10):
        frontier = linspace(xi, xf, npts)

        i = 0
        weights = full(len(frontier), Series)
        rets = zeros(len(frontier))
        sharpe = zeros(len(frontier))
        for f in frontier:
            w = self.efficient_frontier_w(f)
            weights[i] = w
            print(w)
            tmp = self.ret_for_w(w)
            rets[i] = tmp.sum() * scale
            print(('Return: ', round(rets[i], 1)))
            sharpe[i] = mean(tmp) / std(tmp) * sqrt(len(tmp))
            print(('Sharpe Ratio: ', round(sharpe[i], 2)))
            i += 1
        risk = rets/sharpe

        max_sharpe_ratio = sharpe.max()
        max_return = rets.max()

        max_sharpe_index = where(sharpe == max_sharpe_ratio)[0][0]
        max_return_index = where(rets == max_return)[0][0]
        risk_at_max_return = rets[max_return_index] / sharpe[max_return_index]

        print(('Max Sharpe Index: ', max_sharpe_index))
        print(('Max Return Index: ', max_return_index))
        print(('Returns from Max Sharpe: ', rets[max_sharpe_index]))
        print(('Weights from Max Sharpe: ', weights[max_sharpe_index]))
        print(('Returns from Max Return: ', rets[max_return_index]))
        print(('Weights from Max Return: ', weights[max_return_index]))

        return Series(rets, index=risk), max_sharpe_ratio, risk_at_max_return, weights[max_sharpe_index], weights[max_return_index]

    def efficient_frontier_plot(self, xi=0.01, xf=4, npts=100, scale=0.1,
                                col1='b', col2='r', newfig=1, plabel=''):
        eff, m = self.efficient_frontier()

        print(('Max Sharpe Ratio: ', round(m, 2)))

        if newfig == 1:
            figure()

        plot(array(eff.index), array(eff), col1, linewidth=2,
             label="Efficient Frontier " + plabel)
        tmp = zeros(1)
        plot(hstack((tmp, array(eff.index))),
             hstack((0, m * array(eff.index))),
             col2, linewidth=2, label="Max Sharpe Ratio: %6.2g" % m)
        legend(loc='best', shadow=True, fancybox=True)
        xlabel('Risk %', fontsize=16)
        ylabel('Return %', fontsize=16)
        show()

    def min_var_w_ret(self, ret):
        V = self.cov()
        suml = [self.returns()[symbol].sum() for symbol in self.asset.keys()] 
        e = matrix(suml).T
        iV = matrix(inv(V))
        num = iV * e * ret
        denom = e.T * iV * e
        return Series(array(num / denom).flatten(), index=self.asset.keys())

    def ret_for_w(self, w):
        tmp = self.returns()
        tmpl = [v * wi for v, wi in zip(tmp.values.tolist(), w) ] 
        return Series(tmpl, index=tmp.keys()).sum()
