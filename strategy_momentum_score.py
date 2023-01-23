# Copyright 2020 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from moonshot import Moonshot
from moonshot.commission import PerShareCommission

_SID_SNP500 = "FIBBG000BDTBL9"

def get_return(df, interval):
    return df / df.shift(interval) - 1.0

def get_momentum(df, interval_return, interval_vol):
    ret = get_return(df, interval_return)
    daily_return = df/df.shift(1) - 1
    vol = daily_return.rolling(interval_vol).std()
    momentum = ret / (0.05 + vol)
    return momentum

class UpMinusDown(Moonshot):
    """
    Strategy that buys recent winners and sells recent losers.

    Specifically:

    - rank stocks by their performance over the past MOMENTUM_WINDOW days
    - ignore very recent performance by excluding the last RANKING_PERIOD_GAP
    days from the ranking window (as commonly recommended for UMD)
    - buy the TOP_N_PCT percent of highest performing stocks and short the TOP_N_PCT
    percent of lowest performing stocks
    - rebalance the portfolio according to REBALANCE_INTERVAL
    """

    CODE = "momentum-score"
    MOMENTUM_WINDOW = 100 # rank by twelve-month returns
    VOLATILITY_WINDOW = 40
    RANKING_PERIOD_GAP = 22 # but exclude most recent 1 month performance
    TOP_N_PCT = 50 # Buy/sell the top/bottom 50%
    TOP_N_COUNT = None # Buy/sell the top/bottom 5
    LONG_ONLY = False
    SHORT_ONLY = False
    ESCAPE_LONG = False
    ESCAPE_WEEKLY_CHANGE_LIMIT = 0.1
    REBALANCE_INTERVAL = "M" # M = monthly; 
    PRICE_LOWER_LIMIT = 100
    EWM_COM = None
    
    SNP500_WINDOW = 50
    
    BENCHMARK = _SID_SNP500

    def prices_to_signals(self, prices):
        """
        This method receives a DataFrame of prices and should return a
        DataFrame of integer signals, where 1=long, -1=short, and 0=cash.
        """
        closes = prices.loc["Close"]
        closes = closes.where(closes > self.PRICE_LOWER_LIMIT)
        if self.EWM_COM is not None:
            closes = closes.ewm(com=self.EWM_COM).mean()
            
        df_snp500 = closes[[_SID_SNP500]]
        if self.EWM_COM is not None:
            df_snp500 = df_snp500.ewm(com=self.EWM_COM).mean()
        df_snp500_returns = get_return(df_snp500, self.SNP500_WINDOW)

        # Calculate the returns
        momentum_score = get_momentum(closes, self.MOMENTUM_WINDOW, self.VOLATILITY_WINDOW)
        weekly_returns = get_return(closes, 5)

        if self.TOP_N_PCT is not None:
            # Rank the best and worst
            top_ranks = momentum_score.rank(axis=1, ascending=False, pct=True)
            bottom_ranks = momentum_score.rank(axis=1, ascending=True, pct=True)

            top_n_pct = self.TOP_N_PCT / 100

            # Get long and short signals and convert to 1, 0, -1
            longs = (top_ranks <= top_n_pct)
            shorts = (bottom_ranks <= top_n_pct)
        elif self.TOP_N_COUNT is not None:
            # Rank the best and worst
            top_ranks = momentum_score.rank(axis=1, ascending=False)
            bottom_ranks = momentum_score.rank(axis=1, ascending=True)

            # Get long and short signals and convert to 1, 0, -1
            longs = (top_ranks <= self.TOP_N_COUNT)
            shorts = (bottom_ranks <= self.TOP_N_COUNT)
            
        neutral = longs.copy()
        neutral.loc[:] = 0

        # long only when the weekly return is larger than -ESCAPE_WEEKLY_CHANGE_LIMIT
        longs = neutral.where(weekly_returns < -self.ESCAPE_WEEKLY_CHANGE_LIMIT, longs)
        shorts = neutral.where(weekly_returns > self.ESCAPE_WEEKLY_CHANGE_LIMIT, shorts)
        longs = longs.astype(int)
        shorts = -shorts.astype(int)

        # Combine long and short signals
        if self.LONG_ONLY:
            signals = longs
        elif self.SHORT_ONLY:
            signals = shorts
        else:
            signals = longs.where(longs == 1, shorts)

        # Resample using the rebalancing interval.
        # Keep only the last signal of the month, then fill it forward
        signals = signals.resample(self.REBALANCE_INTERVAL).last()
        signals = signals.reindex(closes.index, method="ffill")

        return signals

    def signals_to_target_weights(self, signals, prices):
        """
        This method receives a DataFrame of integer signals (-1, 0, 1) and
        should return a DataFrame indicating how much capital to allocate to
        the signals, expressed as a percentage of the total capital allocated
        to the strategy (for example, -0.25, 0, 0.1 to indicate 25% short,
        cash, 10% long).
        """
        weights = self.allocate_equal_weights(signals)
        return weights

    def target_weights_to_positions(self, weights, prices):
        """
        This method receives a DataFrame of allocations and should return a
        DataFrame of positions. This allows for modeling the delay between
        when the signal occurs and when the position is entered, and can also
        be used to model non-fills.
        """
        # Enter the position in the period/day after the signal
        return weights.shift()

    def positions_to_gross_returns(self, positions, prices):
        """
        This method receives a DataFrame of positions and a DataFrame of
        prices, and should return a DataFrame of percentage returns before
        commissions and slippage.
        """
        # We'll enter on the open, so our return is today's open to
        # tomorrow's open
        opens = prices.loc["Open"]
        # The return is the security's percent change over the period,
        # multiplied by the position.
        gross_returns = opens.pct_change() * positions.shift()
        return gross_returns

class USStockCommission(PerShareCommission):
    BROKER_COMMISSION_PER_SHARE = 0.005

class UpMinusDownDemo(UpMinusDown):
    CODE = "momentum-score"
    DB = "usstock-1d"
    UNIVERSES = "usstock"
    COMMISSION_CLASS = USStockCommission
