from factory import Downloader
import pandas as pd
import numpy as np
import plotly

import plotly.graph_objs as go


r = Downloader(token="your_oanda_token_here", environment="practice")
candles = r("EUR_USD", 9, 13, save=True, granularity="S30")
len_data = len(candles)
window=50

candles = r("EUR_USD", 9, 13, save=True, granularity="S30")
len_data = len(candles)

pip_conversion=10000
data = []
for i in range(0, len_data-window):
    data.append(
        (
            candles[i:i+window],
            candles[i+1:i+window+1],
            round(pip_conversion*(candles[i+1:i+window+1][-1][4]-candles[i:i+window][-1][5]),1)
        )
    )

timeseries = data[0][0]

mid = [f[1] for f in timeseries]
ma21 = pd.Series(mid).rolling(window=21).mean().fillna(0).tolist()
ma7 = pd.Series(mid).rolling(window=7).mean().fillna(0).tolist()


ema26 = pd.Series(mid).ewm(span=26).mean().fillna(0).tolist()
ema12 = pd.Series(mid).ewm(span=12).mean().fillna(0).tolist()

macd = np.array(ema12)-np.array(ema26)
macd = (2.*(macd - np.min(macd))/np.ptp(macd)-1).tolist()


x = [i for i in range(50)]
ys = [mid, [m for m in ma7 if m], [m for m in ma21 if m], ema26, ema12]
xs = [x, [l for i,l in enumerate(x) if ma7[i]], [l for i,l in enumerate(x) if ma21[i]], x, x]
names = ['mid_close', 'ma7', 'ma21', 'ema26', 'ema12']

def get_plot_candles(candles, price_type='mid'):
    quotes = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'time' : [],
        'volume' : []
        }

    for i, candle in enumerate(candles):
        quotes['open'].append(float(candle[0]))
        quotes['close'].append(float(candle[1]))
        quotes['high'].append(float(candle[2]))
        quotes['low'].append(float(candle[3]))
        quotes['time'].append(i)
        quotes['volume'].append(float(candle[6]))

    trace = go.Candlestick(x=quotes['time'],
                    open=quotes['open'],
                    high=quotes['high'],
                    low=quotes['low'],
                    close=quotes['close'])
    return trace



def save(xs, ys, names, candles):
    fig = go.Figure()

    for x, y, name in zip(xs, ys, names):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))

    trace = get_plot_candles(candles)
    fig.add_trace(trace)

    plotly.offline.plot(fig, filename = 'filename.html', auto_open=False)

def plotsave(s, t, t2, f, f2):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=[i for i in range(len(s))], y=s, mode="lines"))
    fig.add_trace(go.Scatter(x=[i+(len(s)-1) for i in range(len(t))], y=t, mode="lines"))
    fig.add_trace(go.Scatter(x=[i+(len(s)-1) for i in range(len(t))], y=f, mode="lines"))
    fig.add_trace(go.Scatter(x=[i+(len(s)-1)+50 for i in range(len(t))], y=t2, mode="lines"))
    fig.add_trace(go.Scatter(x=[i+(len(s)-1)+50 for i in range(len(t))], y=f2, mode="lines"))
    plotly.offline.plot(fig, filename = 'filename.html', auto_open=False)


save(xs, ys, names, timeseries)

sd20 = pd.Series(mid).rolling(window=21).std().fillna(0).tolist()
upper_bound = (np.array(ma21)+np.array(sd20)*2).tolist()
lower_bound = (np.array(ma21)-np.array(sd20)*2).tolist()

ema = pd.Series(mid).ewm(com=0.5).mean().fillna(0).tolist()

names.extend(["lower_bound", "upper_bound", "ema"])
ys = [mid, [m for m in ma7 if m], [m for m in ma21 if m], ema26, ema12, [m for m in upper_bound if m], [m for m in lower_bound if m], ema]
xs = [x, [l for i,l in enumerate(x) if ma7[i]], [l for i,l in enumerate(x) if ma21[i]], x, x,  [l for i,l in enumerate(x) if lower_bound[i]], [l for i,l in enumerate(x) if upper_bound[i]], x]

close_fft = np.fft.fft(np.asarray(mid))
close_fft[9:-9] = 0
ifft9 = np.fft.ifft(close_fft.tolist())
# fft_df = pd.DataFrame({'fft':close_fft})
# fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
# fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))