from datetime import datetime
import plotly
import plotly.graph_objs as go


def plot_candles(candles, price_type='mid'):
    quotes = {
        'open': [], 
        'high': [], 
        'low': [], 
        'close': [],
        'time' : [],
        'volume' : []
        }

    for candle in candles:
        quotes['open'].append(float(candle[price_type]['o']))
        quotes['high'].append(float(candle[price_type]['h']))
        quotes['low'].append(float(candle[price_type]['l']))
        quotes['close'].append(float(candle[price_type]['c']))
        quotes['time'].append(candle['time'][:19])
        quotes['volume'].append(int(candle['volume']))

    trace = go.Candlestick(x=quotes['time'],
                    open=quotes['open'],
                    high=quotes['high'],
                    low=quotes['low'],
                    close=quotes['close'])

    data = [trace]

    plotly.offline.plot(data, filename="candles_{}.html")


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