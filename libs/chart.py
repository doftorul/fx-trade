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