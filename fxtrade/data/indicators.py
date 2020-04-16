import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly

def normalize(X, x_min=None, x_max=None, ran=(0,1)):

    if not x_min : x_min = X.min()
    if not x_max : x_max = X.max()

    nom = X - x_min
    denom = x_max - x_min

    return nom/denom 


def norm_by_latest_close(l, c):
    x = np.array(l)
    x /= c
    return x.tolist()

def compute_trend(target_data, threshold = 0.03, pip_conversion=10000):

    x = np.arange(0,len(target_data))
    coefs = poly.polyfit(x, target_data, 1)
    ffit = poly.polyval(x, coefs)

    reward_pips = (ffit[-1]/ffit[0]-1)*pip_conversion
    slope = (ffit[-1]/ffit[0]-1)*100

    if slope > threshold:
        trend = 'buy'
    elif slope < -threshold:
        trend = 'sell'
    else:
        trend = 'hold'
        reward_pips = 0

    return trend, reward_pips


def add_features(timeseries, scale=False):


    latest_close = timeseries[-1][1]
    # if scale:
    o = norm_by_latest_close([t[0] for t in timeseries], latest_close)
    c = norm_by_latest_close([t[1] for t in timeseries], latest_close)
    h = norm_by_latest_close([t[2] for t in timeseries], latest_close)
    l = norm_by_latest_close([t[3] for t in timeseries], latest_close)
    # v = norm([t[4] for t in timeseries])
    # else:
    #     o = [t[0] for t in timeseries]
    #     c = [t[1] for t in timeseries]
    #     h = [t[2] for t in timeseries]
    #     l = [t[3] for t in timeseries]
    #     v = [t[4] for t in timeseries]

    # ma7 = pd.Series(c).rolling(window=7).mean().fillna(0).tolist()
    # ma21 = pd.Series(c).rolling(window=21).mean().fillna(0).tolist()
    # ema26 = pd.Series(c).ewm(span=26).mean().fillna(0).tolist()
    # ema12 = pd.Series(c).ewm(span=12).mean().fillna(0).tolist()

    # normalized mac
    # macd = np.array(ema12)-np.array(ema26)
    # macd = (2.*(macd - np.min(macd))/np.ptp(macd)-1).tolist()

    #sd20 = pd.Series(mid).rolling(window=20).std().fillna(0).tolist()
    #upper_bound = (np.array(ma21)+np.array(sd20)*2).tolist()
    #lower_bound = (np.array(ma21)-np.array(sd20)*2).tolist()


    #ema = pd.Series(mid).ewm(com=0.5).mean().fillna(0).tolist()


    # timeseries_featured = [list (x) for x in zip(o, c, h, l, v, ma7, ma21, ema26, ema12)]
    timeseries_featured = [list (x) for x in zip(o, c, h, l)]
    # timeseries_featured = [list (x) for x in zip(o, c, h, l, ma7, ma21, ema26, ema12)]
    # for t, x in zip(timeseries[:-1], zip(ma7, ma21, ema26, ema12, lower_bound, upper_bound, ema), timeseries[-1:]):
    # for x in zip(timeseries[:-1], zip(ma7, ma21, ema26, ema12)):
    #     timeseries_featured.append(t+list(x))

    

    return timeseries_featured


def norm_by_latest_close_triplet(timeseries):

    # if scale:
    p1 = norm_by_latest_close([t[0] for t in timeseries], timeseries[-1][0])
    p2 = norm_by_latest_close([t[1] for t in timeseries], timeseries[-1][1])
    p3 = norm_by_latest_close([t[2] for t in timeseries], timeseries[-1][2])
    # v = norm([t[4] for t in timeseries])
    # else:
    #     o = [t[0] for t in timeseries]
    #     c = [t[1] for t in timeseries]
    #     h = [t[2] for t in timeseries]
    #     l = [t[3] for t in timeseries]
    #     v = [t[4] for t in timeseries]

    # ma7 = pd.Series(c).rolling(window=7).mean().fillna(0).tolist()
    # ma21 = pd.Series(c).rolling(window=21).mean().fillna(0).tolist()
    # ema26 = pd.Series(c).ewm(span=26).mean().fillna(0).tolist()
    # ema12 = pd.Series(c).ewm(span=12).mean().fillna(0).tolist()

    # normalized mac
    # macd = np.array(ema12)-np.array(ema26)
    # macd = (2.*(macd - np.min(macd))/np.ptp(macd)-1).tolist()

    #sd20 = pd.Series(mid).rolling(window=20).std().fillna(0).tolist()
    #upper_bound = (np.array(ma21)+np.array(sd20)*2).tolist()
    #lower_bound = (np.array(ma21)-np.array(sd20)*2).tolist()


    #ema = pd.Series(mid).ewm(com=0.5).mean().fillna(0).tolist()


    # timeseries_featured = [list (x) for x in zip(o, c, h, l, v, ma7, ma21, ema26, ema12)]
    timeseries_featured = [list (x) for x in zip(p1, p2, p3)]
    # timeseries_featured = [list (x) for x in zip(o, c, h, l, ma7, ma21, ema26, ema12)]
    # for t, x in zip(timeseries[:-1], zip(ma7, ma21, ema26, ema12, lower_bound, upper_bound, ema), timeseries[-1:]):
    # for x in zip(timeseries[:-1], zip(ma7, ma21, ema26, ema12)):
    #     timeseries_featured.append(t+list(x))

    

    return timeseries_featured