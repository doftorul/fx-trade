import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize

def norm(l):
    x = np.array(l)
    x = normalize(x[:,np.newaxis], axis=0).ravel()
    return x.tolist()


def add_features(timeseries):

    o = norm([t[0] for t in timeseries])
    c = norm([t[1] for t in timeseries])
    h = norm([t[2] for t in timeseries])
    l = norm([t[3] for t in timeseries])
    v = norm([t[4] for t in timeseries])

    ma7 = pd.Series(c).rolling(window=7).mean().fillna(0).tolist()
    ma21 = pd.Series(c).rolling(window=21).mean().fillna(0).tolist()
    ema26 = pd.Series(c).ewm(span=26).mean().fillna(0).tolist()
    ema12 = pd.Series(c).ewm(span=12).mean().fillna(0).tolist()

    # normalized mac
    # macd = np.array(ema12)-np.array(ema26)
    # macd = (2.*(macd - np.min(macd))/np.ptp(macd)-1).tolist()

    #sd20 = pd.Series(mid).rolling(window=20).std().fillna(0).tolist()
    #upper_bound = (np.array(ma21)+np.array(sd20)*2).tolist()
    #lower_bound = (np.array(ma21)-np.array(sd20)*2).tolist()


    #ema = pd.Series(mid).ewm(com=0.5).mean().fillna(0).tolist()


    timeseries_featured = [list (x) for x in zip(o, c, h, l, v, ma7, ma21, ema26, ema12)]
    # for t, x in zip(timeseries[:-1], zip(ma7, ma21, ema26, ema12, lower_bound, upper_bound, ema), timeseries[-1:]):
    # for x in zip(timeseries[:-1], zip(ma7, ma21, ema26, ema12)):
    #     timeseries_featured.append(t+list(x))

    

    return timeseries_featured