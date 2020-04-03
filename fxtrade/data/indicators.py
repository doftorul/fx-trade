import pandas as pd
import numpy as np

def add_features(timeseries):

    mid = [t[1] for t in timeseries]

    ma7 = pd.Series(mid).rolling(window=7).mean().fillna(0).tolist()
    ma21 = pd.Series(mid).rolling(window=21).mean().fillna(0).tolist()
    ema26 = pd.Series(mid).ewm(span=26).mean().fillna(0).tolist()
    ema12 = pd.Series(mid).ewm(span=12).mean().fillna(0).tolist()

    # normalized mac
    # macd = np.array(ema12)-np.array(ema26)
    # macd = (2.*(macd - np.min(macd))/np.ptp(macd)-1).tolist()

    sd20 = pd.Series(mid).rolling(window=20).std().fillna(0).tolist()
    upper_bound = (np.array(ma21)+np.array(sd20)*2).tolist()
    lower_bound = (np.array(ma21)-np.array(sd20)*2).tolist()


    ema = pd.Series(mid).ewm(com=0.5).mean().fillna(0).tolist()


    timeseries_featured = []
    for t, x in zip(timeseries, zip(ma7, ma21, ema26, ema12, lower_bound, upper_bound, ema)):
        timeseries_featured.append(t+list(x))

    return timeseries_featured