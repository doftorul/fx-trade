i = [
            "EUR_USD",
            "GBP_USD",
            "AUD_USD",
            "NZD_USD",
            "USD_CAD",
            "USD_JPY",
            "USD_CHF"
        ]
from fxtrade.data.factory import Downloader
api = Downloader(token="your_oanda_token_here", environment="practice")
data = api.multi_assets_builder(weeks=2, instruments=i)

p = data[0].pct_change()
p = data[0].pct_change(fill_method="bfill")
p100 = p*100
 

import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.graph_objs import *


corrmat = data[0].corr()


trace = {"type":"heatmap", "x": corrmat.columns.tolist(), "y": corrmat.columns.tolist(), "z" : np.asarray(corrmat).tolist()}

fig = Figure(data=[trace], layout={"title": "Features Correlation Matrix"})
plotly.offline.plot(fig, filename = 'filename.html', auto_open=False)