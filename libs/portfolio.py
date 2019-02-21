
class Asset(object):
    def __init__(self, name="EUR_USD", kind="currency"):
        self.name = name
        self.kind = kind

        self.ratio = 0.
    
    def update_ratio(self, new_ratio):
        self.ratio = new_ratio

class Spec(object):
    def __init__(self, instrument="EUR_USD", max_units=10, take_profit=0.1, stop_loss=-0.1):
        self.instrument = instrument
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_units = max_units

class Portfolio(object):
    def __init__(self, api, max_risk_percent=5., asset_names=None):
        self.api = api
        balance = self.api.account_details()["account"]["balance"]
        self.max_risk_percent = max_risk_percent
        self.total_budget = balance*self.max_risk_percent
        """Assets should be a list of Asset class"""
        if asset_names:
            self.asset_names = [a for a in asset_names]
        else:
            self.asset_names = [
                "EUR_USD",
                "GBP_USD",
                "AUD_USD",
                "NZD_USD",
                "USD_CAD",
                "USD_JPY",
                "USD_CHF"
            ]

        self.assets = [Asset(name=a) for a in self.asset_names]
        self.specs = []

    def add_asset(self, asset):
        self.assets.append(asset)

    def get_vector(self):
        return [(a.name, a.ratio) for a in self.assets]

    def update_ratios(self):
        new_ratio = self.compute_new_ratio()
        for i, nr in enumerate(new_ratio):
            self.assets[i].update_ratio(nr)

    def compute_new_ratio(self):
        return 0

    def k_period_simple_return(self):
        pass

    def run(self):
        pass

    def predict(self):
        #TODO give proper specification to trading bot following a winning policy
        for asset in self.asset_names:
            self.specs.append(Spec(asset))

        return self.specs