from time import sleep

class Bot(object):
    def __init__(self, api, strategy, spec):
        self.api = api
        self.strategy = strategy

        self.info = self.api.account_details()

        self.balance = self.info['balance']
        self.account_currency = self.info['accountCurrency']
        self.leverage = 1./self.info['marginRate']


        self.instrument = self.strategy.instrument.upper()
        self.api.close_position(self.instrument)

        self.stop_loss = spec.stop_loss
        self.take_profit = spec.take_profit
        self.max_units = spec.max_units

    def trade(self, max_time=60*60*12): ##max time 12 hours
        time = 0.
        units = 0

        #TODO this need to deal with the real automated trading. 
        #the overall operation needs to be adjusted

        while time < max_time:
            action = self.strategy.action()

            if action == 'buy':
                units_to_buy = self.max_units - units

                if units_to_buy > 0:
                    self.api.create_order(self.instrument, units_to_buy, self.stop_loss, self.take_profit)
                    print('Bought {} units of {}'.format(self.max_units - units, self.instrument))
                    units = units_to_buy
                else:
                    print('No action taken.')

            elif action == 'sell':
                units_to_sell = abs(units + self.max_units)

                if units_to_sell > 0:
                    self.api.create_order(self.instrument, -units_to_sell, self.stop_loss, self.take_profit)
                    print('Sold {} units of {}'.format(units_to_sell, self.instrument))
                    units = units_to_sell
                else:
                    print('No action taken')

            elif action == 'hold':
                print('No action taken.')

            sleep(60)
            time += 60.

        self.api.close_position(self.strategy.instrument)