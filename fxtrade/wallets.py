# pragma pylint: disable=W0603
""" Wallet """
import logging
import numpy as np
from typing import Dict, Any, NamedTuple
logger = logging.getLogger(__name__)

# wallet data structure
class Wallet(NamedTuple):
    exchange: str
    currency: str
    free: float = 0
    used: float = 0
    total: float = 0

class Portfolio(object):
    def __init__(self, api, pairlists, stake):
        self.pairlists = pairlists
        self.api = api
        self.stake = stake
    
    def update(self):

        balance = float(self.api.get_balance())
        balance *= self.stake

        balance /= len(self.pairlists)
        balance = int(balance)

        # funds_p = np.random.dirichlet(np.ones(len(self.pairlists)),size=1)

        funds = [balance for _ in range(len(self.pairlists))]

        # return [np.round(f,2) for f in funds[0].tolist()]
        return funds


# TODO: ADJUST THIS CLASS TO HANDLE WALLET OF OANDA INSTEAD OF CRYPTOCURRENCY
class Wallets(object):

    def __init__(self, exchange):
        self.exchange = exchange
        self.wallets: Dict[str, Any] = {}
        self.update()

    def get_free(self, currency) -> float:

        if self.exchange._conf['dry_run']:
            return 999.9

        balance = self.wallets.get(currency)
        if balance and balance.free:
            return balance.free
        else:
            return 0

    def get_used(self, currency) -> float:

        if self.exchange._conf['dry_run']:
            return 999.9

        balance = self.wallets.get(currency)
        if balance and balance.used:
            return balance.used
        else:
            return 0

    def get_total(self, currency) -> float:

        if self.exchange._conf['dry_run']:
            return 999.9

        balance = self.wallets.get(currency)
        if balance and balance.total:
            return balance.total
        else:
            return 0

    def update(self) -> None:
        balances = self.exchange.get_balances()

        for currency in balances:
            self.wallets[currency] = Wallet(
                self.exchange.id,
                currency,
                balances[currency].get('free', None),
                balances[currency].get('used', None),
                balances[currency].get('total', None)
            )

        logger.info('Wallets synced ...')
