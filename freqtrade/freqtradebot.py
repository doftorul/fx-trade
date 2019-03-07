"""
Freqtrade is the main module of this bot. It contains the class Freqtrade()
"""

import copy
import logging
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import arrow
from requests.exceptions import RequestException

from freqtrade import (DependencyException, OperationalException,
                       TemporaryError, __version__, constants, persistence)
from freqtrade.data.converter import order_book_to_dataframe
from freqtrade.rpc import RPCManager, RPCMessageType
from freqtrade.state import State
from freqtrade.strategy.interface import retrieve_strategy
from freqtrade.wallets import Wallets
from freqtrade.exchange.oanda import Oanda
from multiprocessing import Process, Pool
from libs.factory import DataFactory

logger = logging.getLogger(__name__)


class FreqtradeBot(object):
    """
    Freqtrade is the main class of the bot.
    This is from here the bot start its logic.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Init all variables and objects the bot needs to work
        :param config: configuration dict, you can use Configuration.get_config()
        to get the config dict.
        """

        logger.info(
            'Starting freqtrade %s',
            __version__,
        )

        # Init bot states
        self.state = State.STOPPED

        # Init objects
        self.config = config
        #this should be a class that is initialised
        self.strategy = retrieve_strategy(self.config["strategy"]["name"])
        self.strategy_params = self.config["strategy"]["params"]

        self.rpc: RPCManager = RPCManager(self)

        self.exchange = Oanda(self.config)

        #TODO: ADJUST WALLET CLASS
        self.wallets = Wallets(self.exchange)
        self.dataprovider = self.exchange.get_history

        # Attach Dataprovider to Strategy baseclass
        # IStrategy.dp = self.dataprovider
        # Attach Wallets to Strategy baseclass
        # IStrategy.wallets = self.wallets

        self.pairlists = self.config.get('exchange').get('pair_whitelist', [])
        self.pairlists = [Instrument(pair, "") for pair in self.pairlists]
        self.strategies = [
            self.strategy(
                api=self.exchange, 
                instrument=pair, 
                kwargs=self.strategy_params) for pair in self.pairlists
            ]
        
        # Initializing Edge only if enabled
        # TODO: EDGE COMBINED WITH PORTFOLIO MANAGEMENT STRATEGIES FOR FOREX

        self.portfolio = Portfolio(self.config, self.exchange, self.strategy) if \
            self.config.get('edge', {}).get('enabled', False) else None

        # Set initial application state
        initial_state = self.config.get('initial_state')

        if initial_state:
            self.state = State[initial_state.upper()]
        else:
            self.state = State.STOPPED


    def cleanup(self) -> None:
        """
        Cleanup pending resources on an already stopped bot
        :return: None
        """
        logger.info('Cleaning up modules ...')
        self.rpc.cleanup()
        persistence.cleanup()

    def worker(self, old_state: State = None, idle: int = constants.PROCESS_THROTTLE_SECS) -> State:
        """
        Trading routine that must be run at each loop
        :param old_state: the previous service state from the previous call
        :return: current service state
        """
        # Log state transition
        state = self.state
        if state != old_state:
            self.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': f'{state.name.lower()}'
            })
            logger.info('Changing state to: %s', state.name)
            if state == State.RUNNING:
                self.rpc.startup_messages(self.config, self.pairlists)

        if state == State.STOPPED:
            time.sleep(1)
        elif state == State.RUNNING:
            """
            jobs = []
            for instrument, strategy in zip(self.pairlists, self.strategies):
                p = Process(target=self._process,args=(instrument, strategy))
                jobs.append(p)
            
            for job in jobs:
                job.start()
            
            for job in jobs:
                job.join()
            """
            pool = Pool(processes=len(self.pairlists))
            input_tuple = list(zip(self.pairlists, self.strategies))
            self.pairlists = pool.starmap(self._process, input_tuple)
            pool.close()
            pool.join()

        return state

    

    def _process(self, instrument, strategy) -> bool:
        """
        This is a trade iteration. It checks for new candles every 5 seconds, then performs
        an action. 
        """
        state_changed = False

        self.portfolio.update()

        try:
            while True:
                #strategy.retrieve(instrument) #retrieve data for instrument
                #strategy.idle() #check if the time is updated (this includes a while loop, so remove the outer loop)
                #
                oanda_data = self.dataprovider(instrument.name, 'M1', count=300)  # many data-points to increase EMA and such accuracy
                current_complete_candle_stamp = oanda_data[-1]['time']
                if current_complete_candle_stamp != instrument.latest_time:  # if new candle is complete
                    break
                time.sleep(5)
            instrument.time = current_complete_candle_stamp
        except TemporaryError as error:
            logger.warning(f"Error: {error}, retrying in {constants.RETRY_TIMEOUT} seconds...")
            time.sleep(constants.RETRY_TIMEOUT)
        except OperationalException:
            tb = traceback.format_exc()
            hint = 'Issue `/start` if you think it is safe to restart.'
            #self.rpc.send_msg({
            #    'type': RPCMessageType.STATUS_NOTIFICATION,
            #    'status': f'OperationalException:\n```\n{tb}```{hint}'
            #})
            logger.exception('OperationalException. Stopping trader ...')
            self.state = State.STOPPED
        return state_changed

    

        amount = stake_amount / buy_limit_requested

        order = self.exchange.buy(pair=pair, ordertype=self.strategy.order_types['buy'],
                                  amount=amount, rate=buy_limit_requested,
                                  time_in_force=time_in_force)
        order_id = order['id']
        order_status = order.get('status', None)

        # we assume the order is executed at the price requested
        buy_limit_filled_price = buy_limit_requested

        if order_status == 'expired' or order_status == 'rejected':
            order_type = self.strategy.order_types['buy']
            order_tif = self.strategy.order_time_in_force['buy']

            # return false if the order is not filled
            if float(order['filled']) == 0:
                logger.warning('Buy %s order with time in force %s for %s is %s by %s.'
                               ' zero amount is fulfilled.',
                               order_tif, order_type, pair_s, order_status, self.exchange.name)
                return False
            else:
                # the order is partially fulfilled
                # in case of IOC orders we can check immediately
                # if the order is fulfilled fully or partially
                logger.warning('Buy %s order with time in force %s for %s is %s by %s.'
                               ' %s amount fulfilled out of %s (%s remaining which is canceled).',
                               order_tif, order_type, pair_s, order_status, self.exchange.name,
                               order['filled'], order['amount'], order['remaining']
                               )
                stake_amount = order['cost']
                amount = order['amount']
                buy_limit_filled_price = order['price']
                order_id = None

        # in case of FOK the order may be filled immediately and fully
        elif order_status == 'closed':
            stake_amount = order['cost']
            amount = order['amount']
            buy_limit_filled_price = order['price']
            order_id = None

        self.rpc.send_msg({
            'type': RPCMessageType.BUY_NOTIFICATION,
            'exchange': self.exchange.name.capitalize(),
            'pair': pair_s,
            'market_url': pair_url,
            'limit': buy_limit_filled_price,
            'stake_amount': stake_amount,
            'stake_currency': stake_currency,
            'fiat_currency': fiat_currency
        })

        # Fee is applied twice because we make a LIMIT_BUY and LIMIT_SELL
        fee = self.exchange.get_fee(symbol=pair, taker_or_maker='maker')
        trade = Trade(
            pair=pair,
            stake_amount=stake_amount,
            amount=amount,
            fee_open=fee,
            fee_close=fee,
            open_rate=buy_limit_filled_price,
            open_rate_requested=buy_limit_requested,
            open_date=datetime.utcnow(),
            exchange=self.exchange.id,
            open_order_id=order_id,
            strategy=self.strategy.get_strategy_name(),
            ticker_interval=constants.TICKER_INTERVAL_MINUTES[self.config['ticker_interval']]
        )

        Trade.session.add(trade)
        Trade.session.flush()

        # Updating wallets
        self.wallets.update()

        return True

    def process_maybe_execute_buy(self) -> bool:
        """
        Tries to execute a buy trade in a safe way
        :return: True if executed
        """
        try:
            # Create entity and execute trade
            if self.create_trade():
                return True

            logger.info('Found no buy signals for whitelisted currencies. Trying again..')
            return False
        except DependencyException as exception:
            logger.warning('Unable to create trade: %s', exception)
            return False

    def process_maybe_execute_sell(self, trade: Trade) -> bool:
        """
        Tries to execute a sell trade
        :return: True if executed
        """
        try:
            # Get order details for actual price per unit
            if trade.open_order_id:
                # Update trade with order values
                logger.info('Found open order for %s', trade)
                order = self.exchange.get_order(trade.open_order_id, trade.pair)
                # Try update amount (binance-fix)
                try:
                    new_amount = self.get_real_amount(trade, order)
                    if order['amount'] != new_amount:
                        order['amount'] = new_amount
                        # Fee was applied, so set to 0
                        trade.fee_open = 0

                except OperationalException as exception:
                    logger.warning("Could not update trade amount: %s", exception)

                trade.update(order)

            if self.strategy.order_types.get('stoploss_on_exchange') and trade.is_open:
                result = self.handle_stoploss_on_exchange(trade)
                if result:
                    self.wallets.update()
                    return result

            if trade.is_open and trade.open_order_id is None:
                # Check if we can sell our current pair
                result = self.handle_trade(trade)

                # Updating wallets if any trade occured
                if result:
                    self.wallets.update()

                return result

        except DependencyException as exception:
            logger.warning('Unable to sell trade: %s', exception)
        return False

    def get_real_amount(self, trade: Trade, order: Dict) -> float:
        """
        Get real amount for the trade
        Necessary for self.exchanges which charge fees in base currency (e.g. binance)
        """
        order_amount = order['amount']
        # Only run for closed orders
        if trade.fee_open == 0 or order['status'] == 'open':
            return order_amount

        # use fee from order-dict if possible
        if 'fee' in order and order['fee'] and (order['fee'].keys() >= {'currency', 'cost'}):
            if trade.pair.startswith(order['fee']['currency']):
                new_amount = order_amount - order['fee']['cost']
                logger.info("Applying fee on amount for %s (from %s to %s) from Order",
                            trade, order['amount'], new_amount)
                return new_amount

        # Fallback to Trades
        trades = self.exchange.get_trades_for_order(trade.open_order_id, trade.pair,
                                                    trade.open_date)

        if len(trades) == 0:
            logger.info("Applying fee on amount for %s failed: myTrade-Dict empty found", trade)
            return order_amount
        amount = 0
        fee_abs = 0
        for exectrade in trades:
            amount += exectrade['amount']
            if "fee" in exectrade and (exectrade['fee'].keys() >= {'currency', 'cost'}):
                # only applies if fee is in quote currency!
                if trade.pair.startswith(exectrade['fee']['currency']):
                    fee_abs += exectrade['fee']['cost']

        if amount != order_amount:
            logger.warning(f"Amount {amount} does not match amount {trade.amount}")
            raise OperationalException("Half bought? Amounts don't match")
        real_amount = amount - fee_abs
        if fee_abs != 0:
            logger.info(f"Applying fee on amount for {trade} "
                        f"(from {order_amount} to {real_amount}) from Trades")
        return real_amount

    def handle_trade(self, trade: Trade) -> bool:
        """
        Sells the current pair if the threshold is reached and updates the trade record.
        :return: True if trade has been sold, False otherwise
        """
        if not trade.is_open:
            raise ValueError(f'Attempt to handle closed trade: {trade}')

        logger.debug('Handling %s ...', trade)

        (buy, sell) = (False, False)
        experimental = self.config.get('experimental', {})
        if experimental.get('use_sell_signal') or experimental.get('ignore_roi_if_buy_signal'):
            (buy, sell) = self.strategy.get_signal(
                trade.pair, self.strategy.ticker_interval,
                self.dataprovider.ohlcv(trade.pair, self.strategy.ticker_interval))

        config_ask_strategy = self.config.get('ask_strategy', {})
        if config_ask_strategy.get('use_order_book', False):
            logger.info('Using order book for selling...')
            # logger.debug('Order book %s',orderBook)
            order_book_min = config_ask_strategy.get('order_book_min', 1)
            order_book_max = config_ask_strategy.get('order_book_max', 1)

            order_book = self.exchange.get_order_book(trade.pair, order_book_max)

            for i in range(order_book_min, order_book_max + 1):
                order_book_rate = order_book['asks'][i - 1][0]
                logger.info('  order book asks top %s: %0.8f', i, order_book_rate)
                sell_rate = order_book_rate

                if self.check_sell(trade, sell_rate, buy, sell):
                    return True

        else:
            logger.debug('checking sell')
            sell_rate = self.exchange.get_ticker(trade.pair)['bid']
            if self.check_sell(trade, sell_rate, buy, sell):
                return True

        logger.debug('Found no sell signal for %s.', trade)
        return False

    def handle_stoploss_on_exchange(self, trade: Trade) -> bool:
        """
        Check if trade is fulfilled in which case the stoploss
        on exchange should be added immediately if stoploss on exchange
        is enabled.
        """

        result = False

        # If trade is open and the buy order is fulfilled but there is no stoploss,
        # then we add a stoploss on exchange
        if not trade.open_order_id and not trade.stoploss_order_id:
            if self.edge:
                stoploss = self.edge.stoploss(pair=trade.pair)
            else:
                stoploss = self.strategy.stoploss

            stop_price = trade.open_rate * (1 + stoploss)

            # limit price should be less than stop price.
            # 0.99 is arbitrary here.
            limit_price = stop_price * 0.99

            stoploss_order_id = self.exchange.stoploss_limit(
                pair=trade.pair, amount=trade.amount, stop_price=stop_price, rate=limit_price
            )['id']
            trade.stoploss_order_id = str(stoploss_order_id)
            trade.stoploss_last_update = datetime.now()

        # Or the trade open and there is already a stoploss on exchange.
        # so we check if it is hit ...
        elif trade.stoploss_order_id:
            logger.debug('Handling stoploss on exchange %s ...', trade)
            order = self.exchange.get_order(trade.stoploss_order_id, trade.pair)
            if order['status'] == 'closed':
                trade.sell_reason = SellType.STOPLOSS_ON_EXCHANGE.value
                trade.update(order)
                result = True
            elif self.config.get('trailing_stop', False):
                # if trailing stoploss is enabled we check if stoploss value has changed
                # in which case we cancel stoploss order and put another one with new
                # value immediately
                self.handle_trailing_stoploss_on_exchange(trade, order)

        return result

    def handle_trailing_stoploss_on_exchange(self, trade: Trade, order):
        """
        Check to see if stoploss on exchange should be updated
        in case of trailing stoploss on exchange
        :param Trade: Corresponding Trade
        :param order: Current on exchange stoploss order
        :return: None
        """

        if trade.stop_loss > float(order['info']['stopPrice']):
            # we check if the update is neccesary
            update_beat = self.strategy.order_types.get('stoploss_on_exchange_interval', 60)
            if (datetime.utcnow() - trade.stoploss_last_update).total_seconds() > update_beat:
                # cancelling the current stoploss on exchange first
                logger.info('Trailing stoploss: cancelling current stoploss on exchange '
                            'in order to add another one ...')
                if self.exchange.cancel_order(order['id'], trade.pair):
                    # creating the new one
                    stoploss_order_id = self.exchange.stoploss_limit(
                        pair=trade.pair, amount=trade.amount,
                        stop_price=trade.stop_loss, rate=trade.stop_loss * 0.99
                    )['id']
                    trade.stoploss_order_id = str(stoploss_order_id)

    def check_sell(self, trade: Trade, sell_rate: float, buy: bool, sell: bool) -> bool:
        if self.edge:
            stoploss = self.edge.stoploss(trade.pair)
            should_sell = self.strategy.should_sell(
                trade, sell_rate, datetime.utcnow(), buy, sell, force_stoploss=stoploss)
        else:
            should_sell = self.strategy.should_sell(trade, sell_rate, datetime.utcnow(), buy, sell)

        if should_sell.sell_flag:
            self.execute_sell(trade, sell_rate, should_sell.sell_type)
            logger.info('executed sell, reason: %s', should_sell.sell_type)
            return True
        return False

    def check_handle_timedout(self) -> None:
        """
        Check if any orders are timed out and cancel if neccessary
        :param timeoutvalue: Number of minutes until order is considered timed out
        :return: None
        """
        buy_timeout = self.config['unfilledtimeout']['buy']
        sell_timeout = self.config['unfilledtimeout']['sell']
        buy_timeoutthreashold = arrow.utcnow().shift(minutes=-buy_timeout).datetime
        sell_timeoutthreashold = arrow.utcnow().shift(minutes=-sell_timeout).datetime

        for trade in Trade.query.filter(Trade.open_order_id.isnot(None)).all():
            try:
                # FIXME: Somehow the query above returns results
                # where the open_order_id is in fact None.
                # This is probably because the record got
                # updated via /forcesell in a different thread.
                if not trade.open_order_id:
                    continue
                order = self.exchange.get_order(trade.open_order_id, trade.pair)
            except (RequestException, DependencyException):
                logger.info(
                    'Cannot query order for %s due to %s',
                    trade,
                    traceback.format_exc())
                continue
            ordertime = arrow.get(order['datetime']).datetime

            # Check if trade is still actually open
            if float(order['remaining']) == 0.0:
                self.wallets.update()
                continue

            # Handle cancelled on exchange
            if order['status'] == 'canceled':
                if order['side'] == 'buy':
                    self.handle_buy_order_full_cancel(trade, "canceled on Exchange")
                elif order['side'] == 'sell':
                    self.handle_timedout_limit_sell(trade, order)
                    self.wallets.update()
            # Check if order is still actually open
            elif order['status'] == 'open':
                if order['side'] == 'buy' and ordertime < buy_timeoutthreashold:
                    self.handle_timedout_limit_buy(trade, order)
                    self.wallets.update()
                elif order['side'] == 'sell' and ordertime < sell_timeoutthreashold:
                    self.handle_timedout_limit_sell(trade, order)
                    self.wallets.update()

    def handle_buy_order_full_cancel(self, trade: Trade, reason: str) -> None:
        """Close trade in database and send message"""
        Trade.session.delete(trade)
        Trade.session.flush()
        logger.info('Buy order %s for %s.', reason, trade)
        self.rpc.send_msg({
            'type': RPCMessageType.STATUS_NOTIFICATION,
            'status': f'Unfilled buy order for {trade.pair} {reason}'
        })

    def handle_timedout_limit_buy(self, trade: Trade, order: Dict) -> bool:
        """Buy timeout - cancel order
        :return: True if order was fully cancelled
        """
        self.exchange.cancel_order(trade.open_order_id, trade.pair)
        if order['remaining'] == order['amount']:
            # if trade is not partially completed, just delete the trade
            self.handle_buy_order_full_cancel(trade, "cancelled due to timeout")
            return True

        # if trade is partially complete, edit the stake details for the trade
        # and close the order
        trade.amount = order['amount'] - order['remaining']
        trade.stake_amount = trade.amount * trade.open_rate
        trade.open_order_id = None
        logger.info('Partial buy order timeout for %s.', trade)
        self.rpc.send_msg({
            'type': RPCMessageType.STATUS_NOTIFICATION,
            'status': f'Remaining buy order for {trade.pair} cancelled due to timeout'
        })
        return False

    def handle_timedout_limit_sell(self, trade: Trade, order: Dict) -> bool:
        """
        Sell timeout - cancel order and update trade
        :return: True if order was fully cancelled
        """
        if order['remaining'] == order['amount']:
            # if trade is not partially completed, just cancel the trade
            if order["status"] != "canceled":
                reason = "due to timeout"
                self.exchange.cancel_order(trade.open_order_id, trade.pair)
                logger.info('Sell order timeout for %s.', trade)
            else:
                reason = "on exchange"
                logger.info('Sell order canceled on exchange for %s.', trade)
            trade.close_rate = None
            trade.close_profit = None
            trade.close_date = None
            trade.is_open = True
            trade.open_order_id = None
            self.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': f'Unfilled sell order for {trade.pair} cancelled {reason}'
            })

            return True

        # TODO: figure out how to handle partially complete sell orders
        return False

    def execute_sell(self, trade: Trade, limit: float, sell_reason: SellType) -> None:
        """
        Executes a limit sell for the given trade and limit
        :param trade: Trade instance
        :param limit: limit rate for the sell order
        :param sellreason: Reason the sell was triggered
        :return: None
        """
        sell_type = 'sell'
        if sell_reason in (SellType.STOP_LOSS, SellType.TRAILING_STOP_LOSS):
            sell_type = 'stoploss'

        # if stoploss is on exchange and we are on dry_run mode,
        # we consider the sell price stop price
        if self.config.get('dry_run', False) and sell_type == 'stoploss' \
           and self.strategy.order_types['stoploss_on_exchange']:
            limit = trade.stop_loss

        # First cancelling stoploss on exchange ...
        if self.strategy.order_types.get('stoploss_on_exchange') and trade.stoploss_order_id:
            self.exchange.cancel_order(trade.stoploss_order_id, trade.pair)

        # Execute sell and update trade record
        order_id = self.exchange.sell(pair=str(trade.pair),
                                      ordertype=self.strategy.order_types[sell_type],
                                      amount=trade.amount, rate=limit,
                                      time_in_force=self.strategy.order_time_in_force['sell']
                                      )['id']

        trade.open_order_id = order_id
        trade.close_rate_requested = limit
        trade.sell_reason = sell_reason.value

        profit_trade = trade.calc_profit(rate=limit)
        current_rate = self.exchange.get_ticker(trade.pair)['bid']
        profit_percent = trade.calc_profit_percent(limit)
        pair_url = self.exchange.get_pair_detail_url(trade.pair)
        gain = "profit" if profit_percent > 0 else "loss"

        msg = {
            'type': RPCMessageType.SELL_NOTIFICATION,
            'exchange': trade.exchange.capitalize(),
            'pair': trade.pair,
            'gain': gain,
            'market_url': pair_url,
            'limit': limit,
            'amount': trade.amount,
            'open_rate': trade.open_rate,
            'current_rate': current_rate,
            'profit_amount': profit_trade,
            'profit_percent': profit_percent,
            'sell_reason': sell_reason.value
        }

        # For regular case, when the configuration exists
        if 'stake_currency' in self.config and 'fiat_display_currency' in self.config:
            stake_currency = self.config['stake_currency']
            fiat_currency = self.config['fiat_display_currency']
            msg.update({
                'stake_currency': stake_currency,
                'fiat_currency': fiat_currency,
            })

        # Send the message
        self.rpc.send_msg(msg)
        Trade.session.flush()
