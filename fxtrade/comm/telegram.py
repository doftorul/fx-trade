# pragma pylint: disable=unused-argument, unused-variable, protected-access, invalid-name

"""
This module manage Telegram communication
"""
import logging
from typing import Any, Callable, Dict

from tabulate import tabulate
from telegram import ParseMode, ReplyKeyboardMarkup, Update
from telegram.error import NetworkError, TelegramError
from telegram.ext import CallbackContext, CommandHandler, Updater
import datetime
from fxtrade.__init__ import __version__
from fxtrade.comm.rpc import RPC, RPCException, RPCMessageType

logger = logging.getLogger(__name__)

logger.debug('Included module rpc.telegram ...')


def authorized_only(command_handler: Callable[..., None]) -> Callable[..., Any]:
    """
    Decorator to check if the message comes from the correct chat_id
    :param command_handler: Telegram CommandHandler
    :return: decorated function
    """
    def wrapper(self, *args, **kwargs):
        """ Decorator logic """
        update = kwargs.get('update') or args[0]

        # Reject unauthorized messages
        chat_id = int(self._config['telegram']['chat_id'])

        if int(update.message.chat_id) != chat_id:
            logger.info(
                'Rejected unauthorized message from: %s',
                update.message.chat_id
            )
            return wrapper

        logger.info(
            'Executing handler: %s for chat_id: %s',
            command_handler.__name__,
            chat_id
        )
        try:
            return command_handler(self, *args, **kwargs)
        except BaseException:
            logger.exception('Exception occurred within Telegram module')

    return wrapper


class Telegram(RPC):
    """  This class handles all telegram communication """

    def __init__(self, fxtrade) -> None:
        """
        Init the Telegram call, and init the super class RPC
        :param fxtrade: Instance of a fxtrade bot
        :return: None
        """
        super().__init__(fxtrade)

        self._updater: Updater = None
        self._config = fxtrade.config
        self._init()

    def _init(self) -> None:
        """
        Initializes this module with the given config,
        registers all known command handlers
        and starts polling for message updates
        """
        self._updater = Updater(token=self._config['telegram']['token'], workers=0, use_context=True)

        # Register command handler and start telegram message polling
        handles = [
            CommandHandler('profit', self._profit),
            CommandHandler('report', self._report),
            CommandHandler('trades', self._trades),
            CommandHandler('start', self._start),
            CommandHandler('stop', self._stop),
            CommandHandler('stats', self._stats),
            CommandHandler('help', self._help),
            CommandHandler('closeall', self._close_all),
            CommandHandler('reload', self._reload_conf),
            CommandHandler('pairs', self._whitelist),
            CommandHandler('train', self._train),
        ]
        for handle in handles:
            self._updater.dispatcher.add_handler(handle)
        self._updater.start_polling(
            clean=True,
            bootstrap_retries=-1,
            timeout=30,
            read_latency=60,
        )
        logger.info(
            'rpc.telegram is listening for following commands: %s',
            [h.command for h in handles]
        )

    def cleanup(self) -> None:
        """
        Stops all running telegram threads.
        :return: None
        """
        self._updater.stop()

    def send_msg(self, msg: Dict[str, Any]) -> None:
        """ Send a message to telegram channel """

        if msg['type'] == RPCMessageType.BUY_NOTIFICATION:
            message = ("*LONG:* [{pair}] (`{units}` units @ `{price}`)\n"
                       "TP: `{take_profit:.5f}`\n"
                       "SL: `{stop_loss:.5f}`\n").format(**msg)

        elif msg['type'] == RPCMessageType.SELL_NOTIFICATION:
            message = ("*SHORT:* [{pair}] (`{units}` units @ `{price}`)\n"
                       "TP: `{take_profit:.5f}`\n"
                       "SL: `{stop_loss:.5f}`\n").format(**msg)
        elif msg['type'] == RPCMessageType.HOLD_NOTIFICATION:
            message = ("*HOLD:* `{status}`").format(**msg)

        elif msg['type'] == RPCMessageType.STATUS_NOTIFICATION:
            message = '*Status:* `{status}`'.format(**msg)

        elif msg['type'] == RPCMessageType.WARNING_NOTIFICATION:
            message = '*Warning:* `{status}`'.format(**msg)

        elif msg['type'] == RPCMessageType.CUSTOM_NOTIFICATION:
            message = '{status}'.format(**msg)

        elif msg['type'] == RPCMessageType.IDLE_NOTIFICATION:
            message =  '*Idle:* `{status}`'.format(**msg)

        else:
            raise NotImplementedError('Unknown message type: {}'.format(msg['type']))

        self._send_msg(message)

    @authorized_only
    def _report(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /daily <n>
        Returns a daily profit for the current day, or specify the date
        by inputing "/report DD MM YYYY"
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        date = update.message.text.replace('/report', '').strip()
        report_date = datetime.datetime.utcnow().date()

        if date:
            date_elements = date.split()
            try:
                if len(date_elements) == 1:
                    report_date = report_date.replace(
                        day=int(date_elements[0])
                        )
                if len(date_elements) == 2:
                    report_date = report_date.replace(
                        day=int(date_elements[0]), 
                        month=int(date_elements[1])
                        )
                if len(date_elements) == 3:
                    report_date = report_date.replace(
                        day=int(date_elements[0]), 
                        month=int(date_elements[1]), 
                        year=int(date_elements[2])
                        )
            except:
                pass
            
        try:
            # stats = self._rpc_report(report_date)
            stats = self._rpc_persisted_report(report_date)
            stats_tab = tabulate(stats,
                                 headers=[
                                     'Asset',
                                     'P/L',
                                     'Balance',
                                     'Time'
                                 ],
                                 floatfmt=".4f")
            message = f'<b>Daily report - {report_date.day}/{report_date.month}/{report_date.year} </b>:\n<pre>{stats_tab}</pre>'
            self._send_msg(message, parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _trades(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /trades <n>
        Returnn the open trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        try:
            # stats = self._rpc_report(report_date)
            open_trades = self._rpc_open_trades()
            stats_tab = tabulate(open_trades,
                                 headers=[
                                     'Asset',
                                     'units',
                                     'uP/L'
                                 ],
                                 floatfmt=".4f")
            message = f'<b>Open trades </b>:\n<pre>{stats_tab}</pre>'
            self._send_msg(message, parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e))


    @authorized_only
    def _profit(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /daily <n>
        Returns a daily profit for the current day, or specify the date
        by inputing "/report DD MM YYYY"
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        date = update.message.text.replace('/profit', '').strip()
        report_date = datetime.datetime.utcnow().date()

        if date:
            date_elements = date.split()
            try:
                if len(date_elements) == 1:
                    report_date = report_date.replace(
                        day=int(date_elements[0])
                        )
                if len(date_elements) == 2:
                    report_date = report_date.replace(
                        day=int(date_elements[0]), 
                        month=int(date_elements[1])
                        )
                if len(date_elements) == 3:
                    report_date = report_date.replace(
                        day=int(date_elements[0]), 
                        month=int(date_elements[1]), 
                        year=int(date_elements[2])
                        )
            except:
                pass
            
        try:
            stats = self._rpc_persisted_profit(report_date)

            stats_tab = tabulate(stats,
                                 headers=[
                                     'Asset',
                                     'Overall P/L',
                                 ],
                                 floatfmt=".4f")
            message = f'<b>Daily statistics - {report_date.day}/{report_date.month}/{report_date.year} </b>:\n<pre>{stats_tab}</pre>'
            self._send_msg(message, parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _stats(self, update: Update, context: CallbackContext) -> None:
        date = update.message.text.replace('/stats', '').strip()
        report_date = datetime.datetime.utcnow().date()

        if date:
            date_elements = date.split()
            try:
                if len(date_elements) == 1:
                    report_date = report_date.replace(
                        day=int(date_elements[0])
                        )
                if len(date_elements) == 2:
                    report_date = report_date.replace(
                        day=int(date_elements[0]), 
                        month=int(date_elements[1])
                        )
                if len(date_elements) == 3:
                    report_date = report_date.replace(
                        day=int(date_elements[0]), 
                        month=int(date_elements[1]), 
                        year=int(date_elements[2])
                        )
            except:
                pass
            
        try:
            stats = self._rpc_persisted_decisions(report_date)

            stats_tab = tabulate(stats,
                                 headers=[
                                     'Asset',
                                     'L',
                                     'S',
                                     'H',
                                     '#'
                                 ],
                                 tablefmt="simple")
            message = f'<b>Decisions statistics - {report_date.day}/{report_date.month}/{report_date.year} </b>:\n<pre>{stats_tab}</pre>'
            self._send_msg(message, parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _start(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /start.
        Starts TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc_start()
        self._send_msg('Status: `{status}`'.format(**msg))

    @authorized_only
    def _stop(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /stop.
        Stops TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc_stop()
        self._send_msg('Status: `{status}`'.format(**msg))

    @authorized_only
    def _close_all(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /closeall.
        Close all open trafes
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc_close_all()
        self._send_msg('Status: `{status}`'.format(**msg))

    @authorized_only
    def _reload_conf(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /reload_conf.
        Triggers a config file reload
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc_reload_conf()
        self._send_msg('Status: `{status}`'.format(**msg))

    @authorized_only
    def _whitelist(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /whitelist
        Shows the currently active whitelist
        """
        try:
            stats = self._rpc_whitelist()

            stats_tab = tabulate(stats,
                                 headers=[
                                     'Asset',
                                     'Last traded units',
                                 ],
                                 floatfmt=".4f")
            message = f'<b>Pairs traded </b>:\n<pre>{stats_tab}</pre>'
            self._send_msg(message, parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _train(self, update: Update,  context: CallbackContext) -> None:
        """
        Handler for /train
        """

        self._send_msg(message)


        self._send_msg(message)



    @authorized_only
    def _help(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /help.
        Show commands of the bot
        :param update: message update
        :return: None
        """
        message = "*/start:* `Starts the trader`\n" \
                  "*/stop:* `Stops the trader`\n" \
                  "*/trades:* `Show all the open trades`\n" \
                  "*/closeall:* `Closes all the open trades`\n" \
                  "*/report [DD MM YYYY]:* `Shows detailed P/L per day`\n" \
                  "*/profit [DD MM YYYY]:* `Shows statistics for each pair per day`\n" \
                  "*/stats [DD MM YYYY]:* `Shows decisions statistics for each pair per day`\n" \
                  "*/reload:* `Reload configuration file` \n" \
                  "*/pairs:* `Show current tradeable pairs` \n" \
                  "*/help:* `This help message`\n" \

        self._send_msg(message)


    def _send_msg(self, msg: str, parse_mode: ParseMode = ParseMode.MARKDOWN) -> None:
        """
        Send given markdown message
        :param msg: message
        :param parse_mode: telegram parse mode
        :return: None
        """

        keyboard = [['/profit', '/report', '/trades'],
                    ['/start', '/stop', '/closeall'],
                    ['/reload', '/pairs', '/stats', '/help']]

        reply_markup = ReplyKeyboardMarkup(keyboard)

        try:
            try:
                self._updater.bot.send_message(
                    self._config['telegram']['chat_id'],
                    text=msg,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup
                )
            except NetworkError as network_err:
                # Sometimes the telegram server resets the current connection,
                # if this is the case we send the message again.
                logger.warning(
                    'Telegram NetworkError: %s! Trying one more time.',
                    network_err.message
                )
                self._updater.bot.send_message(
                    self._config['telegram']['chat_id'],
                    text=msg,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup
                )
        except TelegramError as telegram_err:
            logger.warning(
                'TelegramError: %s! Giving up on that message.',
                telegram_err.message
            )
