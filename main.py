#!/usr/bin/env python3
"""
Main Freqtrade bot script.
Read the documentation to know what cli arguments you need.
"""
import logging
import sys
from argparse import Namespace
from fxtrade import OperationalException
from fxtrade.arguments import Arguments
from fxtrade.configuration import Configuration, set_loggers
from fxtrade.fxtradebot import ForexTradeBot
from fxtrade.state import State
from fxtrade.comm.rpc import RPCMessageType

logger = logging.getLogger('fxtrade')


def main(sysargv):
    """
    This function will initiate the bot and start the trading loop.
    :return: None
    """
    arguments = Arguments(
        sysargv,
        'Free, open source crypto trading bot'
    )
    args = arguments.get_parsed_arg()

    # A subcommand has been issued.
    # Means if Backtesting or Hyperopt have been called we exit the bot
    if hasattr(args, 'func'):
        args.func(args)
        return

    fxtrade = None
    return_code = 1
    try:
        # Load and validate configuration
        config = Configuration(args, None).get_config()
        # Init the bot
        fxtrade = ForexTradeBot(config)

        state = None
        while True:
            state = fxtrade.worker(old_state=state)
            if state == State.RELOAD_CONF:
                fxtrade = reconfigure(fxtrade, args)

    except KeyboardInterrupt:
        logger.info('SIGINT received, aborting ...')
        return_code = 0
    except OperationalException as e:
        logger.error(str(e))
        return_code = 2
    except BaseException:
        logger.exception('Fatal exception!')
    finally:
        if fxtrade:
            fxtrade.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': 'process died'
            })
            fxtrade.cleanup()
        sys.exit(return_code)


def reconfigure(fxtrade: ForexTradeBot, args: Namespace) -> ForexTradeBot:
    """
    Cleans up current instance, reloads the configuration and returns the new instance
    """
    # Clean up current modules
    fxtrade.cleanup()

    # Create new instance
    fxtrade = ForexTradeBot(Configuration(args, None).get_config())
    fxtrade.rpc.send_msg({
        'type': RPCMessageType.STATUS_NOTIFICATION,
        'status': 'config reloaded'
    })
    return fxtrade


if __name__ == '__main__':
    set_loggers()
    main(sys.argv[1:])
