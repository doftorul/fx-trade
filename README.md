# Freqtrade

Freqtrade is a ForEx trading bot written in Python. It is designed to support all major exchanges and be controlled via Telegram. It contains backtesting, plotting and money management tools as well as strategy optimization by machine learning.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)


## Bot commands

```
usage: main.py [-h] [-v] [--version] [-c PATH] [-d PATH] [-s NAME]
               [--strategy-path PATH] [--customhyperopt NAME]
               [--dynamic-whitelist [INT]] [--db-url PATH]
               {backtesting,edge,hyperopt} ...

Free, open source crypto trading bot

positional arguments:
  {backtesting,edge,hyperopt}
    backtesting         backtesting module
    edge                edge module
    hyperopt            hyperopt module

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         verbose mode (-vv for more, -vvv to get all messages)
  --version             show program\'s version number and exit
  -c PATH, --config PATH
                        specify configuration file (default: config.json)
  -d PATH, --datadir PATH
                        path to backtest data
  -s NAME, --strategy NAME
                        specify strategy class name (default: DefaultStrategy)
  --strategy-path PATH  specify additional strategy lookup path
  --customhyperopt NAME
                        specify hyperopt class name (default:
                        DefaultHyperOpts)
  --dynamic-whitelist [INT]
                        dynamically generate and update whitelist based on 24h
                        BaseVolume (default: 20) DEPRECATED.
  --db-url PATH         Override trades database URL, this is useful if
                        dry_run is enabled or in custom deployments (default:
                        None)
```

## Telegram RPC commands

Telegram is not mandatory. However, this is a great way to control your bot. More details on our [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)

- `/start`: Starts the trader
- `/stop`: Stops the trader
- `/status [table]`: Lists all open trades
- `/count`: Displays number of open trades
- `/profit`: Lists cumulative profit from all finished trades
- `/forcesell <trade_id>|all`: Instantly sells the given trade (Ignoring `minimum_roi`).
- `/performance`: Show performance of each finished trade grouped by pair
- `/balance`: Show account balance per currency
- `/daily <n>`: Shows profit or loss per day, over the last n days
- `/help`: Show help message
- `/version`: Show version
