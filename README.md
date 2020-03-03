# FX-trade

Freqtrade is a ForEx trading bot written in Python. It is designed to support all major exchanges and be controlled via Telegram. It contains backtesting, plotting and money management tools as well as strategy optimization by machine learning.

## Telegram RPC commands

Telegram is not mandatory. However, this is a great way to control your bot. 

- `/start`: Starts the trader
- `/start XXX_XXX` : start trading XXX_XXX pair
- `/stop`: Stops the trader
- `/stop XXX_XXX` : close and stops trading XXX_XXX pair
- `/open`
- `/pause`


Log other statistical data:
- `/status`: Lists all open trades
- `/count`: Displays number of open trades
- `/profit`: Lists cumulative profit from all finished trades
- `/forcesell <trade_id>|all`: Instantly sells the given trade (Ignoring `minimum_roi`).
- `/performance`: Show performance of each finished trade grouped by pair
- `/balance`: Show account balance per currency
- `/daily <n>`: Shows profit or loss per day, over the last n days
- `/help`: Show help message
- `/version`: Show version


## ToDos

[ ] - beautify logging
[ ] - connect telegram commands with Oanda api
[ ] - Parallelize currency processes
[ ] - Define MACD strategies and other with techical indicators

2nd stage
[ ] - ML/TL training or statistical modeling
[ ] - Integrate news feed with sentiment analysis/keyword detection
[ ] - Test massively with lower granularity