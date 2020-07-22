# FxTrade

FxTrade is a ForEx trading bot written in Python, with the possibility to implement strategies (both classical or based on machine learning) and trade directly on Oanda Broker (both practice or real environment). The bot can be easily integrated with Telegram for monitoring and commanding. It is designed to support all major exchanges and be controlled via Telegram. 
If you want to contribute to this project, just let me know.

## Main components

## Run bot

```bash
python main.py
```

## Run training manually

```bash
python train.py
```

## Telegram RPC commands

Telegram is not mandatory. However, this is a great way to control your bot. 

- `/start`: Starts the trader
- `/stop`: Stops the trader
- `/trades`: Show all the open trades
- `/closeall`: Closes all the open trades
- `/report [DD MM YYYY]`: Shows detailed P/L per day
- `/profit [DD MM YYYY]`: Shows statistics for each pair per day
- `/stats [DD MM YYYY]`: Shows decisions statistics for each pair per day
- `/pairs`: Show current tradeable pairs
- `/help`: This help message
- `/train` [D1 M1 D2 M2 G]' : Starts training on all the whitelist pairs, using data with granularity `G`, and rangin from `D1/M1` to `D2/M2` datetime interval.  
- `/reload`: Reload configuration file or a new neural network after training.


## ToDos

- [x] beautify logging telegram
- [x] connect telegram commands with Oanda api
- [x] Parallelize currency processes
- [x] Define MACD strategies and other with techical indicators
- [x] Add more validation methods for stategies
- [x] ML/TL training or statistical modeling
- [ ] add training option in telegram, and automate neural nets reloading process
- [ ] try to use just one network (strategy) object if it is shared between pairs
- [ ] Dockerfile and deployment on server
- [ ] Default configurations and constants

2nd stage
- [ ] Study RL for Edge: This page explains how to use Edge Positioning module in your bot in order to enter into a trade only if the trade has a reasonable win rate and risk reward ratio, and consequently adjust your position size and stoploss. https://www.freqtrade.io/en/latest/edge/


## Considerations

- Optimizing by minimizing cost cross-entropy with target labels works (i.e. predicting price direction) `neural/train_logistic_regression_v1.py, lstm_v1, cnn_v1`, Optimizing by maximizing average return without target labels does not work (i.e. predicting optimal positions allocation). Because of unstable / uneven gradients maybe..?
- Learning rate does make a big difference. Training logistic regression with really small lr converges much better. It's probably a good idea to decrease lr again after a number of iterations.
- Results are terribly (!!!) dependent on randomization. My guess is, because the surface of objective function is very rough, each random initialization of weights and random pick of first training batches leads to new local optima. Therefore, to find a really good fit each model should be trained multiple times.
- Sometimes cost function is jumping up and down like crazy because batches of input are not homogenious (?) (the set of 'rules' by which objective function is optimized changes dramatically from batch to batch). Nonetheless, it slowly moves towards some kind of optima (not always! it might take a few tries of training from the beginning).
- Adjusting hyper-parameters is hard but it seems it might be worth the effort

Training models V2
This time the idea was to:

- Create dozens of features (ta-lib indicators) of varying periods. Roughly there is 80 indicators, some of which can vary in time-periods, so all-in-all it is reasonable to create ~250 features.
- Perform PCA to simplify everything and get rid of similar and unimportant highly correlated features.
- Experiment with polynomials.


Conclusions:
- Given only price and volume data, predicting price direction is not really accurate.
- For predictions to be reasonable more features are needed. For instance sentiment data, other macroeconomic data or whatever.
- If not only possible profitable strategy would be, to use other models like position sizing and carefully entering trades to decrease total transaction costs.


## On training sessions

### 1st experiment: Batched Episodic Advantage GRU Actor Critic

20200331_112831

Batch 32, total batches 216, one currency-pair EUR_USD, time per epoch: 2'

At 4th epoch learning seems to have reached an optimum (able to gain an average of 170 pips - over 16 trades - at each time step, versus a potential average gain of 230 pips).

With one currency pair, 6903 50-step time-series featured (from monday midnight to friday 9PM, one week), granularity of 1 minute, the optimal number of epochs is 2. After 2 epochs there happens loss divergences and overfitting, exploding gradients.


### 2nd experiment: One week Batched Episodic Advantage GRU Actor Critic

20200331_125414

Batch 32, total batches 1502, all the currency-pairs listed in configs , time per epoch: 11'
Penalties for holding position. *So far the best method with highest gain.*

With all the currency pair over the past week, granularity of 1 minute, the optimal number of epochs is 1 or 2. After 2 epochs there happens loss divergences and overfitting, exploding gradients.

### 2nd experiment: 2 weeks Batched Episodic Advantage GRU Actor Critic

20200331_162432

Batch 32, total batches 3005, all the currency-pairs listed in configs , time per epoch: 25'
Penalties for holding position. *So far the best method with highest gain.*

With all the currency pair over the past 2 weeks, granularity of 1 minute, the optimal number of epochs is 1 or 2. After 2 epochs there happens loss divergences and overfitting, exploding gradients.

### 2nd-B experiment: Batched Episodic Advantage GRU Actor Critic (No holding penalty)

20200331_125414

Batch 32, total batches 1502, all the currency-pairs listed in configs , time per epoch: 11'
No penalties for holding position. No relevant improvements.

### 2nd-C experiment: 2 weeks Batched Episodic Advantage GRU (small) Actor Critic

20200331_170531

Batch 32, total batches 3005, all the currency-pairs listed in configs , time per epoch: '
Penalties for holding position. Good but no real improvements.

### 3rd experiment: Batched Episodic Advantage Convolutional Actor Critic

### 4th experiment: Environmental Double GRU DQN 

### 5th experiment: Environmental Double Convolutional DQN 
