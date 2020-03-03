from strategies import Strategy

class DeepSense(Strategy):
    "DQN approach with DeepSense approximating Q function"
    def __init__(self, *args, **kwargs):
        super(DeepSense, self).__init__(*args, **kwargs)
        pass

    def action(self, candles):
        pass

class A2C(Strategy):
    "Asynchronous Actor-Critic approach"
    def __init__(self, *args, **kwargs):
        super(A2C, self).__init__(*args, **kwargs)
        pass

    def action(self, candles):
        pass

class RRL(Strategy):
    "Recurrent Neural Network basic approach"
    def __init__(self, *args, **kwargs):
        super(RRL, self).__init__(*args, **kwargs)
        pass

    def action(self, candles):
        pass