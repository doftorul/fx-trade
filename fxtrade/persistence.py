from peewee import *
from datetime import datetime
import os 


def to_datetime(s):
    return datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f')

PERSISTENCE_DIR = "database"

proxy = Proxy()

class Transactions(Model):
    trade_id = IntegerField()
    instrument = CharField()
    position = CharField()
    profit = FloatField()
    balance = FloatField()
    time = DateTimeField()

    class Meta:
        database = proxy

class Decisions(Model):
    instrument = CharField()
    position = CharField()
    time = DateTimeField()

    class Meta:
        database = proxy

class Opened(Model):
    trade_id = IntegerField()
    batch_id = IntegerField()
    instrument = CharField()
    price = FloatField()
    position = CharField()
    time = DateTimeField()

    class Meta:
        database = proxy

class Persistor(object):
    def __init__(self):

        # name = 'history_{}.db'.format(datetime.now().strftime("%d%m%Y"))
        self.database_name = 'database.db'
        self.data_dir =  os.path.join(PERSISTENCE_DIR, self.database_name)

        if not os.path.exists(self.data_dir):
            database = SqliteDatabase(self.data_dir)
            # database.connect()
            proxy.initialize(database)
            database.create_tables([Opened, Transactions, Decisions])
            # database.close()
            

    def store_transactions(self, data):

        # name = 'history_{}.db'.format(datetime.now().strftime("%d%m%Y"))
        # data_dir =  os.path.join(PERSISTENCE_DIR, name)

        # if not os.path.exists(data_dir):
        #     database = SqliteDatabase(data_dir)
        #     proxy.initialize(database)
        #     database.create_tables([Opened, Transactions], safe=True)
        # else:
        database = SqliteDatabase(self.data_dir)
        proxy.initialize(database)

        transactions = data["transactions"] if data else []
        since_id = data["lastTransactionID"]

        with proxy.atomic():
            for transaction in transactions:
                if ((transaction.get("reason", "") == "MARKET_ORDER_TRADE_CLOSE") or
                    ((transaction.get("type", "") == "ORDER_FILL") and
                    (transaction.get("reason", "") == "STOP_LOSS_ORDER")) or 
                    ((transaction.get("type", "") == "ORDER_FILL") and
                    (transaction.get("reason", "") == "TAKE_PROFIT_ORDER"))):
                    Transactions.create(
                        trade_id = transaction["id"],
                        instrument = transaction["instrument"],
                        position = "SHORT" if int(transaction["units"]) <0 else "LONG",
                        profit = transaction["pl"],
                        balance = transaction["accountBalance"],
                        time = to_datetime(transaction["time"][:-4])
                        ).save()
                if transaction.get("type", "") == "DAILY_FINANCING":
                    Transactions.create(
                        trade_id = transaction["id"],
                        instrument = "DAILY_FEE",
                        position = "DAILY_FEE",
                        profit = transaction["financing"],
                        balance = transaction["accountBalance"],
                        time = to_datetime(transaction["time"][:-4])
                        ).save()
        
        return since_id
                    

    def get_last_transaction_id(self):

        # data is a datetime.date object datetime.date(YYYY, MM, )
        database = SqliteDatabase(self.data_dir)
        proxy.initialize(database)
        with proxy.atomic():
            last_transaction_id = Transactions.select().order_by(Transactions.trade_id.desc()).get()
        return last_transaction_id.trade_id
        
        
    def store_opened(self, data):
        # name = 'history_{}.db'.format(datetime.now().strftime("%d%m%Y"))
        # data_dir =  os.path.join(PERSISTENCE_DIR, name)

        # if not os.path.exists(data_dir):
        #     database = SqliteDatabase(data_dir)
        #     proxy.initialize(database)
        #     database.create_tables([Opened, Transactions], safe=True)
        # else:
        database = SqliteDatabase(self.data_dir)
        proxy.initialize(database)

        with proxy.atomic():
            for d in data:
                if d:
                    Opened.create(
                        trade_id = d["tradeID"],
                        batch_id = d["batchID"],
                        instrument = d["instrument"],
                        price = d["price"],
                        position = d["type"],
                        time = d["time"]#.replace("T", " "),
                        ).save()


    def store_decisions(self, data):
        # name = 'history_{}.db'.format(datetime.now().strftime("%d%m%Y"))
        # data_dir =  os.path.join(PERSISTENCE_DIR, name)

        # if not os.path.exists(data_dir):
        #     database = SqliteDatabase(data_dir)
        #     proxy.initialize(database)
        #     database.create_tables([Opened, Transactions], safe=True)
        # else:
        database = SqliteDatabase(self.data_dir)
        proxy.initialize(database)

        with proxy.atomic():
            for d in data:
                if d:
                    Decisions.create(
                        instrument = d["instrument"],
                        position = d["position"],
                        time = d["time"]#.replace("T", " "),
                        ).save()


    def retrieve_transactions_trade_by_date(self, date):
        # data is a datetime.date object datetime.date(YYYY, MM, )
        database = SqliteDatabase(self.data_dir)
        proxy.initialize(database)
        with proxy.atomic():
            result = list(Transactions\
                .select(
                    Transactions.instrument, 
                    Transactions.position, 
                    Transactions.profit, 
                    Transactions.time, 
                    Transactions.balance
                    )\
                .where(
                    (Transactions.time.day == date.day) & 
                    (Transactions.time.year == date.year) & 
                    (Transactions.time.month == date.month)
                    )
                )
        return result


    def retrieve_decisions_by_date(self, date):
        # data is a datetime.date object datetime.date(YYYY, MM, )
        database = SqliteDatabase(self.data_dir)
        proxy.initialize(database)
        with proxy.atomic():
            result = list(Decisions\
                .select(
                    Decisions.instrument, 
                    Decisions.position, 
                    )\
                .where(
                    (Decisions.time.day == date.day) & 
                    (Decisions.time.year == date.year) & 
                    (Decisions.time.month == date.month)
                    )
                )
        return result