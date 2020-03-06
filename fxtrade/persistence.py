from peewee import *
from datetime import datetime
import os 

PERSISTENCE_DIR = "database"

proxy = Proxy()

class Closed(Model):
    trade_id = IntegerField()
    instrument = CharField()
    position = CharField()
    profit = FloatField()
    balance = FloatField()
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
            database.create_tables([Opened, Closed])
            # database.close()
            

    def store_closed(self, data):

        # name = 'history_{}.db'.format(datetime.now().strftime("%d%m%Y"))
        # data_dir =  os.path.join(PERSISTENCE_DIR, name)

        # if not os.path.exists(data_dir):
        #     database = SqliteDatabase(data_dir)
        #     proxy.initialize(database)
        #     database.create_tables([Opened, Closed], safe=True)
        # else:
        database = SqliteDatabase(self.data_dir)
        proxy.initialize(database)

        with proxy.atomic():
            for d in data:
                if d:
                    Closed.create(
                        trade_id = d["tradeID"],
                        instrument = d["instrument"],
                        position = d["position"],
                        profit = d["PL"],
                        balance = d["balance"],
                        time = d["time"]#.replace("T", " "),
                        ).save()
        
        
    def store_opened(self, data):
        # name = 'history_{}.db'.format(datetime.now().strftime("%d%m%Y"))
        # data_dir =  os.path.join(PERSISTENCE_DIR, name)

        # if not os.path.exists(data_dir):
        #     database = SqliteDatabase(data_dir)
        #     proxy.initialize(database)
        #     database.create_tables([Opened, Closed], safe=True)
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


