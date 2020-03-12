import oandapyV20.endpoints.transactions as transactions
from oandapyV20 import API
api = API(access_token="39e41febacb7f696aff65ba23713a553-112e0e75a1018a1ffff575cc1c28d5b0", environment="practice")
params = {"from":"2020-03-10T00:00:00.000000000Z", "to":"2020-03-10T23:59:59.999999999Z"}
accountID = "101-004-6618803-001"
r = transactions.TransactionList(accountID, params=params)
rv = api.request(r)
pages = rv["pages"]

start = pages[0].split("from=")[-1].split("&")[0]
end = pages[-1].split("to=")[-1]

params ={
          "to": int(end),
          "from": int(start)
        }

r = transactions.TransactionIDRange(accountID, params=params)
rv = api.request(r)

import json

with open("activity.json", "w") as rrr:
    json.dump(rv, rrr, indent=5)