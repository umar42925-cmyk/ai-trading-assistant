from data_router.router import get_market_data

data, source = get_market_data("AAPL", "1min")
print("SOURCE:", source)
print("DATA:", data)
