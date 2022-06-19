import yfinance as yf
import matplotlib.pyplot as pl
import pandas as pd


if __name__ == '__main__':
    ticker = yf.Ticker('INTC')
    opts_list = []
    dp = pd.DataFrame()

    for exp_date in ticker.options:
        print(exp_date)
        opt = ticker.option_chain(date=exp_date)
        opts_list.extend(opt)
        break

    df = pd.concat(opts_list)
    print(df)

    # pl.figure(figsize=(12, 4))
    # pl.subplot(121)
    # tsla_df['Close'].plot(title="GM's stock price")
    # diff = tsla_df['Close'].pct_change()
    # pl.subplot(122)
    # diff.plot(title="GM's percent diff")

    # diffevol(parabolic, [(-10, 10), (-10, 10)], 0.5, 0.7, 10, 21, gengraph(5))
