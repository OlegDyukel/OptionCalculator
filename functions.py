import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from scipy.stats import norm
from datetime import datetime
import mibian
import plotly.express as px



def d(d_num, future_price, strike, sigma, time_before_expiration, r=0):
    sign = (-1)**(d_num+1)
    return (np.log(future_price/strike) +
            (r + sign*0.5*sigma**2)*time_before_expiration)/(sigma*time_before_expiration**0.5)


def opt_price(type_contract, future_price, strike=0, sigma=0, time_before_expiration=0, r=0):
    sign = type_contract
    underlying_expectations = sign*\
                    norm.cdf(sign*d(1, future_price, strike, sigma, time_before_expiration, r))*\
                              future_price
    strike_expectations = sign*\
                          norm.cdf(sign*d(2, future_price, strike, sigma, time_before_expiration, r))*\
                          strike*np.exp(-r*time_before_expiration)
    theoretical_price = underlying_expectations - strike_expectations
    return theoretical_price


def exp_opt_price(type_contract, future_price, strike=0):
    sign = type_contract
    expiration_price = max(sign*(future_price - strike), 0)
    return expiration_price


######################################################
##### points for the chart
@st.cache(persist=True)
def get_undrl_points(limit_down, limit_up, range_out_of_limits=0.05):
    future_points = np.linspace((1-range_out_of_limits)*limit_down,
                                (1+range_out_of_limits)*limit_up, 40)
    return list(future_points)


@st.cache(persist=True)
def get_price_points(lst_undrl_prices, type_contract, price, amount=1, strike=0, sigma=0, T=0, r=0):
    if type_contract == 0:   ### for futures
        lst_derivative_prices = [(f - price)*amount for f in lst_undrl_prices]
    else:                    ### for options
        lst_derivative_prices = [(opt_price(type_contract, f, strike, sigma, T, r)-price)*amount
                          for f in lst_undrl_prices]
    return lst_derivative_prices

@st.cache(persist=True)
def get_exp_price_points(lst_undrl_prices, type_contract, price, amount=1, strike=0):
    if type_contract == 0:
        lst_derivative_prices = [(f - price)*amount for f in lst_undrl_prices]
    else:
        lst_derivative_prices = [(exp_opt_price(type_contract, f, strike)-price)*amount
                                  for f in lst_undrl_prices]
    return lst_derivative_prices


############## ISS ##########################
def iss_urls():
    url_basic_futures = "https://iss.moex.com/iss/engines/futures/markets/forts/securities.json"
    url_basic_options = "https://iss.moex.com/iss/engines/futures/markets/options/securities.json"
    columns_fut = ["SECID", "SHORTNAME", "PREVSETTLEPRICE", "DECIMALS", "LASTTRADEDATE", "ASSETCODE",
                   "PREVOPENPOSITION",
                   "LASTSETTLEPRICE", "MINSTEP", "STEPPRICE", "LOWLIMIT", "HIGHLIMIT"]
    columns_opt = ["SECID", "SHORTNAME", "PREVSETTLEPRICE", "DECIMALS", "LASTTRADEDATE", "ASSETCODE",
                     "PREVOPENPOSITION", "LASTSETTLEPRICE", "MINSTEP", "STEPPRICE"]

    return {"query_futures_instruments": url_basic_futures + \
                                "?iss.only=securities&securities.columns={}".format(','.join(columns_fut)),
    "query_options_instruments": url_basic_options + \
                                "?iss.only=securities&securities.columns={}".format(','.join(columns_opt))}


def get_option_series_name(s):
    st = re.sub("[CP]A[\d\.\-]+$", "", s)
    return st


def get_option_strike(s):
    st = re.sub("\A\w{1,4}-\d{1,2}.\d{2}M\d{6}[CP]A", "", s)
    return st


def get_option_underlying(option_name):
    st = re.split("M\d{6}", option_name)
    return st[0]


def get_option_type(option_name):
    st = re.search("\d{6}[CP]A\d+", option_name)
    st = re.search("[CP]", st[0])
    if st[0] == 'C': return 'call'
    elif st[0] == 'P': return 'put'
    else: return st[0]


def date_convert(string):
    return datetime.strptime(string, '%Y-%m-%d')


# for the left border sliders in what_if_analysis
def min_vola_time(d):
    min_vola = 0
    min_time = 0
    for key, value in d.items():
        if min_vola == 0 and (value['type'] != 'future'): min_vola = value['original_volatility']
        if min_time == 0 and (value['type'] != 'future'): min_time = value['original_maturity(days)']

        if min_vola > value['original_volatility'] and (value['type'] != 'future'):
            min_vola = value['original_volatility']
        if min_time > value['original_maturity(days)'] and (value['type'] != 'future'):
            min_time = value['original_maturity(days)']

    return [int(np.floor(min_vola)), int(np.floor(min_time))]


def get_volatility(contr_type, undrl_price, strike, maturity_days, price):
    if contr_type == 'call':
        option_for_volat = mibian.BS([undrl_price, strike, 0, maturity_days], callPrice=price)
        return option_for_volat.impliedVolatility
    elif contr_type == 'put':
        option_for_volat = mibian.BS([undrl_price, strike, 0, maturity_days], putPrice=price)
        return option_for_volat.impliedVolatility
    else:   # branch for future
        return 0


def get_plot_stuff(risk_limit_down, risk_limit_up, price_limit_down, price_limit_up, updated_undrl_price):
    fig = px.line()
    ### adding risk limits
    fig.add_vline(x=risk_limit_down, line_width=1, line_dash="dot", line_color="grey",
                  name='risk limit down')
    fig.add_vline(x=risk_limit_up, line_width=1, line_dash="dot", line_color="grey",
                  name='risk limit up')
    ### adding price limits
    fig.add_vline(x=price_limit_down, line_width=1, line_dash="dash", line_color="red",
                  name='price limit down')
    fig.add_vline(x=price_limit_up, line_width=1, line_dash="dash", line_color="red",
                  name='price limit up')
    ### adding underlying price
    fig.add_vline(x=updated_undrl_price, line_width=1, line_dash="dashdot", line_color="black",
                  name='underlying price')
    ### adding horizontal axe
    fig.add_hline(y=0, line_width=1, line_color="black")

    return fig

######################## GREEKS ############################
def get_greeks(contr_type, undrl_price, strike, maturity_days, volatility, amount, r=0.0):
    # calculating greeks
    option = mibian.BS([undrl_price, strike, r, maturity_days], volatility=volatility)

    if contr_type == 'call':
        delta = amount * option.callDelta
        gamma = amount * option.gamma
        vega = amount * option.vega
        theta = amount * option.callTheta
    elif contr_type == 'put':
        delta = amount * option.putDelta
        gamma = amount * option.gamma
        vega = amount * option.vega
        theta = amount * option.putTheta
    else:    # branch for future
        delta = amount
        gamma = 0
        vega = 0
        theta = 0

    return {'delta': round(delta, 3), 'gamma': round(gamma, 6),
            'vega': round(vega, 3), 'theta': round(theta, 3)}


#contr_type, undrl_price, strike, maturity_days, volatility, amount
def get_greek_points(lst_undrl_prices, type_contract, amount, strike, volatility, \
                     maturity_days, r=0.0):
    greeks_dict = {'delta': {}, 'gamma': {}, 'vega': {}, 'theta': {}}
    # calculating greeks
    for f in lst_undrl_prices:
        greeks = get_greeks(type_contract, f, strike, maturity_days, volatility, amount, r)
        greeks_dict['delta'][f] = greeks['delta']
        greeks_dict['gamma'][f] = greeks['gamma']
        greeks_dict['vega'][f] = greeks['vega']
        greeks_dict['theta'][f] = greeks['theta']

    return greeks_dict