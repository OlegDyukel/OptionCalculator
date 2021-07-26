import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from scipy.stats import norm
from datetime import datetime

@st.cache(persist=True)
def d(d_num, future_price, strike, sigma, time_before_expiration, r=0):
    sign = (-1)**(d_num+1)
    return (np.log(future_price/strike) +
            (r + sign*0.5*sigma**2)*time_before_expiration)/(sigma*time_before_expiration**0.5)


@st.cache(persist=True)
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

@st.cache(persist=True)
def exp_opt_price(type_contract, future_price, strike=0):
    sign = type_contract
    expiration_price = max(sign*(future_price - strike), 0)
    return expiration_price

##### points for the chart
@st.cache(persist=True)
def get_undrl_points(limit_down, limit_up, range_out_of_limits=0.05):
    future_points = np.linspace((1-range_out_of_limits)*limit_down,
                                (1+range_out_of_limits)*limit_up, 40)
    return list(future_points)


@st.cache(persist=True)
def get_derivative_points(lst_undrl_prices, type_contract, price, amount=1, strike=0, sigma=0, T=0, r=0):
    if type_contract == 0:   ### for futures
        lst_derivative_prices = [(f - price)*amount for f in lst_undrl_prices]
    else:                    ### for options
        lst_derivative_prices = [(opt_price(type_contract, f, strike, sigma, T, r)-price)*amount
                          for f in lst_undrl_prices]
    return lst_derivative_prices

@st.cache(persist=True)
def get_exp_derivative_points(lst_undrl_prices, type_contract, price, amount=1, strike=0):
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



