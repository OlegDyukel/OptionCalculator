import streamlit as st
import pandas as pd
import requests
import numpy as np
import mibian
import plotly.express as px
from datetime import datetime


from collections import OrderedDict

import SessionState

from functions import get_undrl_points, get_derivative_points, get_exp_derivative_points, iss_urls, \
    get_option_underlying, get_option_series_name, get_option_strike, get_option_type

@st.cache
def get_fut_data():
    ######### futures data
    r_fut = requests.get(iss_urls()["query_futures_instruments"])
    data_fut = r_fut.json()["securities"]["data"]
    columns_fut = r_fut.json()["securities"]["columns"]
    df_fut = pd.DataFrame(data_fut, columns=columns_fut)

    # getting absolute price limits
    df_fut['ABS_LOW_LIMIT'] = df_fut['LASTSETTLEPRICE'] - df_fut['LOWLIMIT']
    df_fut['ABS_HIGH_LIMIT'] = df_fut['LASTSETTLEPRICE'] + df_fut['HIGHLIMIT']

    # getting absolute risk limits
    df_fut['ABS_LOW_RISK_LIMIT'] = df_fut['LASTSETTLEPRICE'] - 1.3*df_fut['LOWLIMIT']
    df_fut['ABS_HIGH_RISK_LIMIT'] = df_fut['LASTSETTLEPRICE'] + 1.3*df_fut['HIGHLIMIT']

    return df_fut

# for the left border sliders in what_if_analysis
@st.cache(persist=True)
def min_vola_time(d):
    min_vola = 0
    min_time = 0
    for key, value in d.items():
        if min_vola == 0 and (value['type'] != 'future'): min_vola = value['original_volatility']
        if min_time == 0 and (value['type'] != 'future'): min_time = value['original_time(days)']

        if min_vola > value['original_volatility'] and (value['type'] != 'future'):
            min_vola = value['original_volatility']
        if min_time > value['original_time(days)'] and (value['type'] != 'future'):
            min_time = value['original_time(days)']

    return [int(np.floor(min_vola)), int(np.floor(min_time))]

@st.cache(allow_output_mutation=True)
def get_opt_data(underlying):
    ######### options data
    r_opt = requests.get(iss_urls()["query_options_instruments"])
    data_opt = r_opt.json()["securities"]["data"]
    columns_opt = r_opt.json()["securities"]["columns"]
    df_opt = pd.DataFrame(data_opt, columns=columns_opt)

    # getting underlying
    df_opt["UNDERLYING"] = df_opt["SHORTNAME"].apply(get_option_underlying)

    # getting short option names = cutting strikes and types
    df_opt["OPT_SERIES"] = df_opt["SHORTNAME"].apply(get_option_series_name)

    # getting strike = cutting strikes and types
    df_opt["STRIKE"] = df_opt["SHORTNAME"].apply(get_option_strike).apply(float)

    # getting type of option
    df_opt["OPT_TYPE"] = df_opt["SHORTNAME"].apply(get_option_type)


    # getting STEPPRICE param for options from futures
    df_opt = df_opt.merge(df_fut[["SHORTNAME", "STEPPRICE"]],
                                  how="left", left_on="UNDERLYING", right_on="SHORTNAME", suffixes=["", "_y"])

    return df_opt[df_opt['ASSETCODE'] == underlying]

#### getting parameters
dict_opt_type = {'call': 1, 'put': -1, 'future': 0}

opt_state = SessionState.get(rerun=False, dict=OrderedDict())
opt_state.dict['points'] = opt_state.dict.get('points', OrderedDict())
opt_state.dict['params'] = opt_state.dict.get('params', OrderedDict())

#opt_state.dict['underlying_type'] = opt_state.str.get('underlying_type', str)


#### sidebar
st.sidebar.title("Exchange")
underlying = st.sidebar.selectbox('', ['Moscow Exchange'], index=0)


df_fut = get_fut_data()


st.sidebar.markdown("### Underlying")
#### trying to get RTS underlying as a default
lst_undrl = df_fut['ASSETCODE'].unique().tolist()
try:
    default_index = lst_undrl.index('RTS')
except ValueError:
    default_index = 0
underlying = st.sidebar.selectbox('', lst_undrl, index=default_index)

# initialization state variable
if 'underlying_type' not in opt_state.dict:
	opt_state.dict['underlying_type'] = underlying

# if undelying is changed, old points will be deleted
if opt_state.dict['underlying_type'] != underlying:
    opt_state.dict['points'] = OrderedDict()
    opt_state.dict['params'] = OrderedDict()
    opt_state.dict['underlying_type'] = underlying

#st.text(opt_state.dict['underlying_type'])

df_opt = get_opt_data(underlying)


st.sidebar.markdown("### Type of derivative")
if df_opt.shape[0] > 0:
    contr_type = st.sidebar.radio('', ['future', 'call', 'put'])
else:
    contr_type = st.sidebar.radio('', ['future'])


if contr_type != 'future' and df_opt.shape[0] > 0:
    st.sidebar.markdown("### Expiration Date")
    lst_exp_dates = df_opt.sort_values('LASTTRADEDATE')['LASTTRADEDATE'].unique()
    exp_date = st.sidebar.selectbox('', lst_exp_dates)

    st.sidebar.markdown("### Strike")
    #st.sidebar.text(df_opt[df_opt['LASTTRADEDATE'] == exp_date])
    df_opt['STRIKE'] = df_opt['STRIKE'].apply(int)
    strike_range = df_opt[df_opt['LASTTRADEDATE'] == exp_date]['STRIKE'].sort_values().unique()
    strike = st.sidebar.selectbox('', strike_range, index=int(0.5 * len(strike_range)))

    ### getting underlying-MM.YY for prices and limits
    underlying_mm_yy = df_opt[df_opt['LASTTRADEDATE'] == exp_date]['UNDERLYING'].unique()[0]
    shortname = df_opt[(df_opt['LASTTRADEDATE'] == exp_date) &
                        (df_opt['STRIKE'] == strike) &
                        (df_opt['OPT_TYPE'] == contr_type)]['SHORTNAME'].unique()[0]
else:
    st.sidebar.markdown("### Expiration Date")
    lst_exp_dates = df_fut[df_fut['ASSETCODE'] == underlying].sort_values('LASTTRADEDATE')['LASTTRADEDATE'].unique()
    exp_date = st.sidebar.selectbox('', lst_exp_dates)
    ### getting underlying-MM.YY for prices and limits
    shortname = df_fut[(df_fut['ASSETCODE'] == underlying) &
                       (df_fut['LASTTRADEDATE'] == exp_date)]['SHORTNAME'].unique()[0]
    underlying_mm_yy = shortname
    strike = 0
    volatility = 0


time_before_expiration = (datetime.strptime(exp_date, '%Y-%m-%d') - datetime.today()).total_seconds()/(365*24*60*60)


# values for plotting vertical lines
undrl_price = df_fut[df_fut['SHORTNAME'] == underlying_mm_yy]['LASTSETTLEPRICE'].values[0]

price_limit_down = df_fut[df_fut['SHORTNAME'] == underlying_mm_yy]['ABS_LOW_LIMIT'].values[0]
price_limit_up = df_fut[df_fut['SHORTNAME'] == underlying_mm_yy]['ABS_HIGH_LIMIT'].values[0]

risk_limit_down = df_fut[df_fut['SHORTNAME'] == underlying_mm_yy]['ABS_LOW_RISK_LIMIT'].values[0]
risk_limit_up = df_fut[df_fut['SHORTNAME'] == underlying_mm_yy]['ABS_HIGH_RISK_LIMIT'].values[0]


# points for calculating meanings for plotting on a chart
undrl_points = df_fut[df_fut['ASSETCODE'] == underlying][['LASTSETTLEPRICE',
                                                     'ABS_LOW_LIMIT', 'ABS_HIGH_LIMIT',
                                                     'ABS_LOW_RISK_LIMIT', 'ABS_HIGH_RISK_LIMIT']].values

extra_undrl_points = np.array(get_undrl_points(undrl_points.min(), undrl_points.max()))

strike_range_for_points = df_opt['STRIKE'].sort_values().unique()

lst_undrl_points = np.unique(np.append(np.append(undrl_points, extra_undrl_points), strike_range_for_points)).tolist()
lst_undrl_points.sort()


st.sidebar.markdown("### Price")
price = st.sidebar.number_input('', min_value=0.0)



st.sidebar.markdown("### Amount")
amount = st.sidebar.number_input('*negative is short', step=1)


df_current_opt_prices = pd.DataFrame({'total_values': [0] * len(lst_undrl_points)},
                                         index=lst_undrl_points)


#### mainbar
st.title("Portfolio")


side_cols_btn = st.sidebar.beta_columns(2)

if side_cols_btn[1].button("Clear"):
    opt_state.dict['points'] = OrderedDict()
    opt_state.dict['params'] = OrderedDict()

chart_slot = st.empty()


if st.checkbox('Show What If Analysis', False):
    sub_col = st.beta_columns(2)
    vola_increment = sub_col[0].slider('Volatility Increment',
                                       min_value=-min_vola_time(opt_state.dict['params'])[0],
                                       max_value=99, value=0)
    time_increment = sub_col[1].slider('Time Increment',
                                       min_value=-min_vola_time(opt_state.dict['params'])[1],
                                       max_value=365, value=0)

    # applying increment
    if sub_col[0].button("Apply increment"):
        for index, row in opt_state.dict['params'].items():
            if row['type'] == 'call' or row['type'] == 'put':
                # getting points for a plot
                next_key = index
                opt_vector = get_derivative_points(lst_undrl_points, dict_opt_type[row['type']], price,
                                                   amount, strike,
                                                   (row['original_volatility'] + vola_increment) / 100,
                                                   (row['original_time(days)'] + time_increment) / 365)

                opt_state.dict['points'][next_key + '_current'] = opt_vector

                # calculating updated greeks
                option = mibian.BS([undrl_price, strike, 0, row['original_time(days)'] + time_increment],
                                   volatility=row['original_volatility'] + vola_increment)

                if contr_type == 'call':
                    opt_state.dict['params'][next_key]['delta'] = amount * option.callDelta
                    opt_state.dict['params'][next_key]['theta'] = amount * option.callTheta
                elif contr_type == 'put':
                    opt_state.dict['params'][next_key]['delta'] = amount * option.putDelta
                    opt_state.dict['params'][next_key]['theta'] = amount * option.putTheta
                opt_state.dict['params'][next_key]['gamma'] = amount * option.gamma
                opt_state.dict['params'][next_key]['vega'] = amount * option.vega

                opt_state.dict['params'][next_key]['time(days)'] = row['original_time(days)'] + time_increment
                opt_state.dict['params'][next_key]['volatility'] = row['original_volatility'] + vola_increment


if side_cols_btn[0].button("Add to portfolio"):

    maturity_days = 365 * time_before_expiration #+ time_increment
    # calculating volatility
    if contr_type == 'call':
        option_for_volat = mibian.BS([undrl_price, strike, 0, maturity_days], callPrice=price)
        volatility = option_for_volat.impliedVolatility
    elif contr_type == 'put':
        option_for_volat = mibian.BS([undrl_price, strike, 0, maturity_days], putPrice=price)
        volatility = option_for_volat.impliedVolatility
    else:   # branch for future
        volatility = 0


    # getting points for a plot
    next_key = shortname
    opt_vector = get_derivative_points(lst_undrl_points, dict_opt_type[contr_type], price,
                            amount, strike, volatility/100, maturity_days/365)
    exp_opt_vector = get_exp_derivative_points(lst_undrl_points, dict_opt_type[contr_type],
                                               price, amount, strike)

    opt_state.dict['points'][next_key + '_current'] = opt_vector
    opt_state.dict['points'][next_key + '_expiration'] = exp_opt_vector

    # calculating greeks
    option = mibian.BS([undrl_price, strike, 0, maturity_days], volatility=volatility)

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


    opt_state.dict['params'][next_key] = {'name': shortname, 'type': contr_type,
                                          'strike': strike, 'underlying_price': undrl_price,
                                          'amount': amount, 'time(days)': maturity_days,
                                          'volatility': volatility, 'price': price,
                                          'is_in_portfolio': True,
                                          'delta': delta, 'gamma': gamma,
                                          'vega': vega, 'theta': theta,
                                          'original_time(days)': maturity_days,
                                          'original_volatility': volatility,
                                          'underlying': underlying}


df_params = pd.DataFrame(opt_state.dict['params']).T

try:
    selected = [key for key, value in opt_state.dict['params'].items() if value['is_in_portfolio'] == True]
    in_portfolio = st.multiselect('In portfolio', options=[k for k in opt_state.dict['params'].keys()],
        default=selected, key='set_portfolio')

    for key, value in opt_state.dict['params'].items():
        if key in in_portfolio:
            opt_state.dict['params'][key]['is_in_portfolio'] = True
        else:
            opt_state.dict['params'][key]['is_in_portfolio'] = False

    df_params = pd.DataFrame(opt_state.dict['params']).T
    st.dataframe(df_params[df_params['is_in_portfolio'] == True][df_params.columns.difference(['name',
                                                                    'is_in_portfolio',
                                                                    'original_volatility',
                                                                    'original_time(days)'])])
except:
    in_portfolio = st.multiselect('In portfolio', [])


### forming points dataframe
df_position = pd.DataFrame(opt_state.dict['points'], index=lst_undrl_points)

lst_current = [col for col in df_position.columns if 'current' in col]
lst_expiration = [col for col in df_position.columns if 'expiration' in col]

are_in = [v['is_in_portfolio'] for v in opt_state.dict['params'].values()]

df_position['total_opt_points'] = (df_position[lst_current]*are_in).sum(axis=1)
df_position['total_exp_opt_points'] = (df_position[lst_expiration]*are_in).sum(axis=1)

if st.checkbox('Show Data', True):
    st.dataframe(df_position)

### plotting
fig = px.line()
fig.add_scatter(x=df_position.index, y=df_position.total_opt_points,
                    name='current price')
fig.add_scatter(x=df_position.index, y=df_position.total_exp_opt_points,
                name='expiration price')
### scaling plot
fig.update_xaxes(range=[np.min(df_position.index), np.max(df_position.index)])# sets the range of xaxis
fig.update_yaxes(range=[1.1*df_position['total_exp_opt_points'].min(),
                        1.1*df_position['total_exp_opt_points'].max()])
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
fig.add_vline(x=undrl_price, line_width=1, line_dash="dashdot", line_color="black",
                name='underlying price')
### adding horizontal axe
fig.add_hline(y=0, line_width=1, line_color="black")
chart_slot.plotly_chart(fig)


