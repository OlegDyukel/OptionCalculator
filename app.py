import streamlit as st
import pandas as pd
import requests
import numpy as np
import mibian
import plotly.express as px
from datetime import datetime, date

from bokeh.models.widgets import Panel, Tabs
from bokeh.plotting import figure


from collections import OrderedDict

import SessionState

from functions import get_undrl_points, get_price_points, get_exp_price_points, iss_urls, \
    get_option_underlying, get_option_series_name, get_option_strike, get_option_type, get_greeks, \
   min_vola_time, get_volatility, get_plot_stuff, get_greek_points


@st.cache
def get_fut_data(date):
    ######### futures data
    r_fut = requests.get(iss_urls()["query_futures_instruments"])
    data_fut = r_fut.json()["securities"]["data"]
    columns_fut = r_fut.json()["securities"]["columns"]
    df_fut = pd.DataFrame(data_fut, columns=columns_fut)

    # filtering old futures
    df_fut = df_fut[pd.to_datetime(df_fut['LASTTRADEDATE']) > date]

    # getting absolute price limits
    df_fut['ABS_LOW_LIMIT'] = df_fut['LASTSETTLEPRICE'] - df_fut['LOWLIMIT']
    df_fut['ABS_HIGH_LIMIT'] = df_fut['LASTSETTLEPRICE'] + df_fut['HIGHLIMIT']

    # getting absolute risk limits
    df_fut['ABS_LOW_RISK_LIMIT'] = df_fut['LASTSETTLEPRICE'] - 1.3*df_fut['LOWLIMIT']
    df_fut['ABS_HIGH_RISK_LIMIT'] = df_fut['LASTSETTLEPRICE'] + 1.3*df_fut['HIGHLIMIT']

    return df_fut


@st.cache(allow_output_mutation=True)
def get_opt_data(underlying, date):
    ######### options data
    r_opt = requests.get(iss_urls()["query_options_instruments"])
    data_opt = r_opt.json()["securities"]["data"]
    columns_opt = r_opt.json()["securities"]["columns"]
    df_opt = pd.DataFrame(data_opt, columns=columns_opt)

    df_opt = df_opt[df_opt['ASSETCODE'] == underlying]

    # getting underlying
    df_opt["UNDERLYING"] = df_opt["SHORTNAME"].apply(get_option_underlying)

    # getting short option names = cutting strikes and types
    df_opt["OPT_SERIES"] = df_opt["SHORTNAME"].apply(get_option_series_name)

    # getting strike = cutting strikes and types
    df_opt["STRIKE"] = df_opt["SHORTNAME"].apply(get_option_strike).apply(float)

    # getting type of option
    df_opt["OPT_TYPE"] = df_opt["SHORTNAME"].apply(get_option_type)

    # filtering old options
    df_opt = df_opt[pd.to_datetime(df_opt['LASTTRADEDATE']) > date]

    # getting STEPPRICE param for options from futures
    df_opt = df_opt.merge(df_fut[["SHORTNAME", "STEPPRICE", "PREVSETTLEPRICE", "LASTTRADEDATE"]],
                                  how="left", left_on="UNDERLYING", right_on="SHORTNAME", suffixes=["", "_fut"])
    df_opt['strike_underl_distance'] = np.abs(df_opt['STRIKE'] - df_opt['PREVSETTLEPRICE_fut'])
    return df_opt


#### getting parameters
dict_opt_type = {'call': 1, 'put': -1, 'future': 0}
current_date = datetime.now()
current_date = current_date.replace(hour=0, minute=0, second=0, microsecond=0)

opt_state = SessionState.get(rerun=False, dict=OrderedDict())
opt_state.dict['points'] = opt_state.dict.get('points', OrderedDict())
opt_state.dict['params'] = opt_state.dict.get('params', OrderedDict())
opt_state.dict['greek_points'] = opt_state.dict.get('greek_points', OrderedDict())


#### sidebar
st.sidebar.title("Exchange")
exchange = st.sidebar.selectbox('', ['Moscow Exchange'], index=0)


df_fut = get_fut_data(current_date)


st.sidebar.markdown("### Underlying")
#### trying to get RTS underlying as a default
lst_undrl = df_fut['ASSETCODE'].unique().tolist()
try:
    default_index = lst_undrl.index('RTS')
except ValueError:
    default_index = 0
underlying = st.sidebar.selectbox('', lst_undrl, index=default_index)

min_step = df_fut[df_fut['ASSETCODE'] == underlying]['MINSTEP'].unique()[0]
decimals = df_fut[df_fut['ASSETCODE'] == underlying]['DECIMALS'].unique()[0]

# initialization state variable
if 'underlying_type' not in opt_state.dict:
	opt_state.dict['underlying_type'] = underlying


# if underlying is changed, old points will be deleted
if opt_state.dict['underlying_type'] != underlying:
    opt_state.dict['points'] = OrderedDict()
    opt_state.dict['params'] = OrderedDict()
    opt_state.dict['greek_points'] = OrderedDict()
    opt_state.dict['underlying_type'] = underlying
    opt_state.dict['default_portfolio'] = False


df_opt = get_opt_data(underlying, current_date)


st.sidebar.markdown("### Type of derivative")
if df_opt.shape[0] > 0:
    contr_type = st.sidebar.radio('', ['future', 'call', 'put'], index=1)
else:
    contr_type = st.sidebar.radio('', ['future'])


if contr_type != 'future' and df_opt.shape[0] > 0:
    # filter contr_type & the closest futures & central strike
    df_opt_subset = df_opt[(df_opt['OPT_TYPE'] == contr_type)
                           & (df_opt['LASTTRADEDATE_fut'].min() == df_opt['LASTTRADEDATE_fut'])]
    df_opt_subset = df_opt_subset[df_opt_subset['LASTTRADEDATE'].max() == df_opt_subset['LASTTRADEDATE']]
    df_opt_subset = df_opt_subset[(df_opt_subset['strike_underl_distance'].min() == df_opt_subset['strike_underl_distance'])]


    st.sidebar.markdown("### Expiration Date")
    tpl_exp_dates = tuple(df_opt.sort_values('LASTTRADEDATE')['LASTTRADEDATE'].unique())
    exp_date = st.sidebar.selectbox('', tpl_exp_dates, index=tpl_exp_dates.index(df_opt_subset['LASTTRADEDATE'].values[0]))

    st.sidebar.markdown("### Strike")
    df_opt['STRIKE'] = df_opt['STRIKE'].apply(float)
    tpl_strike_range = tuple(df_opt[df_opt['LASTTRADEDATE'] == exp_date]['STRIKE'].sort_values().unique())
    strike = st.sidebar.selectbox('', tpl_strike_range, index=tpl_strike_range.index(df_opt_subset['STRIKE'].values[0]))

    ### getting underlying-MM.YY for prices and limits
    underlying_mm_yy = df_opt[df_opt['LASTTRADEDATE'] == exp_date]['UNDERLYING'].unique()[0]
    shortname = df_opt[(df_opt['LASTTRADEDATE'] == exp_date) &
                        (df_opt['STRIKE'] == strike) &
                        (df_opt['OPT_TYPE'] == contr_type)]['SHORTNAME'].unique()[0]
    ## Price
    st.sidebar.markdown("### Price")
    prevsettleprice = df_opt[(df_opt['LASTTRADEDATE'] == exp_date) &
                        (df_opt['STRIKE'] == strike) &
                        (df_opt['OPT_TYPE'] == contr_type)]['PREVSETTLEPRICE'].unique()[0]
    price = st.sidebar.number_input('', min_value=0.0, step=min_step, value=prevsettleprice)
else:
    st.sidebar.markdown("### Expiration Date")
    tpl_exp_dates = df_fut[df_fut['ASSETCODE'] == underlying].sort_values('LASTTRADEDATE')['LASTTRADEDATE'].unique()
    exp_date = st.sidebar.selectbox('', tpl_exp_dates)
    ### getting underlying-MM.YY for prices and limits
    shortname = df_fut[(df_fut['ASSETCODE'] == underlying) &
                       (df_fut['LASTTRADEDATE'] == exp_date)]['SHORTNAME'].unique()[0]
    underlying_mm_yy = shortname
    strike = 0
    volatility = 0
    ## Price
    st.sidebar.markdown("### Price")
    prevsettleprice = df_fut[(df_fut['ASSETCODE'] == underlying) &
                       (df_fut['LASTTRADEDATE'] == exp_date)]['PREVSETTLEPRICE'].unique()[0]
    price = st.sidebar.number_input('', min_value=0.0, step=min_step, value=prevsettleprice)


st.sidebar.markdown("### Amount")
amount = st.sidebar.number_input('*negative is short', step=1, value=42)

######################## inner calculations #########################
###############################################################

maturity_days = 365 * (datetime.strptime(exp_date, '%Y-%m-%d') - datetime.today()).total_seconds()/(365*24*60*60)


# values for plotting vertical lines
undrl_price = df_fut[df_fut['SHORTNAME'] == underlying_mm_yy]['LASTSETTLEPRICE'].values[0]

price_limit_down = df_fut[df_fut['SHORTNAME'] == underlying_mm_yy]['ABS_LOW_LIMIT'].max()
price_limit_up = df_fut[df_fut['SHORTNAME'] == underlying_mm_yy]['ABS_HIGH_LIMIT'].min()

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


df_current_opt_prices = pd.DataFrame({'total_values': [0] * len(lst_undrl_points)},
                                         index=lst_undrl_points)

#################################### mainbar ############################
###########################################################################
# apply default
if 'default_portfolio' not in opt_state.dict and len(opt_state.dict['params']) == 0:

    volatility = get_volatility(contr_type, undrl_price, strike, maturity_days, price)
    dict_greeks = get_greeks(contr_type, undrl_price, strike, maturity_days, volatility, amount)

    # getting points for a plot
    opt_state.dict['points'][shortname + '_current'] = get_price_points(lst_undrl_points,
                                                                        dict_opt_type[contr_type], price,
                                                                        amount, strike, volatility / 100, maturity_days / 365)
    opt_state.dict['points'][shortname + '_expiration'] = get_exp_price_points(lst_undrl_points,
                                                                               dict_opt_type[contr_type],
                                                                               price, amount, strike)
    opt_state.dict['greek_points'][shortname] = get_greek_points(lst_undrl_points, contr_type, amount, strike, \
                                                                 volatility, maturity_days, 0.0)

    # getting params
    opt_state.dict['params'][shortname] = {'name': shortname, 'type': contr_type,
                                          'strike': strike, 'underlying_price': undrl_price,
                                          'amount': amount, 'maturity(days)': round(maturity_days, 1),
                                          'volatility': round(volatility, 2), 'price': price,
                                          'fair price': round(price, decimals),
                                          'is_in_portfolio': True,
                                          'delta': dict_greeks['delta'], 'gamma': dict_greeks['gamma'],
                                          'vega': dict_greeks['vega'], 'theta': dict_greeks['theta'],
                                          'original_maturity(days)': maturity_days,
                                          'original_volatility': volatility,
                                          'underlying': underlying,
                                          'settlement_price': prevsettleprice}

    opt_state.dict['default_portfolio'] = False


st.title("Portfolio")

name_plot = st.selectbox('', options=['P&L', 'Greeks'])


side_cols_btn = st.sidebar.beta_columns(2)

if side_cols_btn[1].button("Clear"):
    opt_state.dict['points'] = OrderedDict()
    opt_state.dict['params'] = OrderedDict()
    opt_state.dict['greek_points'] = OrderedDict()
    opt_state.dict['default_portfolio'] = False

chart_slot = st.empty()


if st.checkbox('Show What If Analysis', False):
    sub_col = st.beta_columns(3)
    vola_increment = sub_col[0].slider('Volatility Increment',
                                       min_value=-min_vola_time(opt_state.dict['params'])[0],
                                       max_value=99, value=0)
    time_increment = sub_col[1].slider('Time Increment',
                                       min_value=-min_vola_time(opt_state.dict['params'])[1],
                                       max_value=365, value=0)

    # changing price
    undrl_price_increment = sub_col[2].slider('Underlying Price Increment',
                                        min_value=float(price_limit_down - undrl_price),
                                        max_value=float(price_limit_up - undrl_price),
                                        value=float(0),
                                        step=min_step)
    updated_undrl_price = undrl_price + undrl_price_increment

    # applying increment
    if sub_col[1].button("Update parameters"):
        for index, row in opt_state.dict['params'].items():
            if row['type'] == 'call' or row['type'] == 'put':
                # getting points for a plot
                opt_state.dict['points'][index + '_current'] = get_price_points(lst_undrl_points, dict_opt_type[row['type']],
                                              row['price'], row['amount'], row['strike'],
                                              (row['original_volatility'] + vola_increment) / 100,
                                              (row['original_maturity(days)'] + time_increment) / 365)

                opt_state.dict['greek_points'][index] = get_greek_points(lst_undrl_points, row['type'], \
                                                                         row['amount'], row['strike'], \
                                                                         (row['original_volatility'] + vola_increment) , \
                                                                         (row['original_maturity(days)'] + time_increment) , 0.0)


                # calculating updated greeks
                option = mibian.BS([row['underlying_price'] + undrl_price_increment,
                                    row['strike'], 0,
                                    row['original_maturity(days)'] + time_increment],
                                   volatility=row['original_volatility'] + vola_increment)
                if row['type'] == 'call':
                    opt_state.dict['params'][index]['delta'] = row['amount'] * round(option.callDelta, 3)
                    opt_state.dict['params'][index]['theta'] = row['amount'] * round(option.callTheta, 3)
                    opt_state.dict['params'][index]['fair price'] = round(option.callPrice, decimals)
                elif row['type'] == 'put':
                    opt_state.dict['params'][index]['delta'] = row['amount'] * round(option.putDelta, 3)
                    opt_state.dict['params'][index]['theta'] = row['amount'] * round(option.putTheta, 3)
                    opt_state.dict['params'][index]['fair price'] = round(option.putPrice, decimals)
                opt_state.dict['params'][index]['gamma'] = row['amount'] * round(option.gamma, 6)
                opt_state.dict['params'][index]['vega'] = row['amount'] * round(option.vega, 3)

                opt_state.dict['params'][index]['maturity(days)'] = round(row['original_maturity(days)'] + time_increment, 1)
                opt_state.dict['params'][index]['volatility'] = round(row['original_volatility'] + vola_increment, 3)
                opt_state.dict['params'][index]['underlying_price'] = row['underlying_price'] + undrl_price_increment
else:
    updated_undrl_price = undrl_price


if side_cols_btn[0].button("Add to portfolio"):

    # calculating volatility
    volatility = get_volatility(contr_type, undrl_price, strike, maturity_days, price)
    dict_greeks = get_greeks(contr_type, undrl_price, strike, maturity_days, volatility, amount)

    # getting points for a plot
    opt_state.dict['points'][shortname + '_current'] = get_price_points(lst_undrl_points,
                                                                        dict_opt_type[contr_type], price,
                                                                        amount, strike, volatility / 100,
                                                                        maturity_days / 365)
    opt_state.dict['points'][shortname + '_expiration'] = get_exp_price_points(lst_undrl_points,
                                                                               dict_opt_type[contr_type],
                                                                               price, amount, strike)
    opt_state.dict['greek_points'][shortname] = get_greek_points(lst_undrl_points, contr_type, amount, strike, \
                                                                 volatility, maturity_days, 0.0)

    # getting params
    opt_state.dict['params'][shortname] = {'name': shortname, 'type': contr_type,
                                          'strike': strike, 'underlying_price': undrl_price,
                                          'amount': amount, 'maturity(days)': round(maturity_days, 1),
                                          'volatility': round(volatility, 2), 'price': price,
                                          'fair price': round(price, decimals),
                                          'is_in_portfolio': True,
                                          'delta': dict_greeks['delta'], 'gamma': dict_greeks['gamma'],
                                          'vega': dict_greeks['vega'], 'theta': dict_greeks['theta'],
                                          'original_maturity(days)': maturity_days,
                                          'original_volatility': volatility,
                                          'underlying': underlying,
                                          'settlement_price': prevsettleprice}


# st.dataframe(pd.DataFrame(opt_state.dict['greek_points'][shortname]))

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

    df_params = pd.DataFrame(opt_state.dict['params']).T #.round({'delta': 3, 'theta': 3})
    st.dataframe(df_params[df_params['is_in_portfolio'] == True][df_params.columns.difference(['name',
                                                                    'is_in_portfolio',
                                                                    'original_volatility',
                                                                    'original_maturity(days)'])].T)
except:
    in_portfolio = st.multiselect('In portfolio', [])


filtered = [v['is_in_portfolio'] for v in opt_state.dict['params'].values()]

if name_plot == "P&L":
    ### summarising points dataframe
    df_position = pd.DataFrame(opt_state.dict['points'], index=lst_undrl_points)

    lst_current = [col for col in df_position.columns if 'current' in col]
    lst_expiration = [col for col in df_position.columns if 'expiration' in col]

    df_position['total_opt_points'] = (df_position[lst_current] * filtered).sum(axis=1)
    df_position['total_exp_opt_points'] = (df_position[lst_expiration] * filtered).sum(axis=1)

    # plotting
    fig = px.line()
    fig.add_scatter(x=df_position.index, y=df_position.total_opt_points,
                        name='current price')
    fig.add_scatter(x=df_position.index, y=df_position.total_exp_opt_points,
                    name='expiration price')
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
    ### scaling plot
    fig.update_xaxes(range=[risk_limit_down, risk_limit_up])# sets the range of xaxis
    fig.update_yaxes(range=[-1.1 * abs(df_position[(df_position.index >= risk_limit_down) & (df_position.index <= risk_limit_up)]['total_exp_opt_points'].min()),
                             1.1 * abs(df_position[(df_position.index >= risk_limit_down) & (df_position.index <= risk_limit_up)]['total_exp_opt_points'].max())])
    chart_slot.plotly_chart(fig)
elif name_plot == "Greeks":
    # preparing data for plot
    frames = []
    for key, value in opt_state.dict['greek_points'].items():
        frames.append(pd.DataFrame(value).stack())

    try:
        df_plot = (pd.concat(frames, axis=1) * filtered).sum(axis=1).reset_index()
        df_plot.columns = ['undrl_points', 'type_greek', 'sum_greek']
    except ValueError:
        tpl_type_greek = ('delta', 'gamma', 'vega', 'theta')
        df_plot = pd.DataFrame({'undrl_points': len(tpl_type_greek)*lst_undrl_points,
                                'type_greek': len(lst_undrl_points)*tpl_type_greek,
                                'sum_greek': len(lst_undrl_points)*len(tpl_type_greek)*[0]})


    # plotting
    fig = px.line(df_plot, x='undrl_points', y='sum_greek', facet_col='type_greek', facet_col_wrap=2\
                  , facet_col_spacing=0.05
                  , labels=dict(undrl_points="Underlying (points)", sum_greek="Greek value", type_greek="Type"))
    fig.update_yaxes(matches=None, showticklabels=True, ticklabelposition="inside top")
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
    ### scaling plot
    fig.update_xaxes(range=[risk_limit_down, risk_limit_up])
    chart_slot.plotly_chart(fig)
else:
    chart_slot.error("Something has gone terribly wrong.")
