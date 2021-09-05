import streamlit as st
import pandas as pd
import requests
import numpy as np
import mibian
import plotly.express as px
from datetime import datetime, date


from collections import OrderedDict

import SessionState

from functions import get_undrl_points, get_price_points, get_exp_price_points, iss_urls, \
    get_option_underlying, get_option_series_name, get_option_strike, get_option_type, get_greeks, \
   min_vola_time, get_volatility, get_plot_stuff, get_greek_points, get_opt_smiles, get_maturity_days, \
   get_dict_language, translate_word


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
dict_language = get_dict_language()

current_date = datetime.now()
current_date = current_date.replace(hour=0, minute=0, second=0, microsecond=0)

opt_state = SessionState.get(rerun=False, dict=OrderedDict())
opt_state.dict['points'] = opt_state.dict.get('points', OrderedDict())
opt_state.dict['params'] = opt_state.dict.get('params', OrderedDict())
opt_state.dict['greek_points'] = opt_state.dict.get('greek_points', OrderedDict())
opt_state.dict['vola_smile_points'] = opt_state.dict.get('vola_smile_points', OrderedDict())


#### sidebar
language = st.sidebar.radio('', ['Русский', 'English'], index=0, key='lang')

st.sidebar.title(dict_language[language]["Exchange"])
exchange = st.sidebar.selectbox('', ['Moex'], index=0, format_func=lambda x: dict_language[language][x])


df_fut = get_fut_data(current_date)


st.sidebar.markdown('### {}'.format(dict_language[language]["underlying"]))
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
    opt_state.dict['vola_smile_points'] = OrderedDict()
    opt_state.dict['underlying_type'] = underlying
    opt_state.dict['default_portfolio'] = False


df_opt = get_opt_data(underlying, current_date)


st.sidebar.markdown('### {}'.format(dict_language[language]["type_derivative"]))
if df_opt.shape[0] > 0:
    contr_type = st.sidebar.radio('', ['future', 'call', 'put'], index=1, format_func=lambda x: dict_language[language][x])
else:
    contr_type = st.sidebar.radio('', ['future'], format_func=lambda x: dict_language[language][x])


if contr_type != 'future' and df_opt.shape[0] > 0:
    # filter contr_type & the closest futures & central strike
    df_opt_subset = df_opt[(df_opt['OPT_TYPE'] == contr_type)
                           & (df_opt['LASTTRADEDATE_fut'].min() == df_opt['LASTTRADEDATE_fut'])]
    df_opt_subset = df_opt_subset[df_opt_subset['LASTTRADEDATE'].max() == df_opt_subset['LASTTRADEDATE']]
    df_opt_subset = df_opt_subset[(df_opt_subset['strike_underl_distance'].min() == df_opt_subset['strike_underl_distance'])]


    st.sidebar.markdown('### {}'.format(dict_language[language]["Expiration_date"]))
    tpl_exp_dates = tuple(df_opt.sort_values('LASTTRADEDATE')['LASTTRADEDATE'].unique())
    exp_date = st.sidebar.selectbox('', tpl_exp_dates, index=tpl_exp_dates.index(df_opt_subset['LASTTRADEDATE'].values[0]))

    st.sidebar.markdown('### {}'.format(dict_language[language]["strike"]))
    df_opt['STRIKE'] = df_opt['STRIKE'].apply(float)
    tpl_strike_range = tuple(df_opt[df_opt['LASTTRADEDATE'] == exp_date]['STRIKE'].sort_values().unique())
    strike = st.sidebar.selectbox('', tpl_strike_range, index=tpl_strike_range.index(df_opt_subset['STRIKE'].values[0]))

    ### getting underlying-MM.YY for prices and limits
    underlying_mm_yy = df_opt[df_opt['LASTTRADEDATE'] == exp_date]['UNDERLYING'].unique()[0]
    shortname = df_opt[(df_opt['LASTTRADEDATE'] == exp_date) &
                        (df_opt['STRIKE'] == strike) &
                        (df_opt['OPT_TYPE'] == contr_type)]['SHORTNAME'].unique()[0]
    ## Price
    st.sidebar.markdown('### {}'.format(dict_language[language]["price"]))
    prevsettleprice = df_opt[(df_opt['LASTTRADEDATE'] == exp_date) &
                        (df_opt['STRIKE'] == strike) &
                        (df_opt['OPT_TYPE'] == contr_type)]['PREVSETTLEPRICE'].unique()[0]
    price = st.sidebar.number_input('', min_value=0.0, step=min_step, value=prevsettleprice, format="%.4f")
else:
    st.sidebar.markdown('### {}'.format(dict_language[language]["Expiration_date"]))
    tpl_exp_dates = df_fut[df_fut['ASSETCODE'] == underlying].sort_values('LASTTRADEDATE')['LASTTRADEDATE'].unique()
    exp_date = st.sidebar.selectbox('', tpl_exp_dates)
    ### getting underlying-MM.YY for prices and limits
    shortname = df_fut[(df_fut['ASSETCODE'] == underlying) &
                       (df_fut['LASTTRADEDATE'] == exp_date)]['SHORTNAME'].unique()[0]
    underlying_mm_yy = shortname
    strike = 0
    volatility = 0
    ## Price
    st.sidebar.markdown('### {}'.format(dict_language[language]["price"]))
    prevsettleprice = df_fut[(df_fut['ASSETCODE'] == underlying) &
                       (df_fut['LASTTRADEDATE'] == exp_date)]['PREVSETTLEPRICE'].unique()[0]
    price = st.sidebar.number_input('', min_value=0.0, step=min_step, value=prevsettleprice, format="%.5f")


st.sidebar.markdown('### {}'.format(dict_language[language]["amount"]))
amount = st.sidebar.number_input(dict_language[language]['negative'], step=1, value=42)

######################## inner calculations #########################
###############################################################

maturity_days = get_maturity_days(exp_date)


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
    opt_state.dict['params'][shortname] = {'name': shortname, 'type_derivative': contr_type,
                                          'strike': strike, 'underlying_price': undrl_price,
                                          'amount': amount, 'maturity_days': round(maturity_days, 1),
                                          'volatility': round(volatility, 2), 'price': price,
                                          'fair_price': round(price, decimals),
                                          'is_in_portfolio': True,
                                          'delta': dict_greeks['delta'], 'gamma': dict_greeks['gamma'],
                                          'vega': dict_greeks['vega'], 'theta': dict_greeks['theta'],
                                          'original_maturity_days': maturity_days,
                                          'original_volatility': volatility,
                                          'underlying': underlying,
                                          'settlement_price': prevsettleprice}

    opt_state.dict['default_portfolio'] = False


title_cols = st.beta_columns([3, 1])
title_cols[0].title(dict_language[language]['Portfolio'])
#title_cols[1].title('')
### for comments and feedbacks
link = '[{}](https://derivatives-map.herokuapp.com/feedback/)'.format(dict_language[language]['Feedback'])
title_cols[1].markdown(link, unsafe_allow_html=True)

name_plot = st.selectbox('', options=['PL', 'Greeks', 'Vola_smile'], format_func=lambda x: dict_language[language][x])


side_cols_btn = st.sidebar.beta_columns(2)

if side_cols_btn[1].button(dict_language[language]["Clear"]):
    opt_state.dict['points'] = OrderedDict()
    opt_state.dict['params'] = OrderedDict()
    opt_state.dict['greek_points'] = OrderedDict()
    opt_state.dict['vola_smile_points'] = OrderedDict()
    opt_state.dict['default_portfolio'] = False

chart_slot = st.empty()


if st.checkbox(dict_language[language]['What_if'], False):
    sub_col = st.beta_columns(3)
    vola_increment = sub_col[0].slider(dict_language[language]['Vola_inc'],
                                       min_value=-min_vola_time(opt_state.dict['params'])[0],
                                       max_value=99, value=0)
    time_increment = sub_col[1].slider(dict_language[language]['Time_inc'],
                                       min_value=-min_vola_time(opt_state.dict['params'])[1],
                                       max_value=365, value=0)

    # changing price
    undrl_price_increment = sub_col[2].slider(dict_language[language]['Undrl_price_inc'],
                                        min_value=float(price_limit_down - undrl_price),
                                        max_value=float(price_limit_up - undrl_price),
                                        value=float(0),
                                        step=min_step)
    updated_undrl_price = undrl_price + undrl_price_increment

    # applying increment
    if sub_col[1].button(dict_language[language]["Upd_params"]):
        for index, row in opt_state.dict['params'].items():
            if row['type_derivative'] == 'call' or row['type_derivative'] == 'put':
                # getting points for a plot
                opt_state.dict['points'][index + '_current'] = get_price_points(lst_undrl_points, dict_opt_type[row['type_derivative']],
                                              row['price'], row['amount'], row['strike'],
                                              (row['original_volatility'] + vola_increment) / 100,
                                              (row['original_maturity_days'] + time_increment) / 365)

                opt_state.dict['greek_points'][index] = get_greek_points(lst_undrl_points, row['type_derivative'], \
                                                                         row['amount'], row['strike'], \
                                                                         (row['original_volatility'] + vola_increment) , \
                                                                         (row['original_maturity_days'] + time_increment) , 0.0)


                # calculating updated greeks
                option = mibian.BS([row['underlying_price'] + undrl_price_increment,
                                    row['strike'], 0,
                                    row['original_maturity_days'] + time_increment],
                                   volatility=row['original_volatility'] + vola_increment)
                if row['type_derivative'] == 'call':
                    opt_state.dict['params'][index]['delta'] = row['amount'] * round(option.callDelta, 3)
                    opt_state.dict['params'][index]['theta'] = row['amount'] * round(option.callTheta, 3)
                    opt_state.dict['params'][index]['fair_price'] = round(option.callPrice, decimals)
                elif row['type_derivative'] == 'put':
                    opt_state.dict['params'][index]['delta'] = row['amount'] * round(option.putDelta, 3)
                    opt_state.dict['params'][index]['theta'] = row['amount'] * round(option.putTheta, 3)
                    opt_state.dict['params'][index]['fair_price'] = round(option.putPrice, decimals)
                opt_state.dict['params'][index]['gamma'] = row['amount'] * round(option.gamma, 6)
                opt_state.dict['params'][index]['vega'] = row['amount'] * round(option.vega, 3)

                opt_state.dict['params'][index]['maturity_days'] = round(row['original_maturity_days'] + time_increment, 1)
                opt_state.dict['params'][index]['volatility'] = round(row['original_volatility'] + vola_increment, 3)
                opt_state.dict['params'][index]['underlying_price'] = row['underlying_price'] + undrl_price_increment
else:
    updated_undrl_price = undrl_price


if side_cols_btn[0].button(dict_language[language]["Add"]):

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
    opt_state.dict['params'][shortname] = {'name': shortname, 'type_derivative': contr_type,
                                          'strike': strike, 'underlying_price': undrl_price,
                                          'amount': amount, 'maturity_days': round(maturity_days, 1),
                                          'volatility': round(volatility, 2), 'price': price,
                                          'fair_price': round(price, decimals),
                                          'is_in_portfolio': True,
                                          'delta': dict_greeks['delta'], 'gamma': dict_greeks['gamma'],
                                          'vega': dict_greeks['vega'], 'theta': dict_greeks['theta'],
                                          'original_maturity_days': maturity_days,
                                          'original_volatility': volatility,
                                          'underlying': underlying,
                                          'settlement_price': prevsettleprice}



df_params = pd.DataFrame(opt_state.dict['params']).T

try:
    selected = [key for key, value in opt_state.dict['params'].items() if value['is_in_portfolio'] == True]
    in_portfolio = st.multiselect(dict_language[language]['In_portfolio'], options=[k for k in opt_state.dict['params'].keys()],
        default=selected, key='set_portfolio')

    for key, value in opt_state.dict['params'].items():
        if key in in_portfolio:
            opt_state.dict['params'][key]['is_in_portfolio'] = True
        else:
            opt_state.dict['params'][key]['is_in_portfolio'] = False

    #st.text(opt_state.dict['params'][key].keys())
    df_params = pd.DataFrame(opt_state.dict['params']).T

    df_params_filtered = df_params[df_params['is_in_portfolio'] == True][df_params.columns.difference(['name',
                                                                    'is_in_portfolio',
                                                                    'original_volatility',
                                                                    'original_maturity_days'])].T
    ### translating name of params
    lst_translated_params = []
    for i in df_params_filtered.index:
        lst_translated_params.append(dict_language[language].get(i, '-'))

    df_params_filtered['translated_params'] = lst_translated_params
    st.dataframe(df_params_filtered.set_index(['translated_params']), height=800)

except:
    in_portfolio = st.multiselect('In portfolio', [])


filtered = [v['is_in_portfolio'] for v in opt_state.dict['params'].values()]

if name_plot == "PL":
    ### summarising points dataframe
    df_position = pd.DataFrame(opt_state.dict['points'], index=lst_undrl_points)

    lst_current = [col for col in df_position.columns if 'current' in col]
    lst_expiration = [col for col in df_position.columns if 'expiration' in col]

    df_position['total_opt_points'] = (df_position[lst_current] * filtered).sum(axis=1)
    df_position['total_exp_opt_points'] = (df_position[lst_expiration] * filtered).sum(axis=1)

    # plotting
    fig = px.line()
    fig.add_scatter(x=df_position.index, y=df_position.total_opt_points,
                         name=dict_language[language]['Current_value'])
    fig.add_scatter(x=df_position.index, y=df_position.total_exp_opt_points,
                          name=dict_language[language]['Expiration_value'])
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
    fig.update_xaxes(range=[risk_limit_down, risk_limit_up], title=dict_language[language]['Underlying_value'])# sets the range of xaxis
    fig.update_yaxes(range=[-1.1 * abs(df_position[(df_position.index >= risk_limit_down) & (df_position.index <= risk_limit_up)]['total_exp_opt_points'].min()),
                             1.1 * abs(df_position[(df_position.index >= risk_limit_down) & (df_position.index <= risk_limit_up)]['total_exp_opt_points'].max())],
                     title=dict_language[language]['Option_value'])

    fig.update_layout(height=400)
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
                  , facet_col_spacing=0.08, facet_row_spacing=0.14
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
    fig.update_xaxes(range=[risk_limit_down, risk_limit_up], title=dict_language[language]['Underlying_value'])
    fig.update_yaxes(title=dict_language[language]['Greek_value'])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=500, font=dict(size=12))
    chart_slot.plotly_chart(fig)
elif name_plot == 'Vola_smile':
    if not opt_state.dict['vola_smile_points']:   # calculate volatility values
        df_opt_for_smiles = get_opt_smiles(df_opt).reset_index()
        df_opt_for_smiles.sort_values(by=['STRIKE'], ascending=[True], inplace=True)
        opt_state.dict['vola_smile_points'] = df_opt_for_smiles.to_dict()
    else:    # in this branch don't calculate volatility values
        df_opt_for_smiles = pd.DataFrame(opt_state.dict['vola_smile_points'])

    lst_ordered = df_opt_for_smiles['LASTTRADEDATE'].sort_values(ascending=[True]).unique()

    # plotting
    fig = px.line(df_opt_for_smiles, x='STRIKE', y='volatility', facet_col='LASTTRADEDATE', facet_col_wrap=2\
                  , facet_col_spacing=0.08, facet_row_spacing=0.14, category_orders={'LASTTRADEDATE': lst_ordered})
    fig.update_layout(height=900)
    fig.update_yaxes(matches=None, showticklabels=True, ticklabelposition="inside top", title=dict_language[language]['volatility'])
    fig.update_xaxes(matches=None, showticklabels=True, title=dict_language[language]['strike'])
    ### adding price limits
    fig.add_vline(x=price_limit_down, line_width=1, line_dash="dash", line_color="red",
                  name='price limit down')
    fig.add_vline(x=price_limit_up, line_width=1, line_dash="dash", line_color="red",
                  name='price limit up')
    ### adding underlying price
    fig.add_vline(x=updated_undrl_price, line_width=1, line_dash="dashdot", line_color="black",
                  name='underlying price')

    fig.update_layout(font=dict(size=12))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    chart_slot.plotly_chart(fig)

else:
    chart_slot.error("Something has gone terribly wrong.")

