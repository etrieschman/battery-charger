
import numpy as np
import pandas as pd
import isodata
import gridstatus
import urllib
from tqdm import tqdm
import os

# define file locations
PATH_HOME = os.path.dirname(os.getcwd())
PATH_DATA = PATH_HOME + '/data/'

# define caiso engine
CAISO = gridstatus.CAISO()
# define nodes
trading_nodes = CAISO.trading_hub_locations[0:2]
moss_node = 'MOSSLDB_2_B1'
kern_node = 'SANDLOT_2_N022' 
NODES = trading_nodes + [moss_node, kern_node]

def make_scrape_profile():
    # set up a profile to access website
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def download_caiso_lmp(years:list, node:str, market:str,  sleep=5):
    make_scrape_profile()
    # make folder to save data
    path_node = f'{PATH_DATA}caiso_{node.lower()}/'
    if not os.path.isdir(path_node):
                os.makedirs(path_node)
    
    # read and save data in chunks
    for y in years:
        for m in tqdm(range(1,12+1), desc=str(y)):
            # define start/ end dates
            dt_start = f'{m}/1/{y}'
            if m == 12:
                dt_end = f'{1}/1/{y+1}'
            else:
                dt_end = f'{m+1}/1/{y}'

            # pull data
            caiso_month = CAISO.get_lmp(date=dt_start, end=dt_end, 
                                                   market=market, locations=[node], 
                                                   sleep=sleep, verbose=False)

            # save chunk            
            caiso_month.to_csv(f'{path_node}{market.lower()}_y{y}m{m}.csv', index=False)

def readin_caiso_lmp(market:str, nodes=None):
    data = pd.DataFrame([])
    # get nodes; if none provided, pull all data
    if nodes is None:
        nodes = [d for d in os.listdir(PATH_DATA) if d.startswith('caiso')]
    else:
         nodes = ['caiso_' + n.lower() for n in nodes]
    print(nodes)
    for n in tqdm(nodes):
        files = [f for f in os.listdir(PATH_DATA + n) if f.startswith(market.lower())]
        for f in files:
            d = pd.read_csv(PATH_DATA + n + f'/{f}')
            data = pd.concat([d, data])

    data.columns = [c.lower() for c in data.columns]
    data = data.sort_values(['location', 'time']).reset_index(drop=True)
    data.loc[:,'datetime'] = pd.to_datetime(data.time.str.slice(0, 19), format='%Y-%m-%d %H:%M:%S')
    data = data.drop(columns='time')
    data.loc[:, 'time'] = data.datetime.dt.time
    data.loc[:, 'day'] = data.datetime.dt.day
    data.loc[:, 'month'] = data.datetime.dt.month
    data.loc[:, 'year'] = data.datetime.dt.year
    data.loc[:, 'week'] = data.datetime.dt.isocalendar().week

    return data
