import sys
from utils_data import CAISO, download_caiso_lmp

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python download_data.py <download RT?> <download DA?>")
    
    read_caiso_rt_lmp = sys.argv[1].lower() == 'true'
    read_caiso_da_lmp = sys.argv[2].lower() == 'true'
    
    print(f'Requested download (CAISO RT, CAISO DA):', read_caiso_rt_lmp, read_caiso_da_lmp)

    # nodes = CAISO.trading_hub_locations # possible trading nodes
    # nodes = ['MOSSLDB_2_B1', 'MOSSLD_1_N055', 'MOSSLAND_5_B1'] # moss landing possible nodes
    # nodes = ['SANDLOT_2_N022', 'SANDLOT_2_N024', 'SUNSPTA_7_N002'] # kern county possible nodes
    nodes = ['TH_NP15_GEN-APND', 'TH_SP15_GEN-APND', 'MOSSLDB_2_B1', 'SANDLOT_2_N022'] # final selected nodes
    years = [2020, 2021, 2022]
    sleep=5

    if read_caiso_rt_lmp:
        print('\tReading realtime 15-min data...')
        # Download realtime 15-minute market
        market = 'REAL_TIME_15_MIN'
        for node in nodes:
            download_caiso_lmp(years=years, node=node, market=market, sleep=sleep)

    if read_caiso_da_lmp:
        print('\tReading day ahead hourly data...')
        # Download day ahead hourly market
        market = 'DAY_AHEAD_HOURLY'
        for node in nodes:
            download_caiso_lmp(years=years, node=node, market=market, sleep=sleep)

if __name__ == '__main__':
    main()