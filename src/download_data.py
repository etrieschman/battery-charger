import sys
from data_utils import CAISO, download_caiso_lmp

def main():
    if len(sys.argv) != 4:
        raise Exception("usage: python project1.py <filename prefix> n_iterations")
    
    read_caiso_rt_lmp = sys.argv[1]
    read_caiso_da_lmp = sys.argv[2]
    read_weather = sys.argv[3]
    
    print(f'Requested download (CAISO RT, CAISO DA, Weather):', read_caiso_rt_lmp, read_caiso_da_lmp, read_weather)

    if read_caiso_rt_lmp:
        # Download realtime 15-minute market
        # nodes = CAISO.trading_hub_locations # possible trading nodes
        # nodes = ['MOSSLDB_2_B1', 'MOSSLD_1_N055', 'MOSSLAND_5_B1'] # moss landing possible nodes
        # nodes = ['SANDLOT_2_N022', 'SANDLOT_2_N024', 'SUNSPTA_7_N002'] # kern county possible nodes
        nodes = ['TH_NP15_GEN-APND', 'TH_SP15_GEN-APND', 'MOSSLDB_2_B1', 'TEMP'] # final selected nodes
        market = 'REAL_TIME_15_MIN'
        years = [2020, 2021, 2022]
        sleep=5

        for node in nodes:
            download_caiso_lmp(years=years, node=node, market=market, sleep=sleep)

if __name__ == '__main__':
    main()