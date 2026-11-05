import os, time
import pandas as pd

info = pd.read_csv('covasim_run_info.csv')
print(info.columns.tolist())
print(info[['seed', 'LOAD_PER_CASE']])

flows = pd.read_csv('covasim_ww_flows.csv')
print('ALL site flow (L/day):', flows[flows['site'] == 'ALL']['flow'].iloc[0])

for name in sorted(f for f in os.listdir('.') if f.startswith('covasim_')):
    print(name, time.ctime(os.path.getmtime(name)))