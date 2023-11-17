#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import seaborn as sb
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

from scipy import stats
import matplotlib.pyplot as plt
import os.path as path
plt.style.use('seaborn-whitegrid')
from itertools import cycle


# In[70]:

class CacheStatsPlugin:
 def input(self, inputfile):
   self.infile = inputfile
   self.df_all = pd.read_csv(inputfile, header=None)
 def run(self):
     pass
 def output(self, outputfile):
  self.df_all.columns = ['traces','algorithm','cache_size','hit_rate']

  self.df_all = self.df_all[self.df_all.algorithm != 'alecar6']
  print(len(self.df_all))
  num_traces = len(self.df_all)

  labels = list(r[-1] for r in self.df_all['traces'].str.split('/'))

  ls = []
  trunk_traces = []
  for i in labels:
    trunk_traces.append(path.splitext(i)[0])
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('vps'):
        ls.append('CloudVPS')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('webserver'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('moodle'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blkparse':
        ls.append('FIU')
    if path.splitext(i)[-1]== '.csv':
        ls.append('MSR')
    if path.splitext(i)[-1]== '.txt':
        ls.append('NEXUS')
  self.df_all['dataset'] = ls
  self.df_all['traces'] = trunk_traces
  #print(self.df_all)
  self.df_all = self.df_all[self.df_all.dataset == 'CloudCache']
  self.df_all['cache_size/dataset'] = list(zip(self.df_all['dataset'], self.df_all['cache_size']))

  sorted_df = self.df_all.sort_values(['traces', 'cache_size', 'hit_rate'], ascending=[True, True, False])
  sorted_df['rank'] = sorted_df.groupby(['traces', 'cache_size'])['hit_rate'].rank(ascending=False, method='first')
  #print(sorted_df)
  #sorted_df.to_csv('sorted_df.csv')
  self.df_all_grouped = sorted_df.groupby(['traces', 'cache_size'])

  hatches = cycle(['////', '//', '\\' , 'xxx', ''])
  flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
  sns.set_palette(flatui)


  # In[72]:


  self.df_all_avg = self.df_all.groupby(['cache_size/dataset', 'algorithm'])['hit_rate'].mean().reset_index()
  sorted_df_avg = self.df_all_avg.sort_values(['cache_size/dataset', 'hit_rate'], ascending=[True, True])

  fig, ax = plt.subplots(figsize=(20, 8))
  ax= sns.barplot(x='cache_size/dataset', y='hit_rate', data=sorted_df_avg, hue='algorithm', edgecolor='k')

  num_locations = 6
  for i, patch in enumerate(ax.patches):
    if i % num_locations == 0:
        hatch = next(hatches)
    patch.set_hatch(hatch)

  label = ["0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"]
  ax.set_xticklabels(label)
  ax.set_title('CloudCache')
  ax.set_xlabel('cache size(% of workload footprint)')
  ax.set_ylabel('Average Hit Rate')

  ax.legend(loc='best')
  #fig.savefig('figure_cloudcache.eps', format='eps', bbox_inches = 'tight')
  plt.show()


  # In[73]:


  self.df_all = pd.read_csv(self.infile, header=None)
  self.df_all.columns = ['traces','algorithm','cache_size','hit_rate']

  self.df_all = self.df_all[self.df_all.algorithm != 'alecar6']
  print(len(self.df_all))
  num_traces = len(self.df_all)

  labels = list(r[-1] for r in self.df_all['traces'].str.split('/'))

  ls = []
  trunk_traces = []
  for i in labels:
    trunk_traces.append(path.splitext(i)[0])
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('vps'):
        ls.append('CloudVPS')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('webserver'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('moodle'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blkparse':
        ls.append('FIU')
    if path.splitext(i)[-1]== '.csv':
        ls.append('MSR')
    if path.splitext(i)[-1]== '.txt':
        ls.append('NEXUS')
  self.df_all['dataset'] = ls
  self.df_all['traces'] = trunk_traces
  #print(self.df_all)
  self.df_all = self.df_all[self.df_all.dataset == 'MSR']
  self.df_all['cache_size/dataset'] = list(zip(self.df_all['dataset'], self.df_all['cache_size']))

  sorted_df = self.df_all.sort_values(['traces', 'cache_size', 'hit_rate'], ascending=[True, True, False])
  sorted_df['rank'] = sorted_df.groupby(['traces', 'cache_size'])['hit_rate'].rank(ascending=False, method='first')
  #print(sorted_df)
  #sorted_df.to_csv('sorted_df.csv')
  self.df_all_grouped = sorted_df.groupby(['traces', 'cache_size'])


  # In[74]:


  self.df_all_avg = self.df_all.groupby(['cache_size/dataset', 'algorithm'])['hit_rate'].mean().reset_index()
  #print(self.df_all_avg)
  sorted_df_avg = self.df_all_avg.sort_values(['cache_size/dataset', 'hit_rate'], ascending=[True, True])
  fig, ax = plt.subplots(figsize=(20, 8))
  ax= sns.barplot(x='cache_size/dataset', y='hit_rate', data=sorted_df_avg, hue='algorithm', edgecolor='k')

  num_locations = 6
  for i, patch in enumerate(ax.patches):
    if i % num_locations == 0:
        hatch = next(hatches)
    patch.set_hatch(hatch)
    
  label = ["0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"]
  ax.set_xticklabels(label)
  ax.set_title('MSR')
  ax.set_xlabel('cache size(% of workload footprint)')
  ax.set_ylabel('Average Hit Rate')

  ax.legend(loc='best')
  #fig.savefig('figure_cloudvps.eps', format='eps', bbox_inches = 'tight')
  plt.show()


  # In[77]:


  self.df_all = pd.read_csv(self.infile, header=None)
  self.df_all.columns = ['traces','algorithm','cache_size','hit_rate']

  #self.df_all = self.df_all[self.df_all.algorithm != 'alecar6']
  print(len(self.df_all))
  num_traces = len(self.df_all)

  labels = list(r[-1] for r in self.df_all['traces'].str.split('/'))

  ls = []
  trunk_traces = []
  for i in labels:
    trunk_traces.append(path.splitext(i)[0])
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('vps'):
        ls.append('CloudVPS')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('webserver'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('moodle'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blkparse':
        ls.append('FIU')
    if path.splitext(i)[-1]== '.csv':
        ls.append('MSR')
    if path.splitext(i)[-1]== '.txt':
        ls.append('NEXUS')
  self.df_all['dataset'] = ls
  self.df_all['traces'] = trunk_traces
  #print(self.df_all)
  self.df_all = self.df_all[self.df_all.dataset == 'FIU']
  self.df_all['cache_size/dataset'] = list(zip(self.df_all['dataset'], self.df_all['cache_size']))

  sorted_df = self.df_all.sort_values(['traces', 'cache_size', 'hit_rate'], ascending=[True, True, False])
  sorted_df['rank'] = sorted_df.groupby(['traces', 'cache_size'])['hit_rate'].rank(ascending=False, method='first')
  #print(sorted_df)
  #sorted_df.to_csv('sorted_df.csv')
  self.df_all_grouped = sorted_df.groupby(['traces', 'cache_size'])


  # In[78]:


  self.df_all_avg = self.df_all.groupby(['cache_size/dataset', 'algorithm'])['hit_rate'].mean().reset_index()
  #print(self.df_all_avg)
  sorted_df_avg = self.df_all_avg.sort_values(['cache_size/dataset', 'hit_rate'], ascending=[True, True])

  # Define some hatches and color
  hatches = cycle(['////', '//', '\\' , 'xxx', ''])
  flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
  #sns.palplot(sns.color_palette(flatui))
  sns.set_palette(flatui)

  fig, ax = plt.subplots(figsize=(20, 8))
  ax= sns.barplot(x='cache_size/dataset', y='hit_rate', data=sorted_df_avg, hue='algorithm', edgecolor='k')

  num_locations = 6
  for i, patch in enumerate(ax.patches):
    if i % num_locations == 0:
        hatch = next(hatches)
    patch.set_hatch(hatch)
    
  label = ["0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"]
  ax.set_xticklabels(label)
  ax.set_title('FIU')
  ax.set_xlabel('cache size(% of workload footprint)')
  ax.set_ylabel('Average Hit Rate')
  ax.legend(loc='best')

  #fig.savefig('figure_cloudvps.eps', format='eps', bbox_inches = 'tight')
  plt.show()


  # In[9]:


  self.df_all = pd.read_csv(self.infile, header=None)
  self.df_all.columns = ['traces','algorithm','cache_size','hit_rate']

  #self.df_all = self.df_all[self.df_all.algorithm != 'alecar6']
  print(len(self.df_all))
  num_traces = len(self.df_all)

  labels = list(r[-1] for r in self.df_all['traces'].str.split('/'))

  ls = []
  trunk_traces = []
  for i in labels:
    trunk_traces.append(path.splitext(i)[0])
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('vps'):
        ls.append('CloudVPS')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('webserver'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('moodle'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blkparse':
        ls.append('FIU')
    if path.splitext(i)[-1]== '.csv':
        ls.append('MSR')
    if path.splitext(i)[-1]== '.txt':
        ls.append('NEXUS')
  self.df_all['dataset'] = ls
  self.df_all['traces'] = trunk_traces
  #print(self.df_all)
  self.df_all = self.df_all[self.df_all.dataset == 'CloudVPS']
  self.df_all['cache_size/dataset'] = list(zip(self.df_all['dataset'], self.df_all['cache_size']))

  sorted_df = self.df_all.sort_values(['traces', 'cache_size', 'hit_rate'], ascending=[True, True, False])
  sorted_df['rank'] = sorted_df.groupby(['traces', 'cache_size'])['hit_rate'].rank(ascending=False, method='first')
  #print(sorted_df)
  #sorted_df.to_csv('sorted_df.csv')
  self.df_all_grouped = sorted_df.groupby(['traces', 'cache_size'])


  # In[10]:


  self.df_all_avg = self.df_all.groupby(['cache_size/dataset', 'algorithm'])['hit_rate'].mean().reset_index()
  #print(self.df_all_avg)
  sorted_df_avg = self.df_all_avg.sort_values(['cache_size/dataset', 'hit_rate'], ascending=[True, True])
  fig, ax = plt.subplots(figsize=(12, 8))
  ax= sns.barplot(x='cache_size/dataset', y='hit_rate', data=sorted_df_avg, hue='algorithm')

  label = ["0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"]
  ax.set_xticklabels(label)
  ax.set_title('CloudVps')
  ax.set_xlabel('cache size(% of workload footprint)')
  ax.set_ylabel('Average Hit Rate')

  #fig.savefig('figure_cloudvps.eps', format='eps', bbox_inches = 'tight')
  plt.show()


  # In[11]:


  self.df_all = pd.read_csv(self.infile, header=None)
  self.df_all.columns = ['traces','algorithm','cache_size','hit_rate']

  #self.df_all = self.df_all[self.df_all.algorithm != 'alecar6']
  print(len(self.df_all))
  num_traces = len(self.df_all)

  labels = list(r[-1] for r in self.df_all['traces'].str.split('/'))

  ls = []
  trunk_traces = []
  for i in labels:
    trunk_traces.append(path.splitext(i)[0])
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('vps'):
        ls.append('CloudVPS')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('webserver'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('moodle'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blkparse':
        ls.append('FIU')
    if path.splitext(i)[-1]== '.csv':
        ls.append('MSR')
    if path.splitext(i)[-1]== '.txt':
        ls.append('NEXUS')
  self.df_all['dataset'] = ls
  self.df_all['traces'] = trunk_traces
  #print(self.df_all)
  self.df_all = self.df_all[self.df_all.dataset == 'NEXUS']
  self.df_all['cache_size/dataset'] = list(zip(self.df_all['dataset'], self.df_all['cache_size']))

  sorted_df = self.df_all.sort_values(['traces', 'cache_size', 'hit_rate'], ascending=[True, True, False])
  sorted_df['rank'] = sorted_df.groupby(['traces', 'cache_size'])['hit_rate'].rank(ascending=False, method='first')
  #print(sorted_df)
  #sorted_df.to_csv('sorted_df.csv')
  self.df_all_grouped = sorted_df.groupby(['traces', 'cache_size'])


  # In[12]:


  self.df_all_avg = self.df_all.groupby(['cache_size/dataset', 'algorithm'])['hit_rate'].mean().reset_index()
  #print(self.df_all_avg)
  sorted_df_avg = self.df_all_avg.sort_values(['cache_size/dataset'], ascending=[True])
  fig, ax = plt.subplots(figsize=(24, 8))
  ax= sns.barplot(x='cache_size/dataset', y='hit_rate', data=sorted_df_avg, hue='algorithm')

  label = ["0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"]
  ax.set_xticklabels(label)
  ax.set_title('Nexus')
  ax.set_xlabel('cache size(% of workload footprint)')
  ax.set_ylabel('Average Hit Rate')

  #fig.savefig('figure_cloudvps.eps', format='eps', bbox_inches = 'tight')
  plt.show()


  # In[82]:


  self.df_all = pd.read_csv(self.infile, header=None)
  self.df_all.columns = ['traces','algorithm','cache_size','hit_rate']

  self.df_all = self.df_all[(self.df_all.algorithm != 'alecar6') & (self.df_all.algorithm != 'scanalecar')]
  print(len(self.df_all))
  num_traces = len(self.df_all)

  labels = list(r[-1] for r in self.df_all['traces'].str.split('/'))

  ls = []
  trunk_traces = []
  for i in labels:
    trunk_traces.append(path.splitext(i)[0])
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('vps'):
        ls.append('CloudVPS')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('webserver'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('moodle'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blkparse':
        ls.append('FIU')
    if path.splitext(i)[-1]== '.csv':
        ls.append('MSR')
    if path.splitext(i)[-1]== '.txt':
        ls.append('NEXUS')
  self.df_all['dataset'] = ls
  self.df_all['traces'] = trunk_traces

  self.df_all['hit_rate/algorithm'] = list(zip(self.df_all['hit_rate'], self.df_all['algorithm']))
  df_avg = self.df_all.groupby(['algorithm', 'cache_size']).mean().reset_index()
  #print(df_avg)


  # In[83]:


  fig, ax = plt.subplots()
  fig.set_size_inches(8, 6)

  row = []
  row_lru = []
  row_lfu = []
  row_arc = []
  row_lirs = []
  row_dlirs = []
  row_lecar = []
  #row_scanalecar =[]

  for index, row in df_avg.iterrows():
    if row['algorithm'] == 'opt':
        row.append(row['hit_rate'])
    if row['algorithm'] == 'lru':
        row_lru.append(row['hit_rate'])
    if row['algorithm'] == 'lfu':
        row_lfu.append(row['hit_rate'])
    if row['algorithm'] == 'arc':
        row_arc.append(row['hit_rate'])
    if row['algorithm'] == 'lirs':
        row_lirs.append(row['hit_rate'])
    if row['algorithm'] == 'dlirs':
        row_dlirs.append(row['hit_rate'])
    if row['algorithm'] == 'lecar':
        row_lecar.append(row['hit_rate'])
  #    if row['algorithm'] == 'scanalecar':
  #            row_scanalecar.append(row['hit_rate'])
        
  x_axis = ["0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"]

  #plt.plot(x_axis, row, color='k', label='OPT', marker='*', linewidth=2)
  plt.plot(x_axis, row_lru, color='#1f77b4', label='LRU', marker='o', linewidth=2)
  plt.plot(x_axis, row_lfu, color='#e377c2', label='LFU', marker='s', linewidth=2)
  plt.plot(x_axis, row_lirs, color='#bcbd22', label='LIRS', marker='x', linewidth=2)
  plt.plot(x_axis, row_arc, color='#d62728', label='ARC', marker='d', linewidth=2)
  plt.plot(x_axis, row_dlirs, color='#9467bd',label='DLIRS', marker='+', linewidth=2)
  plt.plot(x_axis, row_lecar, color='#8c564b', label='LeCaR', marker='^', linewidth=2)
  #plt.plot(x_axis, row_scanalecar, color='b', label='ScanALeCaR', marker='^', linewidth=2)
  sb.set_style("whitegrid")
  #ax.fill_between(x_axis, row, row_lirs, alpha=0.1, facecolor='red')
  plt.margins(0)
  plt.xticks(x_axis)
  plt.xlabel('Cache size (% of workload footprint)', fontsize=16)
  plt.ylabel('Average Hit Rate (%)', fontsize=16)
  #plt.setp(ax.get_xticklabels(), visible=False)
  #plt.setp(ax.get_yticklabels(), visible=False)
  plt.ylim([0, 50])
  #plt.title('Average Hit Rate across FIU, MSR, NEXUS and VISA', fontsize=15)
  plt.legend(loc='lower right')
  plt.savefig(outputfile, format='png', bbox_inches = 'tight', dpi=600)


  # In[ ]:




