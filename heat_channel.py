#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:41:10 2019

Exploring the root file containing the data.

@author: misiak
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt

from read_data import file_path

num_array = np.arange(3)

energy_of = list()
chi2_of = list()

for num in num_array:
    fp = file_path(num) 
    root = uproot.open(fp)

#    tree_run = root["RunTree_Normal"]
#    print(tree_run['Polar_Ion'].array())
    
#    tree_event_trig_raw = root["EventTree_trig_Normal_raw"]
    tree_event_trig_filt = root["EventTree_trig_Normal_filt"]
    tree_event_trig_filt_decor = root["EventTree_trig_Normal_filt_decor"]
#    tree_event_noise_raw = root["EventTree_noise_Normal_raw"]
#    tree_event_noise_filt = root["EventTree_noise_Normal_filt"]
#    tree_event_noise_filt_decor = root["EventTree_noise_Normal_filt_decor"]

    energy_of.append(tree_event_trig_filt['Energy_OF'].array())

    chi2_of.append(tree_event_trig_filt['chi2_OF'].array())

energy = np.concatenate(energy_of)
chi2 = np.concatenate(chi2_of)


energy_chal = energy[:,0].T #keeping only chalA
energy_ion = energy[:,2:].T

chi2_chal = chi2[:, 0].T #keeping only chalA
chi2_ion = chi2[:, 2:].T

###PLOT + CUTS
plt.close('all')
# =============================================================================
# CHI2 CUT
# =============================================================================
chi2_thresh = 300

chi2_cut_cond = chi2_chal<chi2_thresh

ind_chi2_cut = np.where(chi2_cut_cond)[0]

energy_chal_ok = energy_chal[ind_chi2_cut]
chi2_chal_ok = chi2_chal[ind_chi2_cut]

### PLOT Chi2 vs Amp
fig = plt.figure(figsize=(7,5), num='Chal Chi2 vs Amp')
ax = fig.subplots()

ax.set_title('ChalA')
ax.set_ylabel('Chi2')
ax.set_xlabel('Energy [ADU]')
ax.loglog(energy_chal, chi2_chal, ls='none', marker='+', 
             color='r', alpha=0.5)
ax.loglog(energy_chal_ok, chi2_chal_ok, ls='none', marker='+', color='b')

ax.grid(True)
ax.axhline(chi2_thresh, color='k')

fig.tight_layout()

## =============================================================================
## Chi2 Histogramm
## =============================================================================

bin_edges = np.histogram_bin_edges(chi2_chal_ok, bins=60)

bin_array = bin_edges[1:] - (bin_edges[1]-bin_edges[0])/2

hist_chi2, _ = np.histogram(chi2_chal_ok, bins=bin_edges)

chi2_chal_sorted = np.sort(chi2_chal_ok, axis=0)
ndim = chi2_chal_sorted.shape[0]
cfd = (np.arange(ndim)+1) / float(ndim)

### PLOT
fig = plt.figure(figsize=(7,5), num='Chal energy spectrum')
ax = fig.subplots()

ax.set_title('ChalA')

ax.plot(bin_array, hist_chi2, ls='steps-mid', color='slateblue')

axt = ax.twinx()
axt.set_ylabel('CFD', color='coral')
axt.tick_params(axis='y', labelcolor='coral')
axt.plot(chi2_chal_sorted, cfd, ls='steps', color='coral')

ax.grid(True)
ax.set_ylabel('Counts', color='slateblue')
ax.tick_params(axis='y', labelcolor='slateblue')
ax.set_xlabel('Energy [ADU]')
    
fig.tight_layout()
