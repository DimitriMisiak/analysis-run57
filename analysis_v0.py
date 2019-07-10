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
from full_cut import full_cut

ion_label = ('A', 'B', 'C', 'D')
num_array = np.arange(3)
#num_array = [0,]

energy_of = list()
chi2_of = list()

heat_cut = list()
energy_of_cut = list()
chi2_of_cut = list()

for num in num_array:
    fp = file_path(num) 
    root = uproot.open(fp)

    tree_event_trig_filt = root["EventTree_trig_Normal_filt"]
    tree_event_trig_filt_decor = root["EventTree_trig_Normal_filt_decor"]

#    energy_array = tree_event_trig_filt_decor['Energy_OF'].array()
#    chi2_array = tree_event_trig_filt_decor['chi2_OF'].array()

    energy_array = tree_event_trig_filt['Energy_OF'].array()
    chi2_array = tree_event_trig_filt['chi2_OF'].array()

    energy_of.append(energy_array)
    chi2_of.append(chi2_array)

    cut_ind,_ = full_cut(num)
    heat_cut.append(cut_ind)
    energy_of_cut.append(energy_array[cut_ind])
    chi2_of_cut.append(chi2_array[cut_ind])

# data with heat cut
energy_cut = np.concatenate(energy_of_cut)
chi2_cut = np.concatenate(chi2_of_cut)

energy_cut_chal = energy_cut[:,0].T #keeping only chalA
energy_cut_ion = energy_cut[:,2:].T
chi2_cut_chal = chi2_cut[:, 0].T #keeping only chalA
chi2_cut_ion = chi2_cut[:, 2:].T

# correcting the sign of the electrodes
volt_config = np.array([[-1, -1, +1, +1]])
energy_cut_ion *= volt_config.T

###PLOT + CUTS
plt.close('all')
# =============================================================================
# Heat Chi2(Energy)
# =============================================================================

### PLOT Chi2 vs Amp
fig = plt.figure(figsize=(7,5), num='Chal Chi2 vs Amp')
ax = fig.subplots()

ax.set_title('ChalA')
ax.set_ylabel('Chi2')
ax.set_xlabel('Energy [ADU]')

ax.loglog(energy_cut_chal, chi2_cut_chal, ls='none', marker='+', color='slateblue')

ax.grid(True)

fig.tight_layout()

### Histogramm check

bin_edges = np.histogram_bin_edges(energy_cut_chal, bins=500)

bin_array = bin_edges[1:] - (bin_edges[1]-bin_edges[0])/2

hist_cut_chal, _ = np.histogram(energy_cut_chal, bins=bin_edges)


energy_cut_chal_sorted = np.sort(energy_cut_chal, axis=0)
ndim_cut = energy_cut_chal_sorted.shape[0]
cfd_cut = (np.arange(ndim_cut)+1) / float(ndim_cut)

### PLOT
fig = plt.figure(figsize=(7,5), num='Chal energy spectrum')
ax = fig.subplots()
ax0 = ax.twinx()

ax.set_title('ChalA')

ax.plot(bin_array, hist_cut_chal, ls='steps-mid', color='slateblue')

ax0.plot(energy_cut_chal_sorted, cfd_cut, ls='steps', color='coral')

ax.grid(True)
ax.set_ylabel('Counts', color='slateblue')
ax.tick_params(axis='y', labelcolor='slateblue')

ax0.set_ylabel('CFD', color='coral')
ax0.tick_params(axis='y', labelcolor='coral')

ax.set_xlabel('Energy [ADU]')
ax.set_xlim(energy_cut_chal_sorted.min(), energy_cut_chal_sorted.max())
fig.tight_layout()

# =============================================================================
# Ionization Chi2(Energy) wtf is happening ???
# =============================================================================

### PLOT Chi2 vs Amp

fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(14,10),
                       num='Ion Chi2 vs Amp', sharex=True, sharey=True)
axis = np.ravel(axis)

for i,ax in enumerate(axis):
    
    lab = ion_label[i]
    ax.set_title('Ion {}'.format(lab))
    ax.set_ylabel('Chi2 {}'.format(lab))
    ax.set_xlabel('ABS Energy {} [ADU]'.format(lab))

    ax.loglog(np.abs(energy_cut_ion[i]), chi2_cut_ion[i], ls='none', marker='+', color='slateblue')
    
    ax.grid(True)
    
fig.tight_layout()

#### Histogramm check

bin_edges = np.histogram_bin_edges(energy_cut_ion, bins=500)

bin_array = bin_edges[1:] - (bin_edges[1]-bin_edges[0])/2

cut_hist_list = list()
for i in range(4):
    cut_hist, _ = np.histogram(energy_cut_ion[i], bins=bin_edges)
    cut_hist_list.append(cut_hist)


energy_cut_ion_sorted = np.sort(energy_cut_ion, axis=1)
ndim = energy_cut_ion_sorted.shape[-1]
cfd_cut = (np.arange(ndim)+1) / float(ndim)

### PLOT
fig, ax = plt.subplots(nrows=4, figsize=(7,20), sharex=True,
                       num='Ion. energy spectrum')
ax = np.ravel(ax)

for i,a in enumerate(ax):
    
    lab = ion_label[i]
    
    a.plot(bin_array, cut_hist_list[i], ls='steps-mid', color='slateblue')
    
    a0 = a.twinx()
    a0.set_ylabel('CFD', color='coral')
    a0.tick_params(axis='y', labelcolor='coral')
    
    a0.plot(energy_cut_ion_sorted[i], cfd_cut, ls='steps', color='coral')
    
    a.grid(True)
    a.set_ylabel('Counts Ion {}'.format(lab), color='slateblue')
    a.tick_params(axis='y', labelcolor='k')
    a.set_xlabel('Energy [ADU]')
    a.set_yscale('log')
    
fig.tight_layout()

# =============================================================================
# CRYSTAL EVENT CUT PLOT
# =============================================================================
ion_config = ((1,3), (2,3), (1,0), (2,0))

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,15), num='Ion vs Ion',
                       sharex=True, sharey=True)
ax = np.ravel(ax)

for i, config in enumerate(ion_config):
    
    ionx, iony = config
    
    ax[i].plot(energy_cut_ion[ionx], energy_cut_ion[iony], ls='none', marker='+')

    ax[i].set_ylabel('Ion {}'.format(ion_label[iony]))
    ax[i].set_xlabel('Ion {}'.format(ion_label[ionx]))
    ax[i].grid(True)

fig.tight_layout()

### energy are not calibrated so not working :/
