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

num_array = np.arange(3)
#num_array = (0,)

energy_of = list()
chi2_of = list()
ind_heat_cut = list()

for num in num_array:
    fp = file_path(num) 
    root = uproot.open(fp)

    tree_run = root["RunTree_Normal"]
    polar_ion = tree_run['Polar_Ion'].array()
    
#    tree_event_trig_raw = root["EventTree_trig_Normal_raw"]
    tree_event_trig_filt = root["EventTree_trig_Normal_filt"]
    tree_event_trig_filt_decor = root["EventTree_trig_Normal_filt_decor"]
#    tree_event_noise_raw = root["EventTree_noise_Normal_raw"]
#    tree_event_noise_filt = root["EventTree_noise_Normal_filt"]
#    tree_event_noise_filt_decor = root["EventTree_noise_Normal_filt_decor"]

#    energy_of.append(tree_event_trig_filt['Energy_OF_i'].array())
#    chi2_of.append(tree_event_trig_filt['chi2_OF_i'].array())

    energy_of.append(tree_event_trig_filt_decor['Energy_OF'].array())
    chi2_of.append(tree_event_trig_filt_decor['chi2_OF'].array())

    ind_heat_cut.append(full_cut(num)[0])

#energy = np.concatenate(energy_of)
#chi2 = np.concatenate(chi2_of)
#
#energy_chal = energy[:,:2].T
#energy_ion = energy[:,2:].T
#
#chi2_chal = chi2[:, :2].T
#chi2_ion = chi2[:, 2:].T

###PLOT + CUTS
plt.close('all')
# =============================================================================
# CHI2 vs AMP comparison partition
# =============================================================================
energy_ion = [energy_part[ind,2:].T for energy_part, ind in zip(energy_of, ind_heat_cut)]
chi2_ion = [chi2_part[ind,2:].T for chi2_part, ind in zip(chi2_of, ind_heat_cut)]

# to have positive signal, whatever the polarisation
# +1 * negative polar collecting holes
# -1 * positive polar collecting electrons
sign_polar = np.sign(polar_ion)
energy_ion = [(- energy_part.T * sign_polar).T for energy_part in energy_ion]

chi2_thresh = 3000

cut_ok_index = list()
for num in num_array:
    
    c_array = chi2_ion[num]
    
    cond_list = list()
    for i in range(4):
        cond = c_array[i]<chi2_thresh
        cond_list.append( cond )

    cond_array = np.logical_and.reduce(cond_list)
    cut_ok_index_num = np.nonzero(cond_array)[0]

    cut_ok_index.append(cut_ok_index_num)
    
energy_ion_ok = [e_array[:,ind_array] for e_array, ind_array in zip(energy_ion, cut_ok_index)]
chi2_ion_ok = [c_array[:,ind_array] for c_array, ind_array in zip(chi2_ion, cut_ok_index)]

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15,15),
                       num='Ion. Chi2 vs Amp')
ax = np.ravel(ax)

for i,a in enumerate(ax):
    
    for num in num_array:

        # all points
        e_array_no = energy_ion[num][i]
        c_array_no = chi2_ion[num][i]
        
        ind_pos_no = np.nonzero(e_array_no>0)
        ind_neg_no = np.nonzero(e_array_no<0)
        
        p_pos = a.loglog(e_array_no[ind_pos_no], c_array_no[ind_pos_no],
                         ls='none', marker='+', alpha=0.1)
    
        col = p_pos[0].get_color()
    
        p_neg = a.loglog(-e_array_no[ind_neg_no], c_array_no[ind_neg_no],
                         ls='none', marker='x', color=col, alpha=0.1)
        
        # points passing cut    
        e_array_ok = energy_ion_ok[num][i]
        c_array_ok = chi2_ion_ok[num][i]
        
        ind_pos_ok = np.nonzero(e_array_ok>0)
        ind_neg_ok = np.nonzero(e_array_ok<0)
        
        p_pos = a.loglog(e_array_ok[ind_pos_ok], c_array_ok[ind_pos_ok],
                         ls='none', marker='+', color=col)
    
        p_neg = a.loglog(-e_array_ok[ind_neg_ok], c_array_ok[ind_neg_ok],
                         ls='none', marker='x', color=col)

    a.set_ylabel('Chi2_{}'.format(i))
    a.set_xlabel('Energy [ADU]')
    a.grid(True)
    a.axhline(chi2_thresh, color='k')
    a.legend(title='Ion {}'.format(i))

fig.tight_layout()

# =============================================================================
# IONIZATION ENERGY SPECTRUM
# =============================================================================
energy_ion_ok = np.concatenate(energy_ion_ok, axis=1)
chi2_ion_ok = np.concatenate(chi2_ion_ok, axis=1)

bin_edges = np.histogram_bin_edges(energy_ion_ok[0], bins=100)

bin_array = bin_edges[1:] - (bin_edges[1]-bin_edges[0])/2

hist_list = list()
for i in range(4):
    hist, _ = np.histogram(energy_ion_ok[i], bins=bin_edges)
    hist_list.append(hist)

energy_ion_sorted = np.sort(energy_ion_ok, axis=1)
ndim = energy_ion_sorted.shape[-1]
cfd = (np.arange(ndim)+1) / float(ndim)

### PLOT
fig, ax = plt.subplots(nrows=4, figsize=(7,20), sharex=True,
                       num='Ion. energy spectrum')
ax = np.ravel(ax)

for i,a in enumerate(ax):
    
    a.plot(bin_array, hist_list[i], ls='steps-mid', color='slateblue')
    
    a0 = a.twinx()
    a0.set_ylabel('CFD', color='coral')
    a0.tick_params(axis='y', labelcolor='coral')
    a0.plot(energy_ion_sorted[i], cfd, ls='steps', color='coral')
    
    a.grid(True)
    a.set_ylabel('Counts Ion {}'.format(i), color='slateblue')
    a.tick_params(axis='y', labelcolor='slateblue')
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
    
    ax[i].plot(energy_ion_ok[ionx], energy_ion_ok[iony], ls='none', marker='+')

    ax[i].set_ylabel('Ion {}'.format(iony))
    ax[i].set_xlabel('Ion {}'.format(ionx))
    ax[i].grid(True)

fig.tight_layout()

### energy are not calibrated so not working :/
