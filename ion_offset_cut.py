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


def ion_offset_cut(num, off_thresh=14000):
    """ Return the indexes of the events passing the chi2 cut on the heat
    channel for the given partition.    
    Also return the truth array.
    """
    fp = file_path(num) 
    root = uproot.open(fp)
 
    tree = root["EventTree_trig_Normal_raw"]
    
    offset = tree['Off'].array()
    offset_ion = offset[:, 2:].T #keeping only ion channel
    
    offset_cut_cond = np.abs(offset_ion)<off_thresh

    cond = offset_cut_cond[0]
    for cond_aux in offset_cut_cond[1:]:
        cond = np.logical_and(cond, cond_aux)

    ind_chi2_cut = np.nonzero(cond)[0]
    
    return ind_chi2_cut, cond

def ion_offset_cut_noise(num, off_thresh=14000):
    """ Return the indexes of the events passing the chi2 cut on the heat
    channel for the given partition.    
    Also return the truth array.
    """
    fp = file_path(num) 
    root = uproot.open(fp)
 
    tree = root["EventTree_noise_Normal_filt_decor"]
    
    offset = tree['Off'].array()
    offset_ion = offset[:, 2:].T #keeping only ion channel
    
    offset_cut_cond = np.abs(offset_ion)<off_thresh

    cond = offset_cut_cond[0]
    for cond_aux in offset_cut_cond[1:]:
        cond = np.logical_and(cond, cond_aux)

    ind_chi2_cut = np.nonzero(cond)[0]
    
    return ind_chi2_cut, cond


if __name__ == '__main__':
    
    ion_label = ('A', 'B', 'C', 'D')
    off_thresh = 14000
    num_array = np.arange(3)
    
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
    
        energy_array = tree_event_trig_filt_decor['Energy_OF'].array()
        chi2_array = tree_event_trig_filt_decor['chi2_OF'].array()
    
        energy_of.append(energy_array)
        chi2_of.append(chi2_array)
    
        cut_ind, _ = ion_offset_cut(num, off_thresh)
        heat_cut.append(cut_ind)
        energy_of_cut.append(energy_array[cut_ind])
        chi2_of_cut.append(chi2_array[cut_ind])
    
    # raw data
    energy_raw = np.concatenate(energy_of)
    chi2_raw = np.concatenate(chi2_of)
    
    energy_raw_chal = energy_raw[:,0].T #keeping only chalA
    energy_raw_ion = energy_raw[:,2:].T
    chi2_raw_chal = chi2_raw[:, 0].T #keeping only chalA
    chi2_raw_ion = chi2_raw[:, 2:].T
    
    # data with heat cut
    energy_cut = np.concatenate(energy_of_cut)
    chi2_cut = np.concatenate(chi2_of_cut)
    
    energy_cut_chal = energy_cut[:,0].T #keeping only chalA
    energy_cut_ion = energy_cut[:,2:].T
    chi2_cut_chal = chi2_cut[:, 0].T #keeping only chalA
    chi2_cut_ion = chi2_cut[:, 2:].T
    
    
    
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
    ax.loglog(energy_raw_chal, chi2_raw_chal, ls='none', marker='+', 
                 color='r', alpha=0.5)
    ax.loglog(energy_cut_chal, chi2_cut_chal, ls='none', marker='+', color='b')
    
    ax.grid(True)
    
    fig.tight_layout()
    
    ### Histogramm check
    
    bin_edges = np.histogram_bin_edges(energy_cut_chal, bins=2000)
    
    bin_array = bin_edges[1:] - (bin_edges[1]-bin_edges[0])/2
    
    hist_cut_chal, _ = np.histogram(energy_cut_chal, bins=bin_edges)
    
    
    energy_cut_chal_sorted = np.sort(energy_cut_chal, axis=0)
    ndim_cut = energy_cut_chal_sorted.shape[0]
    cfd_cut = (np.arange(ndim_cut)+1) / float(ndim_cut)
    
    hist_raw_chal, _ = np.histogram(energy_raw_chal, bins=bin_edges)
    
    energy_raw_chal_sorted = np.sort(energy_raw_chal, axis=0)
    ndim_raw = energy_raw_chal_sorted.shape[0]
    cfd_raw = (np.arange(ndim_raw)+1) / float(ndim_raw)
    
    ### PLOT
    fig = plt.figure(figsize=(7,5), num='Chal energy spectrum')
    ax = fig.subplots()
    ax0 = ax.twinx()
    
    ax.set_title('ChalA')
    
    ax.plot(bin_array, hist_raw_chal, ls='steps-mid', color='r')
    
    ax.plot(bin_array, hist_cut_chal, ls='steps-mid', color='b')
    
    ax0.plot(energy_raw_chal_sorted, cfd_raw, ls='steps', color='coral')
    
    ax0.plot(energy_cut_chal_sorted, cfd_cut, ls='steps', color='slateblue')
    
    ax.grid(True)
    ax.set_ylabel('Counts', color='k')
    ax.tick_params(axis='y', labelcolor='k')
    
    ax0.set_ylabel('CFD', color='grey')
    ax0.tick_params(axis='y', labelcolor='grey')
    
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
        ax.loglog(np.abs(energy_raw_ion[i]), chi2_raw_ion[i], ls='none', marker='+', 
                  color='r', alpha=0.5)
        ax.loglog(np.abs(energy_cut_ion[i]), chi2_cut_ion[i], ls='none', marker='+', color='b')
        
        ax.grid(True)
        
    fig.tight_layout()
    
    #### Histogramm check
    
    bin_edges = np.histogram_bin_edges(energy_cut_ion[0], bins=10000)
    
    bin_array = bin_edges[1:] - (bin_edges[1]-bin_edges[0])/2
    
    cut_hist_list = list()
    raw_hist_list = list()
    for i in range(4):
        cut_hist, _ = np.histogram(energy_cut_ion[i], bins=bin_edges)
        cut_hist_list.append(cut_hist)
        raw_hist, _ = np.histogram(energy_raw_ion[i], bins=bin_edges)
        raw_hist_list.append(raw_hist)
    
    energy_cut_ion_sorted = np.sort(energy_cut_ion, axis=1)
    ndim = energy_cut_ion_sorted.shape[-1]
    cfd_cut = (np.arange(ndim)+1) / float(ndim)
    
    energy_raw_ion_sorted = np.sort(energy_raw_ion, axis=1)
    ndim = energy_raw_ion_sorted.shape[-1]
    cfd_raw = (np.arange(ndim)+1) / float(ndim)
    
    ### PLOT
    fig, ax = plt.subplots(nrows=4, figsize=(7,20), sharex=True,
                           num='Ion. energy spectrum')
    ax = np.ravel(ax)
    
    for i,a in enumerate(ax):
        
        a.plot(bin_array, raw_hist_list[i], ls='steps-mid', color='r')
        a.plot(bin_array, cut_hist_list[i], ls='steps-mid', color='b')
        
        a0 = a.twinx()
        a0.set_ylabel('CFD', color='grey')
        a0.tick_params(axis='y', labelcolor='grey')
        
        a0.plot(energy_raw_ion_sorted[i], cfd_raw, ls='steps', color='coral')
        a0.plot(energy_cut_ion_sorted[i], cfd_cut, ls='steps', color='deepskyblue')
        
        a.grid(True)
        a.set_ylabel('Counts Ion {}'.format(i), color='k')
        a.tick_params(axis='y', labelcolor='k')
        a.set_xlabel('Energy [ADU]')
        a.set_yscale('log')
        
    fig.tight_layout()
