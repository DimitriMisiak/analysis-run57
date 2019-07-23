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

from read_data import file_path, part_in_run, save_dir_full_cut_event

save_dir = save_dir_full_cut_event

from heat_chi2_cut import heat_chi2_cut
from ion_chi2_cut import ion_chi2_cut
from ion_offset_cut import ion_offset_cut

def full_cut_event(num):
    """ Return the indexes of the events passing the chi2 cut on the heat
    channel for the given partition.    
    """
    _, cond_heat_chi2_cut = heat_chi2_cut(num)
    _, cond_ion_chi2_cut = ion_chi2_cut(num)
    _, cond_ion_off_cut = ion_offset_cut(num)
    
    cond_full = cond_heat_chi2_cut
    for condi in (cond_ion_chi2_cut, cond_ion_off_cut):
        cond_full = np.logical_and(cond_full, condi)
    
    ind_full_cut = np.nonzero(cond_full)[0]
    
    return ind_full_cut, cond_full

#def full_cut_noise(num):
#    """ Return the indexes of the events passing the chi2 cut on the heat
#    channel for the given partition.    
#    """
#    _, cond_heat_chi2_cut = heat_chi2_cut_noise(num)
#    _, cond_ion_chi2_cut = ion_chi2_cut_noise(num)
#    _, cond_ion_off_cut = ion_offset_cut_noise(num)
#    
#    cond_full = cond_heat_chi2_cut
#    for condi in (cond_ion_chi2_cut, cond_ion_off_cut):
#        cond_full = np.logical_and(cond_full, condi)
#    
#    ind_full_cut = np.nonzero(cond_full)[0]
#    
#    return ind_full_cut, cond_full

if __name__ == '__main__':
#    
#    ion_label = ('A', 'B', 'C', 'D')
#    num_array = np.arange(part_in_run)
#    energy_of = list()
#    chi2_of = list()
#    
#    energy_of_cut = list()
#    chi2_of_cut = list()
#    
#    # non decor
#    energy_of_cut_nd = list()
#    chi2_of_cut_nd = list()
#    
#    # offset and slope
#    offset_list = list()
#    slope_list = list()
#    
#    # pulse time
#    pulse_stp = list()
#    
#    # noise
#    energy_of_noise = list()
#    chi2_of_noise = list()
#    
#    for num in num_array:
#        fp = file_path(num) 
#        root = uproot.open(fp)
#    
#        run_tree = root['RunTree_Normal']
#        
#        tree_raw = root["EventTree_trig_Normal_raw"]
#        tree_filt = root["EventTree_trig_Normal_filt"]
#        tree_filt_decor = root["EventTree_trig_Normal_filt_decor"]
#    
#        tree_noise = root["EventTree_noise_Normal_filt_decor"]
#    
#        # chi2 and energy
#        energy_array = tree_filt_decor['Energy_OF'].array()
#        chi2_array = tree_filt_decor['chi2_OF'].array()
#    
#        # label non decor = nd
#        energy_array_nd = tree_filt['Energy_OF'].array()
#        chi2_array_nd = tree_filt['chi2_OF'].array()
#    
#        energy_of.append(energy_array)
#        chi2_of.append(chi2_array)
#    
#        cut_ind,_ = full_cut_event(num)
#        
#        energy_of_cut.append(energy_array[cut_ind])
#        chi2_of_cut.append(chi2_array[cut_ind])
#        
#        energy_of_cut_nd.append(energy_array_nd[cut_ind])
#        chi2_of_cut_nd.append(chi2_array_nd[cut_ind])
#        
#        # offset with cuts
#        offset = tree_raw['Off'].array()
#        offset_list.append(offset[cut_ind])
#        
#        # slope with cuts
#        slope = tree_raw['Slope_Ion'].array()
#        slope_list.append(slope[cut_ind])
#        
#        # noise with cuts
#        energy_array_noise = tree_noise['Energy_OF_t0'].array()
#        chi2_array_noise = tree_noise['chi2_OF_t0'].array()
#    
#        cut_ind_noise,_ = full_cut_noise(num)
#        
#        energy_of_noise.append(energy_array_noise[cut_ind_noise])
#        chi2_of_noise.append(chi2_array_noise[cut_ind_noise])    
#        
#        # time stamp with cut
#        micro_stp = tree_filt_decor['MicroStp'].array()
#        num_part = tree_filt_decor['NumPart'].array()
#        freq = run_tree['f_max_heat'].array()[0]
#        hour_stp = (micro_stp/3600.) / freq + num_part
#        
#        pulse_stp.append(hour_stp[cut_ind])
#        run_duration = hour_stp[-1]
#        
#        # gain
#        chan_gain = run_tree['Chan_Gain'].array()[0]
#        gain_chal = chan_gain[0]
#        gain_ion = chan_gain[2:]
#        
#        # maintenance
#        maint_cycle = run_tree['MaintenanceCycle'].array()[0] *1.05 / 3600.
#        maint_duration = run_tree['MaintenanceDuration'].array()[0] / 3600.
#    
#    
#    
#    ### CUT ###
#    # data with decor
#    energy_cut = np.concatenate(energy_of_cut)
#    chi2_cut = np.concatenate(chi2_of_cut)
#    
#    stp_cut = np.concatenate(pulse_stp)
#    
#    offset_cut = np.concatenate(offset_list)
#    offset_chal = offset_cut[:,0].T
#    offset_ion = offset_cut[:, 2:].T
#    
#    slope_cut = np.concatenate(slope_list)
#    slope_ion = slope_cut.T
#    
#    # data without decor
#    energy_cut_nd = np.concatenate(energy_of_cut_nd)
#    chi2_cut_nd = np.concatenate(chi2_of_cut_nd)
#    
#    energy_cut_chal = energy_cut[:,0].T #keeping only chalA
#    energy_cut_ion = energy_cut[:,2:].T
#    chi2_cut_chal = chi2_cut[:, 0].T #keeping only chalA
#    chi2_cut_ion = chi2_cut[:, 2:].T
#    
#    energy_cut_chal_nd = energy_cut_nd[:,0].T #keeping only chalA
#    energy_cut_ion_nd = energy_cut_nd[:,2:].T
#    chi2_cut_chal_nd = chi2_cut_nd[:, 0].T #keeping only chalA
#    chi2_cut_ion_nd = chi2_cut_nd[:, 2:].T
#    
#    # noise
#    energy_cut_noise = np.concatenate(energy_of_noise)
#    chi2_cut_noise = np.concatenate(chi2_of_noise)
#    
#    energy_chal_noise = energy_cut_noise[:, 0].T
#    energy_ion_noise = energy_cut_noise[:, 2:].T
#    
#    chi2_chal_noise = chi2_cut_noise[:, 0].T
#    chi2_ion_noise = chi2_cut_noise[:, 2:].T
#    
#    # correcting the sign of the electrodes
#    energy_cut_conv = np.sum(energy_cut_ion_nd, axis=0)
#    
#    volt_config = np.array([[-1, -1, +1, +1]])
#    energy_cut_ion *= volt_config.T
#    energy_cut_ion_nd *= volt_config.T
#    
#    # maintenance time
#    num_maint_cycle = int(part_in_run // maint_cycle + 1)
#    
#    # =============================================================================
#    # CRYSTAL EVENT CUT PLOT + FIDUCIAL CUT
#    # =============================================================================
#    
#    ###PLOT + CUTS
#    plt.close('all')
#    
#    ion_config = ((1,3), (2,3), (1,0), (2,0))
#    
#    ### PLOT
#    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10), num='Ion vs Ion',
#                           sharex=True, sharey=True)
#    ax = np.ravel(ax)
#    
#    for i, config in enumerate(ion_config):
#        
#        ionx, iony = config
#    
#        ax[i].plot(energy_cut_ion_nd[ionx], energy_cut_ion_nd[iony], label='all events (non decor)',
#                   color='grey', ls='none', marker='.', alpha=0.3)
#        
#        ax[i].plot(energy_cut_ion[ionx], energy_cut_ion[iony], 
#                   color='r', ls='none', marker='+',
#                   label='all events = {}'.format(energy_cut_ion.shape[-1]))
#    
#        ax[i].plot(energy_fid_ion[ionx], energy_fid_ion[iony], 
#                   color='slateblue', ls='none', marker='+',
#                   label='fiducial events = {}'.format(energy_fid_ion.shape[-1]))
#    
#        ax[i].set_ylabel('Ion {} [ADU]'.format(ion_label[iony]))
#        ax[i].set_xlabel('Ion {} [ADU]'.format(ion_label[ionx]))
#        ax[i].grid(True)
#        
#    xylim = (energy_cut_ion.min()*1.08, energy_cut_ion.max()*1.08)
#    
#    vline = [y*1.1 for y in xylim]
#    hline = [x*1.1 for x in xylim]
#    e_inf = [e_cut_inf, e_cut_inf]
#    e_sup = [e_cut_sup, e_cut_sup]
#    
#    ax[2].fill_between(vline, e_inf, e_sup, color='lime', alpha=0.5, 
#                       label='fiducial cut\ne_inf={0} ADU\ne_sup={1} ADU'.format(e_cut_inf, e_cut_sup))
#    ax[3].fill_between(vline, e_inf, e_sup, color='lime', alpha=0.5)
#    
#    ax[1].fill_betweenx(vline, e_inf, e_sup, color='lime', alpha=0.5)
#    ax[3].fill_betweenx(vline, e_inf, e_sup, color='lime', alpha=0.5)
#    
#    ax[2].legend(title='With quality cuts:')
#    
#    ax[0].set_title('Ion vs Ion with Quality Cuts')
#    ax[0].set_xlim(*xylim)
#    ax[0].set_ylim(*xylim)
#    
#    fig.tight_layout()
#    fig.savefig(save_dir+'/ion_vs_ion.png')
#    
#    # =============================================================================
#    # Ionization Histogramm
#    # =============================================================================
#    
#    #### Histogramm check
#    bin_edges = np.histogram_bin_edges((energy_cut_ion), bins=100)
#    
#    bin_array = bin_edges[1:] - (bin_edges[1]-bin_edges[0])/2
#    
#    cut_hist_list = list()
#    fid_hist_list = list()
#    for i in range(4):
#        cut_hist, _ = np.histogram(energy_cut_ion[i], bins=bin_edges)
#        cut_hist_list.append(cut_hist)
#        
#        fid_hist, _ = np.histogram(energy_fid_ion[i], bins=bin_edges)
#        fid_hist_list.append(fid_hist)
#    
#    energy_cut_ion_sorted = np.sort(energy_cut_ion, axis=1)
#    ndim = energy_cut_ion_sorted.shape[-1]
#    cdf_cut = (np.arange(ndim)+1) / float(ndim)
#    
#    energy_fid_ion_sorted = np.sort(energy_fid_ion, axis=1)
#    ndim = energy_fid_ion_sorted.shape[-1]
#    cdf_fid = (np.arange(ndim)+1) / float(ndim)
#    
#    ### PLOT
#    fig, ax = plt.subplots(nrows=4, figsize=(7,10), sharex=True,
#                           num='Ion. energy spectrum')
#    ax = np.ravel(ax)
#    
#    for i,a in enumerate(ax):
#        
#        lab = ion_label[i]
#        
#        hist_line, = a.plot(bin_array, cut_hist_list[i], ls='steps-mid', color='r')
#    
#        a.fill_between(bin_array, cut_hist_list[i], color='coral',
#                       alpha=0.3, step='mid')
#    
#        hist_line_2, = a.plot(bin_array, fid_hist_list[i], ls='steps-mid', color='slateblue')
#    
#        a.fill_between(bin_array, fid_hist_list[i], color='deepskyblue',
#                       alpha=0.3, step='mid')
#    
#        a0 = a.twinx()
#        a0.set_ylabel('CDF', color='grey')
#        a0.tick_params(axis='y', labelcolor='grey')
#        
#        cdf_line, = a0.plot(energy_cut_ion_sorted[i], cdf_cut, ls='steps', color='coral',
#                            path_effects=style)
#    
#        cdf_line_2, = a0.plot(energy_fid_ion_sorted[i], cdf_fid, ls='steps', color='deepskyblue',
#                            path_effects=style)
#        
#        a.grid(True)
#        a.set_ylabel('Counts Ion {}'.format(lab), color='k')
#        a.tick_params(axis='y', labelcolor='k')
#        a.set_xlabel('Energy [ADU]')
#    #    a.set_yscale('log')
#    
#        sens = xmax_histo(bin_array, fid_hist_list[i])
#        a.axvline(sens, color='k', ls='-.', lw=0.5)
#    
#        a0.legend((hist_line, cdf_line, hist_line_2, cdf_line_2), 
#                  ('Hist quality evts', 'Cdf Quality evts', 'Hist FID evts', 'Cdf FID evts'),
#                  title=(
#                          'Ion {0}\n$S_{{V,ion}}$= {1:.3f} ADU/keV\nGain={2}nV/ADU'
#                  ).format(lab, sens/10, gain_ion[i]),
#                  loc=0
#        )
#    
#    ax[0].set_title(
#            ('Ion Channels: {} Quality Events, {} Fiducial Events'
#             ).format(energy_cut_ion.shape[-1], energy_fid_ion.shape[-1])
#    )
#    
#    fig.tight_layout()
#    fig.savefig(save_dir+'/ion_energy_spectrum.png')
#    
#    # =============================================================================
#    # Heat Histogramm 
#    # =============================================================================
#    
#    ### Histogramm check
#    
#    bin_edges = np.histogram_bin_edges(energy_cut_chal, bins=250)
#    
#    bin_array = bin_edges[1:] - (bin_edges[1]-bin_edges[0])/2
#    
#    hist_cut_chal, _ = np.histogram(energy_cut_chal, bins=bin_edges)
#    energy_cut_chal_sorted = np.sort(energy_cut_chal, axis=0)
#    ndim_cut = energy_cut_chal_sorted.shape[0]
#    cdf_cut = (np.arange(ndim_cut)+1) / float(ndim_cut)
#    
#    hist_fid_chal, _ = np.histogram(energy_fid_chal, bins=bin_edges)
#    energy_fid_chal_sorted = np.sort(energy_fid_chal, axis=0)
#    ndim_fid = energy_fid_chal_sorted.shape[0]
#    cdf_fid = (np.arange(ndim_fid)+1) / float(ndim_fid)
#    
#    #### PLOT
#    #fig = plt.figure(figsize=(7,5), num='Chal energy spectrum')
#    #ax = fig.subplots()
#    
#    
#    fig, axis = plt.subplots(nrows=2, figsize=(10,7), num='Ion. B+D and Chal energy spectrum')
#    
#    
#    ### CHAL HISTO
#    ax = axis[1]
#    ax0 = ax.twinx()
#    
#    ax.set_title('Chal A')
#    
#    hist_line, = ax.plot(bin_array, hist_cut_chal, ls='steps-mid', color='r')
#    
#    ax.fill_between(bin_array, hist_cut_chal, color='coral', alpha=0.3, step='mid')
#    
#    hist_line_2, = ax.plot(bin_array, hist_fid_chal, ls='steps-mid', color='slateblue')
#    
#    ax.fill_between(bin_array, hist_fid_chal, color='deepskyblue', alpha=0.3, step='mid')
#    
#    cdf_line, = ax0.plot(energy_cut_chal_sorted, cdf_cut, ls='steps', color='coral',
#                         path_effects=style)
#    
#    cdf_line_2, = ax0.plot(energy_fid_chal_sorted, cdf_fid, ls='steps', color='deepskyblue',
#                           path_effects=style)
#    
#    ax.grid(True)
#    ax.set_ylabel('Counts', color='k')
#    ax.tick_params(axis='y', labelcolor='k')
#    
#    ax0.set_ylabel('CDF', color='grey')
#    ax0.tick_params(axis='y', labelcolor='grey')
#    
#    ax.set_xlabel('Energy [ADU]')
#    #ax.set_xlim(energy_cut_chal_sorted.min(), energy_cut_chal_sorted.max())
#    
#    sens = xmax_histo(bin_array, hist_fid_chal)
#    ax.axvline(sens, color='k', ls='-.', lw=0.5)
#    
#    ax.legend((hist_line, cdf_line, hist_line_2, cdf_line_2), 
#              ('Hist quality evts', 'Cdf Quality evts', 'Hist FID evts', 'Cdf FID evts'),
#              title='ChalA\n$S_{{V,chal}}$= {:.3f} ADU/keV\nGain={}nV/ADU'.format(sens/10, gain_chal), loc=2)
#    
#    #fig.tight_layout()
#    #fig.savefig(save_dir+'/chal_energy_spectrum.png')
#    
#    # =============================================================================
#    # HISTOGRAMM B+D
#    # =============================================================================
#    energy_ion_bd_fid = energy_fid_ion[1] + energy_fid_ion[3]
#    
#    ### Histogramm check
#    
#    bin_edges = np.histogram_bin_edges(energy_ion_bd , bins=200)
#    
#    bin_array = bin_edges[1:] - (bin_edges[1]-bin_edges[0])/2
#    
#    hist_ion_bd, _ = np.histogram(energy_ion_bd, bins=bin_edges)
#    hist_ion_bd_fid, _ = np.histogram(energy_ion_bd_fid, bins=bin_edges)
#    
#    energy_ion_bd_sorted = np.sort(energy_ion_bd, axis=0)
#    ndim_bd = energy_ion_bd_sorted.shape[0]
#    cdf_bd = (np.arange(ndim_bd)+1) / float(ndim_bd)
#    
#    energy_ion_bd_sorted_fid = np.sort(energy_ion_bd_fid, axis=0)
#    ndim_bd_fid = energy_ion_bd_sorted_fid.shape[0]
#    cdf_bd_fid = (np.arange(ndim_bd_fid)+1) / float(ndim_bd_fid)
#    
#    ### PLOT
#    #fig = plt.figure(figsize=(7,5), num='Ion B+D energy spectrum')
#    #ax = fig.subplots()
#    
#    ax = axis[0]
#    ax0 = ax.twinx()
#    
#    ax.set_title(
#            ('Ion. B+D: {} Quality Events, {} Fiducial Events'
#             ).format(energy_cut_chal.shape[-1], energy_fid_chal.shape[-1])
#    )
#    
#    hist_line, = ax.plot(bin_array, hist_ion_bd, ls='steps-mid', color='r')
#    ax.fill_between(bin_array, hist_ion_bd, color='coral', alpha=0.3, step='mid')
#    
#    hist_line_2, = ax.plot(bin_array, hist_ion_bd_fid, ls='steps-mid', color='slateblue')
#    ax.fill_between(bin_array, hist_ion_bd_fid, color='deepskyblue', alpha=0.3, step='mid')
#    
#    cdf_line, = ax0.plot(energy_ion_bd_sorted, cdf_bd, ls='steps', color='coral',
#                         path_effects=style)
#    
#    cdf_line, = ax0.plot(energy_ion_bd_sorted_fid, cdf_bd_fid, ls='steps', color='deepskyblue',
#                         path_effects=style)
#    
#    ax.grid(True)
#    ax.set_ylabel('Counts', color='k')
#    ax.tick_params(axis='y', labelcolor='k')
#    
#    ax0.set_ylabel('CDF', color='grey')
#    ax0.tick_params(axis='y', labelcolor='grey')
#    
#    ax.set_xlabel('Energy [ADU]')
#    
#    sens_bd = xmax_histo(bin_array, hist_ion_bd_fid)
#    ax.axvline(sens_bd, color='k', ls='-.', lw=0.5)
#    
#    ax.legend((hist_line, cdf_line, hist_line_2, cdf_line_2), 
#              ('Hist quality evts', 'Cdf Quality evts', 'Hist FID evts', 'Cdf FID evts'),
#              title='Ion. B+D\n$S_{{V,ion}}$= {:.3f} ADU/keV\nGain={}nV/ADU'.format(sens_bd/10, gain_ion[0]), loc=2)
#    
#    fig.tight_layout()
#    fig.savefig(save_dir+'/ion_bd_chal_energy_spectrum.png')
       
        
    
    
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
    
    bin_edges = np.histogram_bin_edges(energy_cut_ion[0], bins=1000)
    
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
