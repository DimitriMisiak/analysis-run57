#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 19:19:50 2019

@author: misiak
"""
import matplotlib.pyplot as plt

from spec_classes import Analysis_red
import representation as r

import os
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    # first command
    plt.close('all')
    plt.rcParams['text.usetex']=True

    savedir = 'output/batch'

#    run_list = ('tg22l008', 'tg22l010', 'tg22l012')
    run_list = np.loadtxt('batch_run', dtype=str)
    run_list = run_list
    
    detector_list = ('RED80', 'RED70')
    
    save_choice = True
    
    failed_list = list()
    
    results_list = [[None, None]]*len(run_list)
    
    for i in tqdm(range(len(run_list))):
        run = run_list[i]
        
        for j in range(len(detector_list)):
            
            detector = detector_list[j]
            
            try:
                savepath = '/'.join([savedir, run])
                
                if detector == 'RED80':
                    ana = Analysis_red(run, detector='RED80')
                elif detector == 'RED70':
                    ana = Analysis_red(run, detector='RED70',
                                         chan_valid=(0, 2, 3, 4, 5),
                                         chan_signal=(0,)
                                         )   
                else:
                    raise Exception('Detector {} not recognized.'.format(detector))
            
                fig_temp = r.temporal_plot(ana)
                fig_chi2_trig, fig_chi2_noise = r.plot_chi2_vs_energy(ana)
                fig_hist_trig, fig_hist_noise = r.histogram_adu(ana)
                fig_hist_trig_ev, fig_hist_noise_ev = r.histogram_ev(ana)
                fig_ion = r.ion_vs_ion(ana)
                fig_virtual = r.virtual_vs_virtual_ev(ana)
                
                if save_choice is True:
                    
                    sdir = '/'.join([savedir, run, detector])
                    os.makedirs(sdir, exist_ok=True)
                    
                    fig_list = (fig_temp, fig_chi2_trig, fig_chi2_noise,
                                fig_hist_trig, fig_hist_noise,
                                fig_hist_trig_ev, fig_hist_noise_ev,
                                fig_ion, fig_virtual)
                    
                    lab_list = ('0_temporal_plot',
                               '1_chi2_trig',
                               '2_chi2_noise',
                               '3_hist_trig_adu',
                               '4_hist_trig_ev',
                               '5_hist_noise_adu',
                               '6_hist_noise_ev',
                               '7_ion_vs_ion'
                               '8_virtual_vs_virtual'
                               )
                
                    for fig, lab in zip(fig_list, lab_list):
                        if fig is not None:
                            fname = '_'.join([run, detector, lab])
                            save_path = sdir + '/' + lab + '.pdf'
                            fig.savefig(save_path)
                        
                    results = r.optimization_info(ana, 0)
                    save_r = np.array([(run, ) + results])
#                    results_list[i][j] = (run,) + results
                    
                    with open('output/batch/batch_{}.txt'.format(detector), 'ba') as file:
                        np.savetxt(file, save_r, fmt='%s', delimiter='\t')
            except:
                print(run + ' ' + detector + ' has failed us...')
                failed_list.append((run, detector))
            
            plt.close('all')
        
    print('DONE.')
                