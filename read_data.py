#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:41:10 2019

Exploring the root file containing the data.

@author: misiak
"""

import uproot
import matplotlib.pyplot as plt

run_name = 'tg09l000'
detector_name = 'RED80'

DATA_DIR = (
        '/home/misiak/Data/data_run57/{0}/{1}/'
).format(run_name, detector_name)

def file_path(num):
    
    file_name = (
            "ProcessedData_{0}_S{1:02d}_{2}_ChanTrig0.root"
    ).format(run_name, int(num), detector_name)
    
    file_path = DATA_DIR + file_name

    return file_path


if __name__ == '__main__':
    NUM = 0
    
    fp = file_path(NUM) 
    
    root = uproot.open(fp)
    
    tree_run = root["RunTree_Normal"]
    tree_event_trig_raw = root["EventTree_trig_Normal_raw"]
    tree_event_trig_filt = root["EventTree_trig_Normal_filt"]
    tree_event_trig_filt_decor = root["EventTree_trig_Normal_filt_decor"]
    tree_event_noise_raw = root["EventTree_noise_Normal_raw"]
    tree_event_noise_filt = root["EventTree_noise_Normal_filt"]
    tree_event_noise_filt_decor = root["EventTree_noise_Normal_filt_decor"]
    
    tree = tree_event_trig_raw
    #tree = tree_run
    
    energy_of = tree['Energy_OF'].array()
    energy_of_h = tree['Energy_OF_h'].array()
    energy_of_i = tree['Energy_OF_i'].array()
    energy_of_t0 = tree['Energy_OF_t0'].array()
    
    chi2_of = tree['chi2_OF'].array()
    chi2_of_h = tree['chi2_OF_h'].array()
    chi2_of_i = tree['chi2_OF_i'].array()
    
    plt.close('all')
    
    def ploty(energy, chi2):
        
        plt.figure()
        for x_data, y_data in zip(energy.T, chi2.T):
            plt.loglog(x_data, y_data, ls='none', marker='.', alpha=0.1)
        
        plt.ylabel('Chi2')
        plt.xlabel('Energy [ADU]')
        plt.grid(True)
        
    ploty(energy_of, chi2_of)
    ploty(energy_of_h, chi2_of_h)
    ploty(energy_of_i, chi2_of_i)
