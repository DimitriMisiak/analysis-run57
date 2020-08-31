#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:30:13 2019

@author: misiak
"""

import numpy as np
import matplotlib.pyplot as plt
import uproot

from core_classes import Tree

###
#plt.close('all')

#file_path = '/home/misiak/Data/data_run57/tg21l000/RED70/RootFiles/TriggerData_tg21l000_S02_RED70_ChanTrig0.root'
#file_path = '/home/misiak/Data/data_run57/tg21l000/RED70/RootFiles/NoiseData_tg21l000_S02_RED70_ChanTrig0.root'

detector = 'RED7'

#runname = 'tg31l003'
#file_path = '/home/misiak/Data/data_run57/{0}/{1}0/RootFiles/TriggerData_{0}_S00_{1}0_ChanTrig0.root'.format(runname, detector)

runname = 'ti04l001'
file_path = '/home/misiak/Data/data_run59/{0}/{1}1/RootFiles/TriggerData_{0}_S00_{1}1_ChanTrig0.root'.format(runname, detector)


root = uproot.open(file_path)

tree = Tree(root, 'tree')

tree.samples = tree.Event_Number.shape[0]

#we need some cut
cut1 = np.all(np.abs(tree.Trace_Heat_A_Raw) < 10000, axis=1)

cut2 = abs(tree.Trace_Heat_A_Raw[:, 0] - tree.Trace_Heat_A_Raw[:, -1]) < 1000

cut3 = np.max(tree.Trace_Heat_A_Raw,axis=1) - 2*np.mean(tree.Trace_Heat_A_Raw, axis=1) > - np.min(tree.Trace_Heat_A_Raw,axis=1)
cut = np.logical_and(cut1, cut2)
cut = np.logical_and(cut, cut3)

trace_heat = tree.Trace_Heat_A_Raw.T[:, cut]
trace_ion = tree.Trace_Ion_B_Raw.T[:, cut]

#Trace Heat Raw
plt.figure('Heat Raw')
#plt.plot(trace_heat, alpha=0.1)

A = np.mean(trace_heat, axis=1)
plt.plot(A - np.mean(A[:90]), alpha=1)
#plt.plot(trace_heat[:,61], alpha=1)
#plt.plot(trace_heat[:,50], alpha=1)

##Trace Ion A Raw
#plt.figure('Ion A Raw')
#plt.plot(trace_ion, alpha=0.1)
#
##Trace OF, meh this is the OF filter
#plt.figure('trace OF')
#plt.plot(tree.Trace_OF.T, alpha=0.1)



