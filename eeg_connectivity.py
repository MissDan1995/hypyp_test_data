# In[0]
"""
Spyder Editor

This is a temporary script file.
"""

import os
import mne
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
import io
from copy import copy
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import scipy
import requests
from hypyp import prep  
from hypyp import analyses
from hypyp import stats
from hypyp import viz

# Setting parameters

# Frequency bands used in the study
freq_bands = {
    "Theta": [4, 7],
    "Alpha-Low": [7.5, 11],
    "Alpha-High": [11.5, 13],
    "Beta": [13.5, 29.5],
    "Gamma": [30, 48],
}
freq_bands = OrderedDict(freq_bands)  # Force to keep order

# Specify sampling frequency
sampling_rate = 1000  # Hz

#convert .set format to .fif format
# In[1]
subjects=0
allsub = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26'];
for subjects in allsub:
    data_path='C:\\wangdan\\study 2\\EC_ICA\\'+subjects+' post_ica_1.set'
    data=mne.io.read_epochs_eeglab(data_path)
    data_path_save = 'C:\\wangdan\\study 2\\EC_ICA\\'+subjects+' post_ica_1-epo.fif'
    data.save(data_path_save,overwrite=True) 
##########################################################
# In[2]
#Loading data files

epochs1 = mne.read_epochs(os.path.join(r'C:\wangdan\study 2\EC_ICA','01 post_ica_1-epo.fif'), preload=True,)

epochs2 = mne.read_epochs(os.path.join(r'C:\wangdan\study 2\EC_ICA','02 post_ica_1-epo.fif'), preload=True,)
print(epochs1,epochs2.info)

#filter
epochs1=epochs1.filter(l_freq=0.5, h_freq=60)
epochs2=epochs2.filter(l_freq=0.5, h_freq=60)
# In[3]
# Connectivity
#initializing data and storage
data_inter=np.array([epochs1,epochs2])
result_intra=[]

#computing analytic signal per frequency band
complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)
# computing frequency- and time-frequency-domain connectivity
'''**supported connectivity measures**
     - 'envelope_corr': envelope correlation
     - 'pow_corr': power correlation
     - 'plv': phase locking value
     - 'ccorr': circular correlation coefficient
     - 'coh': coherence
     - 'imaginary_coh': imaginary coherence
     '''
result = analyses.compute_sync(complex_signal, mode="ccorr")
# slicing results to get the Inter-brain part of the matrix
n_ch = len(epochs1.info["ch_names"])
theta, alpha_low, alpha_high, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]
# choosing Alpha_Low for futher analyses for example
values = alpha_low
values -= np.diag(np.diag(values))
# computing Cohens'D for further analyses for example
C = (values - np.mean(values[:])) / np.std(values[:])

# slicing results to get the Intra-brain part of the matrix
for i in [0, 1]:
    theta, alpha_low, alpha_high, beta, gamma = result[:, i : i + n_ch, i : i + n_ch]
    # choosing Alpha_Low for futher analyses for example
    values_intra = alpha_low
    values_intra -= np.diag(np.diag(values_intra))
    # computing Cohens'D for further analyses for example
    C_intra = (values_intra - np.mean(values_intra[:])) / np.std(values_intra[:])
    # can also sample CSD values directly for statistical analyses
    result_intra.append(C_intra)
# In[4]
# Statistical anlyses
# Comparing Inter-brain connectivity values to random signal

# No a priori connectivity between channels is considered
# between the two participants
# in Alpha_Low band for example (see above)
# consitute two artificial groups with 2 'participant1' and 2 'participant2'
data = [np.array([values, values]), np.array([result_intra[0], result_intra[0]])]

statscondCluster = stats.statscondCluster(
    data=data,
    freqs_mean=np.arange(7.5, 11),
    ch_con_freq=None,
    tail=0,
    n_permutations=5000,
    alpha=0.05,
)
# In[5]
# Visualization
# Visualization of inter-brain connectivity in 3D
# Visualization of inter-brain connectivity in 2D
viz.viz_2D_topomap_inter(epochs1, epochs2, C, threshold='auto', steps=10, lab=True)
viz.viz_3D_inter(epochs1, epochs2, C, threshold='auto', steps=10, lab=False)
