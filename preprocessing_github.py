# -*- coding: utf-8 -*-
"""
Example EEG Preprocessing script
with automatized ICA blink component selection
and automatized trial rejection
A HTML report for quality checks is made for each subject

@author: Luc Vermeylen
"""

#%% MODULES

import mne # i used version 0.24
import numpy as np
import pandas as pd
import os
import autoreject
import time
from datetime import datetime

# %% FLAGS & VARIABLES

subjects = [1] # selection array, e.g. [1,3,5] or range(1,31)

# create prime or target locked epochs?
event_type = "prime" # prime or response locked

# flags
re_epoch = True
re_fit_ica = True
re_apply_ica = True
re_fit_local_reject = True
re_apply_local_reject = True

# epoch settings
highpass_freq = .1 # for ERPs only: is "None" for time frequency (TF) and MVPA
lowpass_freq = 40
if event_type == 'prime':
    epoch_tmin = -1
    epoch_tmax = 1.5
elif event_type == 'response':
    epoch_tmin = -1.5
    epoch_tmax = 1
desired_sfreq = 250 # after decimation (better than resampling, see MNE's recommendations)
fixed_rej_crit = 150e-6
n_autoreject = int(4680/1) # how many trials to use for estimation of autoreject

# ica settings
ica_epoch_tmin = -.5 # note: is also the reject_min and reject_max (time frame in which rejection criteria apply)
ica_epoch_tmax = 1
ica_highpass = 1

n_random_plots = 10

start_time = datetime.now()

#%% BAD ELECTRODES

bad_electrodes = [
    ["T8"],
    [],
    [],
    ["C2","FC6"],
    [], # 5
    [],
    ["AF7","Fp1","P8","P7"], # 7: a bit of a noisy subject, but ok
    [],
    ["F4"], # 9: O1, O2, Oz are quite noisy
    [], # 10: has two blink components in the ICA
    [],
    ["Pz","F7","FC1"], # 12
    ["AF8","F6","AF3"], # 13: a bit more noisy, but seems ok
    ["Pz"], # 14
    ["AF4","F4","AF8","AF7"], # 15: noisy subject 7.4 % rejected (fixed threshold)
    [],
    [], # 17: weird spikes in the EEG
    ["FT10", "FC4"], # 18
    ["TP8"], # 19
    [], # 20
    ["PO4","FC2","TP8"], # 21
    ["CPz","TP8","AF4"], # 22
    [],
    [],
    [], # 25
    [],
    ["O2","CPz","TP8","AF4"], # 27
    [],
    [],
    [], # 30
    [],
    ['C4'],
    ['P4'],
    [],
    ['F2','CP2','AF4'], # 35
    [],
    ['FT10'],
    [],
    ['AF8','F7','AF7','FT9'],
    [], # 40
    [],
    ["P1"],
    [],
    ["T7","FT8"],
    [], # 45
    ["T8","F1"],
    [],
    ["PO8","O2","P7"],
    []
    ]

lengths = []
for l in bad_electrodes:
    lengths.append(len(l))
pd_lengths = pd.Series(lengths)
print(pd_lengths.value_counts())

#%% BAD ICA COMPONENTS

# in addition to the automatic selection of the EOG component below
bad_ica = [
    [],[],[],[],[], # 5
    [],[],[],[],[1], # 10 (sub 10 has two blink components!)
    [],[],[],[],[], # 15
    [],[1],[],[],[], # 20 (sub 17 has an ECG component [1]! works well to remove the spikes present in the EEG.)
    [],[],[],[],[], # 25
    [],[],[],[],[], # 30
    [1],[],[],[],[], # 35 (sub 31 has two blink components)
    [],[],[],[],[], # 40
    [],[],[1],[],[], # 45
    [],[],[],[]
    ]

#%% SUBJECT LOOP

for subj in subjects:

    startTime = time.time()

    #%% DATA PATHS

    base_path = 'E:/EEG/data/'
    raw_path = base_path + 'raw/'
    derived_path = base_path + 'derived/'
    metadata_path = base_path + 'behavior/'
    derived_folder = base_path + 'derived/sub-' + str(subj).zfill(2) + '/'
    sub_number = str(subj).zfill(2)
    sub_name = 'sub_' + str(subj).zfill(2)
    if not os.path.exists(derived_folder):
        os.makedirs(derived_folder)

    #%% IMPORT RAW

    if re_epoch:
        raw = mne.io.read_raw_brainvision(raw_path + sub_name + '.vhdr', preload=True)
    else:
        raw = mne.io.read_raw_brainvision(raw_path + sub_name + '.vhdr', preload=False)
    montage = mne.channels.read_custom_montage(base_path + 'channel_layout/AP-64.bvef')
    raw.set_montage(montage)
    original_sfreq = raw.info['sfreq'] # get original sampling frequency for later calculations
    raw.info['bads'] = bad_electrodes[subj-1] # select the bad electrodes

    #%% MNE REPORT

    # Open the MNE report (saves figures and info for quality check)
    report = mne.Report(info_fname=raw, # Name of the file containing the info dictionary.
                        subject=sub_number,
                        title='Preprocessing Report for "'+event_type+'-locked events" (Subject '+sub_number+')',
                        baseline=None,
                        image_format='png',
                        raw_psd=False, # If True, include PSD plots for raw files.
                        projs=True # Whether to include topographic plots of SSP projectors, if present in the data.
                        )

    # adding the butterfly takes too much time, so leave out for now...
    report.add_raw(raw=raw, title='Raw Info', psd=False, butterfly=False, tags=['Raw'])

    #%% FILTERING

    # MNE recommendation regarding filtering and resampling:
    # 1) low-pass filter the Raw data at or below 1/3 of the desired sample rate
    # 2) decimate the data after epoching, by either passing the decim parameter to the Epochs constructor, or using the decimate() method after the Epochs have been created.

    # Bandpass firwin filter .1 - 40 Hz (no highpass for TF/MVPA). For the ICA data, MNE recommends high pass at 1 Hz
    if re_epoch:
        filt = raw.copy().filter(highpass_freq, lowpass_freq, n_jobs=4, fir_design='firwin') # version for ERP
        filt_TF = raw.copy().filter(None, lowpass_freq, n_jobs=4, fir_design='firwin') # version for TF/decoding
        filt_1hz = raw.copy().filter(ica_highpass, lowpass_freq, n_jobs=4, fir_design='firwin') # version for ICA
    del raw

    #%% EVENTS
    ###############################################################################################################
    # READING AND NAMING EVENTS
    ###############################################################################################################

    if re_epoch:
        # read events from the filtered dataset
        events, event_id = mne.events_from_annotations(filt)

        # make new event dictionary
        if event_type == 'prime':
            event_dict = {
                'PRIME/CO/L/C': 20, 'PRIME/CO/R/C': 21, 'PRIME/IC/R/C': 22, 'PRIME/IC/L/C': 23, # note! 22 a 23 now denote target direction, not flanker direction!
                'PRIME/CO/L/E': 24, 'PRIME/CO/R/E': 25, 'PRIME/IC/R/E': 26, 'PRIME/IC/L/E': 27, # for errors: +4
                'PRIME/NR': 28,
            }
        elif event_type == 'response':
            event_dict = {
                'RESP/CO/L/C': 40, 'RESP/CO/R/E': 41, 'RESP/IC/L/C': 42, 'RESP/IC/R/E': 43,
                'RESP/CO/R/C': 44, 'RESP/CO/L/E': 45, 'RESP/IC/R/C': 46, 'RESP/IC/L/E': 47,
                'RESP/NR': 48,
            }

        # rename the prime events so they include information about accuracy (but you could also use the metadata for this)
        for i in range(events.shape[0]):
            if events[i, 2] in [41, 43, 45, 47]:  # if the trial is an error, add four to the event ID
                events[i-1, 2] = events[i-1, 2] + 4
                events[i-2, 2] = events[i-2, 2] + 4
            elif events[i, 2] == 48: # if the trial didn't have a response, its code is 28
                events[i-2, 2] = 28

        # limit events to either primes or targets only
        if event_type == 'prime':
            events = events[events[:,2] > 19,:]
            events = events[events[:,2] < 30,:]
        elif event_type == 'response':
            events = events[events[:,2] > 39,:]
            events = events[events[:,2] < 50,:]

    #%% EPOCHING

    decim = np.round(original_sfreq / desired_sfreq).astype(int) # decimate to reach desired sfreq (rather than resample)

    if re_epoch:
        meta = pd.read_csv(metadata_path +'FT_' + str(subj).zfill(3) + '_0.csv')
        meta = meta[meta['phase'] == 'flanker']

        epochs = mne.Epochs(filt, # epoch the filtered signal
                            events,
                            event_id=event_dict, # using only the events specified in the dict above
                            tmin=epoch_tmin, tmax=epoch_tmax, # watch out with time-frequency analyses!
                            baseline=None, # a tuple of length 2: (a, b). None = beginning or end: (None, None) = from beginning to end.
                            picks=None, # all channels. Note: channels in info['bads'] are automatically removed
                            preload=True,
                            reject=None, # reject epochs based on maximum peak-to-peak signal amplitude
                            flat=None, # reject epochs based on minimum peak-to-peak signal amplitude (PTP).
                            proj=False, # no projections applied (e.g., reference projection)
                            decim=decim, # Low-pass filtering is not performed (as would be with resample), this simply selects every Nth sample (where N is the value passed to decim), i.e., it compresses the signal (see Notes). If the data are not properly filtered, aliasing artifacts may occur.
                            reject_tmin=ica_epoch_tmin, reject_tmax=ica_epoch_tmax, # min/max for reject/flat criteria
                            detrend=None, # This parameter controls the time period used in conjunction with both, reject and flat.
                            on_missing='warn', #What to do if one or several event ids are not found in the recording.
                            reject_by_annotation=True, # If True (default), epochs overlapping with segments whose description begins with 'bad' are rejected. If False, no rejection based on annotations is performed.
                            metadata=meta, # len(metadata) must equal len(events). The DataFrame may only contain values of type (str | int | float | bool). If metadata is given, then pandas-style queries may be used to select subsets of data
                            event_repeated='error', # How to handle duplicates in events[:, 0].
                            verbose=None
                            )

        epochs_TF = mne.Epochs(filt_TF, # epoch the filtered signal
                            events,
                            event_id=event_dict, # using only the events specified in the dict above
                            tmin=epoch_tmin, tmax=epoch_tmax, # watch out with time-frequency analyses!
                            baseline=None, # a tuple of length 2: (a, b). None = beginning or end: (None, None) = from beginning to end.
                            picks=None, # all channels. Note: channels in info['bads'] are automatically removed
                            preload=True,
                            reject=None, # reject epochs based on maximum peak-to-peak signal amplitude
                            flat=None, # reject epochs based on minimum peak-to-peak signal amplitude (PTP).
                            proj=False, # no projections applied (e.g., reference projection)
                            decim=decim, # Low-pass filtering is not performed (as would be with resample), this simply selects every Nth sample (where N is the value passed to decim), i.e., it compresses the signal (see Notes). If the data are not properly filtered, aliasing artifacts may occur.
                            reject_tmin=ica_epoch_tmin, reject_tmax=ica_epoch_tmax, # min/max for reject/flat criteria
                            detrend=None, # This parameter controls the time period used in conjunction with both, reject and flat.
                            on_missing='warn', #What to do if one or several event ids are not found in the recording.
                            reject_by_annotation=True, # If True (default), epochs overlapping with segments whose description begins with 'bad' are rejected. If False, no rejection based on annotations is performed.
                            metadata=meta, # len(metadata) must equal len(events). The DataFrame may only contain values of type (str | int | float | bool). If metadata is given, then pandas-style queries may be used to select subsets of data
                            event_repeated='error', # How to handle duplicates in events[:, 0].
                            verbose=None
                            )

        epochs_1hz = mne.Epochs(filt_1hz, # epoch the filtered signal
                            events,
                            event_id=event_dict, # using only the events specified in the dict above
                            tmin=ica_epoch_tmin, tmax=ica_epoch_tmax, # For ICA, don't have too much overlap if you do the ICA on epochs! perhaps [-.5, .7] is better?
                            baseline=None, # a tuple of length 2: (a, b). None = beginning or end: (None, None) = from beginning to end.
                            picks=None, # all channels. Note: channels in info['bads'] are automatically removed
                            preload=True,
                            reject=None, # reject epochs based on maximum peak-to-peak signal amplitude
                            flat=None, # reject epochs based on minimum peak-to-peak signal amplitude (PTP).
                            proj=False, # no projections applied (e.g., reference projection)
                            decim=decim, # Low-pass filtering is not performed (as would be with resample), this simply selects every Nth sample (where N is the value passed to decim), i.e., it compresses the signal (see Notes). If the data are not properly filtered, aliasing artifacts may occur.
                            reject_tmin=ica_epoch_tmin, reject_tmax=ica_epoch_tmax, # min/max for reject/flat criteria
                            detrend=None, # This parameter controls the time period used in conjunction with both, reject and flat.
                            on_missing='warn', #What to do if one or several event ids are not found in the recording.
                            reject_by_annotation=True, # If True (default), epochs overlapping with segments whose description begins with 'bad' are rejected. If False, no rejection based on annotations is performed.
                            metadata=meta, # len(metadata) must equal len(events). The DataFrame may only contain values of type (str | int | float | bool). If metadata is given, then pandas-style queries may be used to select subsets of data
                            event_repeated='error', # How to handle duplicates in events[:, 0].
                            verbose=None
                            )

        report.add_epochs(epochs=epochs, psd=True, title='Raw Epochs', tags=["Epochs"])

        # save raw epochs
        epochs.save(fname=derived_folder + sub_name + '_' + event_type + '-raw_ERP-epo.fif',overwrite=True)
        epochs_TF.save(fname=derived_folder + sub_name + '_' + event_type + '-raw_TF-epo.fif',overwrite=True)
        epochs_1hz.save(fname=derived_folder + sub_name + '_' + event_type + '-raw_ICA-epo.fif',overwrite=True)
        del filt, filt_1hz, filt_TF
    else:
        epochs = mne.read_epochs(derived_folder + sub_name + '_' + event_type + '-raw_ERP-epo.fif', preload=True)
        epochs_TF = mne.read_epochs(derived_folder + sub_name + '_' + event_type + '-raw_TF-epo.fif', preload=True)
        epochs_1hz = mne.read_epochs(derived_folder + sub_name + '_' + event_type + '-raw_ICA-epo.fif', preload=True)

    #%% CHECK SFREQ

    # sampling frequency information after low pass and decimation
    obtained_sfreq = original_sfreq / decim
    lowpass_freq_limit = obtained_sfreq / 3. # WATCH OUT! low pass should be at OR below this frequency!
    print('original sampling frequency was {} Hz;'
          'desired sampling frequency was {} Hz; decim factor of {} yielded an '
          'actual sampling frequency of {} Hz.;'
          'low pass frequency recommended (by MNE) to be at or below {} Hz'
          .format(original_sfreq, desired_sfreq, decim, epochs.info['sfreq'], lowpass_freq_limit))

    #%% PLOT EPOCHS

    random_indexes = np.sort(np.random.randint(0, len(epochs.events), size = 40 * n_random_plots))
    epoch_figs = []
    for i in range(0,n_random_plots):
        random_epochs = epochs[random_indexes[(i*40):((i*40)+40)]].plot(n_epochs=40, n_channels=63, decim=2, scalings=dict(eeg=50e-6))
        epoch_figs.append(random_epochs)
    report.add_figure(epoch_figs, title="Inspect Raw Epochs", tags=["Epochs"])

    #%% ICA

    # estimate the ICA on the highly filtered (1 Hz) epoched data (99% of variance)
    if re_fit_ica:
        ica = mne.preprocessing.ICA(n_components=0.999999)
        ica.fit(epochs_1hz, decim=2)
        ica.save(fname=derived_folder + sub_name + '-auto-ica.fif', overwrite=True)
    else:
        ica = mne.preprocessing.read_ica(fname=derived_folder + sub_name + '-auto-ica.fif')

    #%% BLINK SELECTION

    # Find EOG components by creating EOG epochs from Fp1 and Fp2 and calculating correlation between component and EOG.
    eog_inds, scores = ica.find_bads_eog(epochs_1hz, ch_name=['Fp1','Fp2'])
    #ica.plot_scores(scores, exclude=eog_inds) # returns scores for both electrodes

    # auto selected components
    print()
    print("AUTOMATICALLY MARKED EOG ICA COMPONENTS")
    print(eog_inds)

    # get maximum component
    max_score = np.max(np.abs(scores[0][0:20]))
    max_index = [np.argmax(np.abs(scores[0][0:20]))]
    print()
    print("MAXIMAL EOG ICA COMPONENTS")
    print(max_index)

    # manually selected components
    print()
    print("MANUALLY SELECTED ICA COMPONENTS")
    print(bad_ica[subj-1])

    # add other known components (or if automatic version fails)
    if subj == 10 or subj == 17 or subj == 31 or subj == 43:
        ica.exclude = [int(max_index[0]), int(bad_ica[subj-1][0])]
    else:
        ica.exclude = max_index
    print()
    print("FINAL EXCLUDED ICA COMPONENTS")
    print(ica.exclude)

    #%% ICA REPORT

    report.add_ica(ica=ica,title='ICA cleaning',picks=range(20),inst=None,eog_scores=scores,n_jobs=4)

    # apply the ICA
    if re_apply_ica:
        epochs_ERP_ica = ica.apply(epochs.copy())
        epochs_TF_ica = ica.apply(epochs_TF.copy())
        # save ica corrected epochs
        epochs_ERP_ica.save(fname=derived_folder + sub_name + '_' + event_type + '-raw-ica-ERP-epo.fif',overwrite=True)
        epochs_TF_ica.save(fname=derived_folder + sub_name + '_' + event_type + '-raw-ica-TF-epo.fif',overwrite=True)
    else:
        epochs_ERP_ica = mne.read_epochs(derived_folder + sub_name + '_' + event_type + '-raw-ica-ERP-epo.fif', preload=True)
        epochs_TF_ica = mne.read_epochs(derived_folder + sub_name + '_' + event_type + '-raw-ica-TF-epo.fif', preload=True)

    #%% REJECTION

    # autoreject (local): find the optimal threshold for each electrode separately
    if re_apply_local_reject:
        # estimate autoreject (on the cropped ERP data, which is high-pass filtered, as autoreject does not seem to work well on the data that is not high-pass filtered)
        if re_fit_local_reject:
            ar_ERP = autoreject.AutoReject(n_jobs=4) # instantiate the autoreject object (after ICA (but remove bad channels before)... because otherwise this seems to be getting the eye blinks rather than other artifacts !!!)
            random_indexes = np.sort(np.random.randint(0, len(epochs_ERP_ica.events), size = n_autoreject))
            #ar_ERP.fit(epochs_ERP_ica.copy().crop(tmin=-.5,tmax=1).decimate(2)[random_indexes]) # note: you can fit the threshold model on a part of the data, to make go faster
            ar_ERP.fit(epochs_ERP_ica.copy().decimate(2)[random_indexes]) # no crop
            ar_ERP.save(fname=derived_folder + sub_name + '_' + event_type + '-N' + str(n_autoreject) + '-ERP-autoreject.fif', overwrite=True)
        else:
            if event_type == 'prime':
                ar_ERP = autoreject.read_auto_reject(fname=derived_folder + sub_name + '_' + event_type + '-N' + str(n_autoreject) + '-ERP-autoreject.fif')

        # apply (tranform) it to the cropped (ERP) epochs (because the epochs are pretty long, and artifacts near the unimportant datapoints are not of interest...)
        # autoreject_epochs = ar_ERP.transform(epochs_ERP_ica.copy().crop(tmin=-.5,tmax=1)) # transform all the data
        if event_type == 'prime':
            autoreject_epochs = ar_ERP.transform(epochs_ERP_ica.copy()) # transform all the data
            autoreject_epochs = autoreject_epochs.drop_bad(reject=dict(eeg=fixed_rej_crit), flat=dict(eeg=1e-6)) # also apply a flat channel criteria
            autoreject_plot = autoreject_epochs.plot_drop_log() # drop log from only local method
            report.add_figure(autoreject_plot, title="After Local Rejection Criteria", tags=["Reject"])
            # get the selected epochs and limit the real ERP/TF epochs in the same way
            epochs_ERP_ica_reject = autoreject_epochs
            epochs_TF_ica_reject = epochs_TF_ica[autoreject_epochs.selection]
        elif event_type == 'response':
            prime_trial = mne.read_epochs(fname=derived_folder + sub_name + '_' + 'prime' + '-raw-ica-reject-ERP-epo.fif', preload=False)
            epochs_ERP_ica_reject = epochs_ERP_ica[prime_trial.selection]
            epochs_TF_ica_reject = epochs_TF_ica[prime_trial.selection]

    #%% INTERPOLATE

    if re_apply_local_reject:
        epochs_ERP_ica_reject.interpolate_bads()
        epochs_TF_ica_reject.interpolate_bads()
    del epochs, epochs_ERP_ica, epochs_TF_ica, epochs_1hz

    #%% REREFERENCE

    # rereference to average (add C1 back to the data) & save the data
    if re_apply_local_reject:
        # 1) ERP epochs
        epochs_ERP_ica_reject_ref = mne.add_reference_channels(epochs_ERP_ica_reject.copy(), 'C1')
        epochs_ERP_ica_reject_ref.set_montage(montage)
        epochs_ERP_ica_reject_ref.set_eeg_reference(ref_channels='average', ch_type='eeg', projection=False)  # apply rereference immediately!
        epochs_ERP_ica_reject_ref.save(fname=derived_folder + sub_name + '_' + event_type + '-raw-ica-reject-ERP-epo.fif', overwrite=True)
        # 2) TF epochs
        epochs_TF_ica_reject_ref = mne.add_reference_channels(epochs_TF_ica_reject.copy(), 'C1')
        epochs_TF_ica_reject_ref.set_montage(montage)
        epochs_TF_ica_reject_ref.set_eeg_reference(ref_channels='average', ch_type='eeg', projection=False)  # apply rereference immediately!
        epochs_TF_ica_reject_ref.save(fname=derived_folder + sub_name + '_' + event_type + '-raw-ica-reject-TF-epo.fif', overwrite=True)

    #%% PLOT EPOCHS

    if re_apply_local_reject:
        # 1) ERP epochs
        random_indexes = np.sort(np.random.randint(0, len(epochs_ERP_ica_reject_ref.events), size = 40 * n_random_plots))
        epoch_figs = []
        for i in range(0,n_random_plots):
            random_epochs = epochs_ERP_ica_reject_ref[random_indexes[(i*40):((i*40)+40)]].plot(n_epochs=40, n_channels=64, decim=2, scalings=dict(eeg=50e-6))
            epoch_figs.append(random_epochs)
        report.add_figure(epoch_figs, title="Inspect Final Epochs", tags=["Epochs"])

    #%% SAVE REPORT

    # save MNE report
    report.save(base_path + 'reports/preprocessing/sub_' + sub_number + '_' + event_type + '-locked.html', overwrite=True)
    report.save(base_path + 'reports/preprocessing/sub_' + sub_number + '_' + event_type + '-locked.hdf5', overwrite=True)

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

end_time = datetime.now()
current_time = end_time.strftime("%H:%M:%S")
print('Start time of script: ' + str(start_time))
print('End time of script: ' + str(end_time))
