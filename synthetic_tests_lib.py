import subprocess
from shlex import split
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import randint
import shutil, os, glob, sys
import time
from obspy.core import Stream
import obspy
from tqdm import tqdm

dist2deg = 1 / 111.1
from dtaidistance import dtw
from scipy import stats

def plot_seismogram(stn, out_suffix='orig',azm=None,dist=None):
    st = Stream()
    st += read(f"{stn}_{out_suffix}.r")
    st += read(f"{stn}_{out_suffix}.t")
    st += read(f"{stn}_{out_suffix}.z")

    ylim_maxs = []
    fig, ax = plt.subplots(len(st),1,figsize=(8,6), sharex=True)
    for ii, tr in enumerate(st):
        tt = tr.times()
        dd = tr.data
        ylim_maxs.append(np.max(np.abs(dd[np.logical_not(np.isnan(dd))])))
        ax[ii].plot(tt, dd, color=f'C{ii}',lw=0.5)
    ylabels=['R','T','Z']
    for ii in range(len(st)):
        # ax[ii].set_ylim([-ylim, ylim])
        ax[ii].set_ylabel(ylabels[ii],fontsize=18)
    if azm and dist:
        ax[0].set_title(f'Azm: {azm}, Dist: {dist}')

    plt.savefig(f'{stn}_plot.png',bbox_inches='tight',dpi=300)
    plt.close('all')

meta_info = {'depth' : 49.7, 'dist_val' : 200, 'mag' : 5.3, 'st' : 161, 'dp' : 40, 'rk' : 81, 'sd' : 1.1, 'az' : 3.99, 'NFFT': 2**8, 'dt': 1, 'dk': 0.9, 'source_type': 2}
def compute_synthetics(stn='STN',model_name='earth_model_orig.txt',meta_info=meta_info, calc_green_func= True, calc_syn= True, plot_seism=True, out_suffix='orig', outerr=True):
    depth = meta_info['depth']
    dist_val = meta_info['dist_val']

    ## EQ info
    mag = meta_info['mag']
    st = meta_info['st']
    dp = meta_info['dp']
    rk = meta_info['rk']
    sd = meta_info['sd']

    az = meta_info['az']

    NFFT= meta_info['NFFT'] #512 #2**13  #512 # number of points #2**17 for over a day
    dt=meta_info['dt']  #sampling interval

    dk = meta_info['dk'] #non-dimensional sampling interval of wavenumber

    source_type = meta_info['source_type']

    if calc_green_func:
        command=split("fk.pl -M"+model_name+"/"+str(depth)+"/k -N"+str(NFFT)+"/"+str(dt)+'/1/'+repr(dk)+f' -S{source_type}'+" "+str(dist_val))
        p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        out,err=p.communicate()
        if outerr:
            error = str(err, 'utf-8')
            print(error)


    if calc_syn:
        if glob.glob(model_name+"_"+str(depth)+f"/{dist_val}.grn.0"):
            earthquake_source_azm = f"-D{sd:.2f}"+" "+f"-A{az:.2f}" + " "

            output_sac_file = f"{stn}_{out_suffix}.z"
            first_comp_gf = model_name+"_"+str(depth)+f"/{dist_val}.grn.0"

            earthquake_source = f"-M{mag:.2f}/{st:.2f}/{dp:.2f}/{rk:.2f} "
            comp_syn=split("syn "+ earthquake_source +earthquake_source_azm + f"-O{output_sac_file}"+ " "+"-G"+first_comp_gf)


            p=subprocess.Popen(comp_syn,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            out,err=p.communicate()
            if outerr:
                error = str(err, 'utf-8')
                print(error)

    junks = glob.glob('junk.*')
    for jnk in junks:
        os.remove(jnk)
    if os.path.exists('floatArray.txt'):
        os.remove('floatArray.txt')

    

    if os.path.exists(f"{stn}_{out_suffix}.z"):
        if plot_seism:
            plot_seismogram(stn, out_suffix=out_suffix, azm=az, dist=dist_val)
        return True
    else:
        return False



def create_dir(direc):
    '''
    Create a directory
    '''
    try:
        os.makedirs(direc, exist_ok=True)
    except OSError:
        print("-> Creation of the directory {} failed".format(direc))
    else:
        print("-> Successfully created the directory {}".format(direc))

def rem_dir(direc):
    '''
    Delete a directory
    '''
    if os.path.exists(direc):
        shutil.rmtree(direc)


def norm_xcorr(a, b, max_lag=10):
    '''
    Adds normalization to the NumPy cross correlation function and returns the maximum within
    a time lag limit
    
    Returns the maximum correlation
    '''
    
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    xcorr = np.correlate(a, b, 'full')
    
    i = len(a)-1                       # Zeroth lag
    xcorr = xcorr[i-max_lag:i+max_lag]
    return np.max(xcorr)


def calc_dtw_dist(origData,synthData):
    s1, s2 = origData, synthData
    s1, s2 = stats.zscore(s1), stats.zscore(s2)

    s1 = s1.astype(np.double)
    s2 = s2.astype(np.double)

    dist = dtw.distance_fast(s1, s2) #uncontrained
    return dist


# f"{stations[0]}_orig"
def gen_noise(length, mean, sigma, snr=0.2):
    noise = snr * np.random.normal(mean,sigma,length)
    return noise


def read_waveforms(fileprefix, snr = None):
    orig1_r = read(fileprefix+".r")
    orig1_r = orig1_r[0].data 
    orig1_t = read(fileprefix+".t")
    orig1_t = orig1_t[0].data
    orig1_z = read(fileprefix+".z")
    orig1_times = orig1_z[0].times()
    orig1_z = orig1_z[0].data
    if snr:
        orig1_r = orig1_r + gen_noise(orig1_r.shape[0], orig1_r.mean(), orig1_r.std(), snr=snr)
        orig1_t = orig1_t + gen_noise(orig1_t.shape[0], orig1_t.mean(), orig1_t.std(), snr=snr)
        orig1_z = orig1_z + gen_noise(orig1_z.shape[0], orig1_z.mean(), orig1_z.std(), snr=snr)

    return orig1_times, orig1_r, orig1_t, orig1_z


def plot_array(data_list, out_suffix='orig',azm=None,dist=None):
    fig, ax = plt.subplots(3,1,figsize=(8,6), sharex=True)
    for ii, tr in enumerate(data_list):
        ax[ii].plot(tr, color=f'C{ii}',lw=0.5)
    ylabels=['R','T','Z']
    for ii in range(3):
        # ax[ii].set_ylim([-ylim, ylim])
        ax[ii].set_ylabel(ylabels[ii],fontsize=18)

    plt.savefig(f'plot_{out_suffix}.png',bbox_inches='tight',dpi=300)
    plt.close('all')

from sklearn import preprocessing
from scipy import signal
min_max_scaler11 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
def warping_func_point_matches(s1,s2,window=None,detrend=False,psi=2):
    if detrend:
        s1,s2 = signal.detrend(s1),signal.detrend(s2)

    s1 = min_max_scaler11.fit_transform((s1).reshape(-1, 1))
    s2 = min_max_scaler11.fit_transform((s2).reshape(-1, 1))

    if not window:
        window = len(s1)#int(0.90*len(s1))

    d, paths = dtw.warping_paths(s1, s2, window=window, psi=psi)
    p = dtw.best_path(paths)
    py, px = zip(*p)
    py, px = np.array(py), np.array(px)
    

    # norm_fact = np.sqrt(len(s1)**2+len(s2)**2)
    # norm_d = d/norm_fact
    # norm_d = norm_d*1000
    return p, paths, px, py, d

def calc_distance(s1,s2,detrend=False, window=None):
    if detrend:
        s1,s2 = signal.detrend(s1),signal.detrend(s2)

    s1 = min_max_scaler11.fit_transform((s1).reshape(-1, 1))
    s2 = min_max_scaler11.fit_transform((s2).reshape(-1, 1))
    s1, s2 = s1.reshape(-1),s2.reshape(-1)

    s1 = s1.astype(np.double)
    s2 = s2.astype(np.double)

    distance = dtw.distance_fast(s1, s2, window=window) #uncontrained
    # norm_fact = np.sqrt(len(s1)**2+len(s2)**2)
    # distance = distance/norm_fact
    
    return distance
min_max_scaler01 = preprocessing.MinMaxScaler(feature_range=(0, 1))
def plot_figure(orig_times, series_x, series_y,orig_x,phase_df, distance,center_x, outfigprefix,ylimX,ylimY,vline=None,dashed=True,metric='dissimilarity'):

    fig, ax = plt.subplots(3,1,figsize=(8,6), sharex=True)
    ax[0].plot(orig_times, orig_x, '--',color='k',lw=0.5)
    ax[0].plot(orig_times, series_x, '-',color='C0')

    ax[0].set_ylabel('Series X',fontsize=10)
    # ax[0].yaxis.set_label_coords(-0.1,0.5)
    ax[0].set_ylim([-ylimX,ylimX])
    if vline:
        if dashed:
            ax[0].axvline(x=vline,lw=0.8,color='b',ls='--')
        else:
            ax[0].axvline(x=vline,lw=2,color='r')
    if metric=='dissimilarity':
        ax[2].plot(phase_df['xvals'],phase_df['dist_vals'],color='C2',lw=1,ls='-')
        if vline:
            if dashed:
                ax[2].axvline(x=vline,lw=0.8,color='b',ls='--')
            else:
                ax[2].axvline(x=vline,lw=2,color='r')
        ax[2].scatter(center_x,distance,color='r',s=20,label="Dissimilarity: {:.2f}".format(distance))
        ax[2].set_ylabel('Dissimilarity',fontsize=10)
        ax[2].legend(loc=2, fontsize=10)
        ax[2].set_ylim([-phase_df['dist_vals'].min()+0.01*phase_df['dist_vals'].min(),phase_df['dist_vals'].max()])
    elif metric=='similarity':
        sim_array = np.array(min_max_scaler01.fit_transform((1/phase_df['dist_vals'].values).reshape(-1, 1)))
        ax[2].plot(phase_df['xvals'],sim_array,color='C2',lw=1,ls='-')
        if vline:
            if dashed:
                ax[2].axvline(x=vline,lw=0.8,color='b',ls='--')
            else:
                ax[2].axvline(x=vline,lw=2,color='r')
        sim_loc = phase_df['dist_vals'].index[phase_df['dist_vals'].isin([distance])][0]
        ax[2].scatter(center_x,sim_array[sim_loc],color='r',s=20,label=f"Similarity: {sim_array[sim_loc][0]:.2f}")
        ax[2].set_ylabel('Similarity',fontsize=10)
        ax[2].legend(loc=2, fontsize=10)
        ax[2].set_ylim([0,1.02])


    ax[1].plot(orig_times, series_y, '-',color='C1')
    ax[1].set_ylabel('Series Y',fontsize=10)
    # ylim2 = np.max(np.abs(series_y))
    ax[1].set_ylim([-ylimY,ylimY])
    # ax[1].yaxis.set_label_coords(-0.1,0.5)
    plt.subplots_adjust(hspace=0.1)
    fig.align_ylabels()
    plt.savefig(f'{outfigprefix}.png',bbox_inches='tight')
    plt.close('all')