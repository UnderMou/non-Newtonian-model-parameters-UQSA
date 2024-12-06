import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from foam_models import *
from opt_def import *
from foam_fit import *

class pfx:
        def __init__(self, F, X):
            self.F = F
            self.X = X

def Read_postProcfiles(filesDir = None, n_results = None, root_seed = None):
    bp = []
    if filesDir is not None:
        if n_results  is not None:
            for i in range(root_seed, n_results + 1 ):
                print("reading '%s%s' ..." % (filesDir, i))
                with open(filesDir+"%s.pkl" % (i), 'rb') as inp:
                    bp.append(pickle.load(inp))
            return bp
        else:
            raise print("'n_results ' parameter not provided")
    else: 
        raise print("'filesDir' parameter not provided.")

def get_best(bp):
    f_values = []
    for i in range(len(bp)):
        res = bp[i]
        f_values.append(res.F)

    f_values = np.array(f_values)
    print('Statistics: min = ', str(np.min(f_values)), ' | mean = ', str(np.mean(f_values)), ' | std = ', str(np.std(f_values)))
    return np.argmin(np.array(f_values))

def indexes_of_values(arr, values):
    indexes = []
    for i, val in enumerate(arr):
        if val in values:
            indexes.append(i)
    return indexes

def indexes_of_values_with_tolerance(arr, values, epsilon):
    indexes = []
    for i, val in enumerate(arr):
        for target_val in values:
            if abs(val - target_val) <= epsilon:
                indexes.append(i)
                break  # Stop checking for this value
                
    return indexes

def get_idx(fg_fit1, target_arr, tol):
    idx = indexes_of_values_with_tolerance(fg_fit1, target_arr, tol)
    # print(fg_fit1[idx], mu_app_fit1[idx]*1e3)

    idx_nonDuplicates = [idx[0]]
    if (len(target_arr) != 1):
        for i in range(1,len(idx)):
            if abs(fg_fit1[idx[i]] - fg_fit1[idx[i-1]]) > tol:
                idx_nonDuplicates.append(idx[i])

    aux = []
    for i in range(len(idx_nonDuplicates)):
        aux.append(idx_nonDuplicates[-i-1])
    
    return aux

def get_thetas(bp):
    all_thetas = []
    Fs = []
    for i in range(len(bp)):
        res = bp[i]
        theta = res.X
        # if abs(theta[4]-1e-08) <= 1e-12: print(theta)
        all_thetas.append(theta)
        Fs.append(res.F)
    return all_thetas, np.array(Fs)

def summary_of_results(all_thetas, core_params, figname, fmcap_plot = True, epcap_plot = True):
    p1 = [p[0] for p in all_thetas]
    p2 = [p[1] for p in all_thetas]
    p3 = [p[2] for p in all_thetas]
    p4 = [p[3] for p in all_thetas]
    p5 = [p[4] for p in all_thetas]

    print("mins : ", min(p1), min(p2), min(p3), min(p4), min(p5))
    print("maxs : ", max(p1), max(p2), max(p3), max(p4), max(p5))
    print("mean : ", np.mean(p1), np.mean(p2), np.mean(p3), np.mean(p4), np.mean(p5))
    print("std : ", np.std(p1), np.std(p2), np.std(p3), np.std(p4), np.std(p5))
    print("median : ",np.median(p1), np.median(p2), np.median(p3), np.median(p4), np.median(p5))
    print("\n")

    plt.rc('font', size=15)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3))   

    nbins = 60  

    axs[0].hist(p1, bins=np.linspace(min(p1),max(p1),nbins), density=False, color='darkblue', edgecolor=None)
    # axs[0].hist(p1, bins=np.linspace(10,10000,50), density=False, edgecolor=None)
    axs[0].set_title('$fmmob$')
    axs[0].set_ylabel('frequency')

    axs[1].hist(p2, bins=np.linspace(min(p2),max(p2),nbins), density=False, color='darkblue', edgecolor=None)
    # axs[1].hist(p2, bins=np.linspace(core_params[0],1.0-core_params[1],50), density=False, edgecolor=None)
    axs[1].set_title('$SF$')

    axs[2].hist(p3, bins=np.linspace(min(p3),max(p3),nbins), density=False, color='darkblue', edgecolor=None)
    # axs[2].hist(p3, bins=np.linspace(10,4000,50), density=False, edgecolor=None)
    axs[2].set_title('$sfbet$')

    if epcap_plot:
        axs[3].hist(p4, bins=np.linspace(min(p4),max(p4),nbins), density=False, color='darkblue', edgecolor=None)
        # axs[3].hist(p4, bins=np.linspace(0,1,50), density=False, edgecolor=None)
        axs[3].set_title('$epcap$')
    else:
        axs[3].axis('off') 
        
    # if fmcap_plot:
    #     axs[4].hist(p5, bins=np.linspace(min(p5),max(p5),500), density=False, edgecolor='black')
    #     axs[4].set_xscale('log')
    #     axs[4].set_title('$fmcap$')
    # else: 
    #     axs[4].axis('off') 

    plt.tight_layout()
    plt.savefig('post_proc/figs/' + figname + '.pdf', dpi=300)
    plt.close()

def display_corr(X, nfigs=5):
    labels=['fmmob', 'SF', 'sfbet', 'epcap', 'fmcap']
    ranges = [[610, 7500], [0.4, 0.5], [100, 200], [0.5, 0.6], [1e-8, 1e-6]]
    X = pd.DataFrame(X)
    
    fig, axs = plt.subplots(nfigs,nfigs,figsize=(9,8))

    for ax in fig.get_axes():
        ax.label_outer()

    for axy in range(nfigs):
        # axs[axy, 0].set_ylabel('X'+str(axy))
        axs[axy, 0].set_ylabel(labels[axy])
        # axs[axy, 0].set_ylim(ranges[axy])
        for axx in range(axy):
            axs[axy, axx].scatter(X.iloc[:,axx],X.iloc[:,axy], s=0.5)

            axs[axy, axx].set_xlim(ranges[axx])
            axs[axy, axx].set_ylim(ranges[axy])

            # # # SHOW FMMOB CORRELATION
            # if labels[axx] == 'fmmob' and labels[axy] == 'fmcap':
            #     # axs[axy, 0].set_ylim([1e-8, 1e-6])
            #     val_fmmob = X.iloc[:,axx]
            #     val_fmcap = X.iloc[:,axy]

            #     filt = val_fmcap < 1e-6

            #     val_fmmob = val_fmmob[filt]
            #     val_fmcap = val_fmcap[filt]
            #     df = pd.concat([val_fmmob ,val_fmcap], axis=1)
            #     csv_file = "correlation.csv"
            #     df.to_csv(csv_file, index=False)
                
            # if labels[axx] == 'fmmob' and labels[axy] == 'epcap':
            #     axs[axy, 0].set_ylim([0.5, 0.6])
            # if labels[axx] == 'fmmob' and labels[axy] == 'sfbet':
            #     axs[axy, 0].set_ylim([100, 200])
            # if labels[axx] == 'fmmob' and labels[axy] == 'SF':
            #     axs[axy, 0].set_ylim([0.4, 0.5])

            # # SHOW SF CORRELATION
            # if labels[axx] == 'SF' and labels[axy] == 'fmcap':
            #     axs[axy, axx].set_ylim([1e-8, 1e-6])
            #     axs[axy, axx].set_xlim([0.4, 0.5])
            # if labels[axx] == 'SF' and labels[axy] == 'epcap':
            #     axs[axy, axx].set_ylim([0.5, 0.6])
            #     axs[axy, axx].set_xlim([0.4, 0.5])

        for axx in range(axy,nfigs):
            axs[axy, axx].axis('off') 
            # axs[axy, axx].set_facecolor('black')

    for axx in range(nfigs):
        # axs[axy, axx].set_xlabel('X'+str(axx))
        axs[axy, axx].set_xlabel(labels[axx])
        #axs[0, axx].set_xlabel(labels[axx])
        #axs[axy, axx].set_xlim(ranges[axx])
            
    plt.savefig('correlation3.2.pdf', dpi=300)

def generate_FoamQS_plotData(bps, filenames, core_params):
    xx_sw = np.linspace(core_params[0],1.0-core_params[1], 100, endpoint=False)
    xx_ut = core_params[8] * np.ones_like(xx_sw)
    
    for i in range(len(bps)):
        vec = np.zeros((len(bps[i]),len(xx_sw)))
        for j in range(len(bps[i])):
            print("Quality scan: ", j+1, "/", len(bps[i]))

            res = bps[i][j]
            theta = res.X
            model_fit = stars_full_class(core_params, xx_sw, xx_ut)
            fg_fit, mu_app_fit = model_fit.vec_func(theta[0] , theta[1] , theta[2], theta[3] , theta[4])
            vec[j,:] = mu_app_fit *1000
        
        fg_filename = filenames[i][:len(filenames[i])-4] + '_fg.csv'
        np.savetxt(fg_filename, fg_fit, delimiter=',')

        filename = filenames[i][:len(filenames[i])-4] + '_muapp_FoamQS.csv'
        df = pd.DataFrame(vec)
        df.to_csv(filename, index=False) 
        print('Foam quality scan .csv file has been written successfully:\n'+filenames[i])

def generate_FoamQS_plotData_v2(bp, filenames, core_params):
    xx_sw = np.linspace(core_params[0],1.0-core_params[1], 1000, endpoint=False)
    xx_ut = core_params[8] * np.ones_like(xx_sw)
    
    vec = np.zeros((len(bp),len(xx_sw)))
    for j in range(len(bp)):
        print("Quality scan: ", j+1, "/", len(bp))

        res = bp[j]
        theta = res.X
        model_fit = stars_full_class(core_params, xx_sw, xx_ut)
        fg_fit, mu_app_fit = model_fit.vec_func(theta[0] , theta[1] , theta[2], theta[3] , theta[4])
        vec[j,:] = mu_app_fit *1000
        
    fg_filename = filenames[:len(filenames)-4] + '_fg.csv'
    np.savetxt(fg_filename, fg_fit, delimiter=',')

    filename = filenames[:len(filenames)-4] + '_muapp_FoamQS.csv'
    df = pd.DataFrame(vec)
    df.to_csv(filename, index=False) 
    print('Foam quality scan .csv file has been written successfully:\n'+filenames)
        
def plot_FoamQS(filenames, colors, exp_data):
    assert len(filenames) == len(colors), "Input arrays must have the same length"

    for i in range(len(filenames)):
        fg_filename = filenames[i][:len(filenames[i])-4] + '_fg.csv'
        fg = np.loadtxt(fg_filename, delimiter=',')

        filename = filenames[i][:len(filenames[i])-4] + '_muapp_FoamQS.csv'
        df = pd.read_csv(filename)

        label = filenames[i]
        last_slash_index = label.rfind('/')
        label = label[last_slash_index + 1:-4]

        for index, row in df.iterrows():
            np_array = row.values
            if index == df.shape[0]-1:
                plt.plot(fg, np_array, color=colors[i], alpha=1.0, label=label)
            else:
                plt.plot(fg, np_array, color=colors[i], alpha=1.0)
    
    plt.errorbar(exp_data['fg_QS'], exp_data['mu_app_QS']*1000, yerr=exp_data['mu_app_QS_std']*1000, fmt='bo', capsize=5, label=r'LASURF - $u_t\;\approx\;6.8$ [ft/day]')
    
    plt.ylabel(r"$\mu_{app} \, [\mathrm{cP}]$")
    plt.xlabel("foam quality")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.legend(loc='lower center',fontsize="6")
    plt.savefig('post_proc/figs/FQS.pdf', dpi = 300)
    plt.close()

def plot_FoamQS_CI(filenames, colors, exp_data):
    assert len(filenames) == len(colors), "Input arrays must have the same length"

    core_params, exp_data = read_expData()

    for i in range(len(filenames)):
        fg_filename = filenames[i][:len(filenames[i])-4] + '_fg.csv'
        fg = np.loadtxt(fg_filename, delimiter=',')

        filename = filenames[i][:len(filenames[i])-4] + '_muapp_FoamQS.csv'
        df = pd.read_csv(filename)

        label = filenames[i]
        last_slash_index = label.rfind('/')
        label = label[last_slash_index + 1:-4]

        vec = np.zeros((df.shape[0],len(fg)))

        for index, row in df.iterrows():
            # np_array = row.values
            vec[index,:] = row.values
            # if index == df.shape[0]-1:
                # plt.plot(fg, np_array, color=colors[i], alpha=1.0, label=label)
            # else:
                # plt.plot(fg, np_array, color=colors[i], alpha=1.0)
    
        pcer5 = []
        pcer95 = []
        for k in range(len(fg)):
            pcer5.append(np.percentile(vec[:,k], 5))
            pcer95.append(np.percentile(vec[:,k], 95))

        if filenames[i] == 'post_proc/parameter_optimization/approach2.2/approach_2.2.csv': plt.fill_between(fg, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1),linewidth=2.0, linestyle='--', edgecolor='dimgray',facecolor='none', label='Approach B - CI [5\%,95\%]')
        if filenames[i] == 'post_proc/parameter_optimization/approach2.1/approach_2.1.csv': plt.fill_between(fg, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1), alpha=1.0, color='tab:cyan', label='Approach C - CI [5\%,95\%]')
        if filenames[i] == 'post_proc/parameter_optimization/approach3/approach_3.csv': plt.fill_between(fg, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1), alpha=1.0, color='tab:red', label='Approach D - CI [5\%,95\%]')
        if filenames[i] == 'post_proc/parameter_optimization/approach3.2/approach_3.2.csv': plt.fill_between(fg, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1), alpha=1.0, color='orange', label='Approach E - CI [5\%,95\%]')

    fit_2 = {
            'fmmob' :  229.2946839614043, 
            'SF'    :  0.4565260114073254, 
            'sfbet' :  494.7222375526418
        }
    xx_sw = np.linspace(core_params[0],1.0-core_params[1], 2000, endpoint=False)
    xx_ut = core_params[8] * np.ones_like(xx_sw)
    model_fit2 = stars_class(core_params, xx_sw, xx_ut)
    fg_fit2, mu_app_fit2   = model_fit2.vec_func(fit_2['fmmob'] , fit_2['SF'] , fit_2['sfbet'])
    plt.plot(fg_fit2, mu_app_fit2*1000, c='k', linewidth=1.0, linestyle='-.', label='Approach A')
    

    plt.errorbar(exp_data['fg_QS'], exp_data['mu_app_QS']*1000, yerr=exp_data['mu_app_QS_std']*1000, fmt='bo', capsize=5, label=r'LASURF - $u_t\;\approx\;6.8$ [ft/day]')
    
    # plt.ylim([-30,85])
    plt.ylabel(r"$\mu_{app} \, [\mathrm{cP}]$")
    plt.xlabel("foam quality")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.legend(loc='lower center',fontsize="6")
    plt.savefig('post_proc/figs/FQS.pdf', dpi = 300)
    plt.close()

def plot_FoamQS_CI_v2(filenames, exp_data):

    core_params, exp_data = read_expData()
    
    fg_filename = filenames[:len(filenames)-4] + '_fg.csv'
    print(fg_filename)
    fg = np.loadtxt(fg_filename, delimiter=',')

    filename = filenames[:len(filenames)-4] + '_muapp_FoamQS.csv'
    df = pd.read_csv(filename)

    label = filenames
    last_slash_index = label.rfind('/')
    label = label[last_slash_index + 1:-4]

    vec = np.zeros((df.shape[0],len(fg)))

    for index, row in df.iterrows():
        # np_array = row.values
        vec[index,:] = row.values
        # if index == df.shape[0]-1:
            # plt.plot(fg, np_array, color=colors[i], alpha=1.0, label=label)
        # else:
            # plt.plot(fg, np_array, color=colors[i], alpha=1.0)

    pcer5 = []
    pcer95 = []
    for k in range(len(fg)):
        pcer5.append(np.percentile(vec[:,k], 5))
        pcer95.append(np.percentile(vec[:,k], 95))

    if filenames == './post_proc/parameter_optimization/approach2.2/approach2.2.csv': plt.fill_between(fg, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1),alpha=1.0, color='tab:cyan', label='Approach B - CI [5\%,95\%]')
    # if filenames == './post_proc/parameter_optimization/approach2.2/approach2.2.csv': plt.fill_between(fg, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1),linewidth=2.0, linestyle='--', edgecolor='dimgray',facecolor='none', label='Approach B - CI [5\%,95\%]')
    if filenames == './post_proc/parameter_optimization/approach2.1_wideRange/approach2.1_wideRange.csv': plt.fill_between(fg, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1), alpha=1.0, color='tab:cyan', label='Approach C - CI [5\%,95\%]')
    if filenames == './post_proc/parameter_optimization/approach3/approach3.csv': plt.fill_between(fg, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1), alpha=1.0, color='tab:red', label='Approach D - CI [5\%,95\%]')
    # if filenames == './post_proc/parameter_optimization/approach3.2/approach3.2.csv': plt.fill_between(fg, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1), alpha=1.0, color='orange', label='Approach E - CI [5\%,95\%]')
    # if filenames == './post_proc/parameter_optimization/approach3.2/approach3.2.csv': plt.fill_between(fg, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1), alpha=1.0, color='tab:red', label='Approach E - CI [5\%,95\%]')
    
    if filenames == './post_proc/parameter_optimization/approach3.2/approach3.2.csv': plt.fill_between(fg, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1), alpha=1.0, color='tab:red', label='Approach C - CI [5\%,95\%]')


    # If it is the last plot, then...
    if filenames == './post_proc/parameter_optimization/approach3.2/approach3.2.csv':
        fit_2 = {
                'fmmob' :  170.11309586607092, 
                'SF'    :  0.45656355172634727, 
                'sfbet' :  504.1431556457855
            }
        xx_sw = np.linspace(core_params[0],1.0-core_params[1], 1000, endpoint=False)
        xx_ut = core_params[8] * np.ones_like(xx_sw)
        model_fit2 = stars_class(core_params, xx_sw, xx_ut)
        fg_fit2, mu_app_fit2   = model_fit2.vec_func(fit_2['fmmob'] , fit_2['SF'] , fit_2['sfbet'])
        plt.plot(fg_fit2, mu_app_fit2*1000, c='k', linewidth=1.0, linestyle='-.', label='Approach A')
    

    
    
    # If it is the last plot, then...
    if filenames == './post_proc/parameter_optimization/approach3.2/approach3.2.csv':
        plt.errorbar(exp_data['fg_QS'], exp_data['mu_app_QS']*1000, yerr=exp_data['mu_app_QS_std']*1000, fmt='bo', capsize=5, label=r'\textrm{Exp. Data} - $u_t\;\approx\;6.8$ [ft/day]')
        plt.ylabel(r"$\mu_{app} \, [\mathrm{cP}]$")
        plt.xlabel(r"$f_g \, [-]$")

        # plt.legend(loc='best')
        # plt.legend(loc='lower center',fontsize="6")

        # reordering the labels 
        handles, labels = plt.gca().get_legend_handles_labels() 
        # specify order 
        order = [2, 0, 1, 3] 
        # order = [0,1,2,3] 
        # pass handle & labels lists along with order as below 
        plt.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower center',fontsize="6") 

        plt.tight_layout()
        plt.savefig('post_proc/figs/FQS.pdf', dpi = 300)
        plt.close()

def generate_FoamFRS_plotData(bps, filenames, core_params, exp_data):
    xx_sw = np.linspace(core_params[0],1.0-core_params[1], 20000, endpoint=False)
    xx_ut = np.linspace(2.39947600e-06, 5.99869001e-05, 50)

    for i in range(len(exp_data['fg_FR_exp'])):
        for j in range(len(bps)):
            vec = np.zeros((len(bps[j]),len(xx_ut)))
            for k in range(len(bps[j])):
                res = bps[j][k]
                theta = res.X
                muapp = []
                for h in range(len(xx_ut)):
                    xx_ut_prov = xx_ut[h] * np.ones_like(xx_sw)
                
                    model_fit = stars_full_class(core_params, xx_sw, xx_ut_prov)
                    fg_fit, mu_app_fit   = model_fit.vec_func(theta[0] , theta[1] , theta[2], theta[3] , theta[4])

                    aux = get_idx(fg_fit, [exp_data['fg_FR_exp'][i]], 1e-3)
                    muapp.append(mu_app_fit[aux][0])
                vec[k,:] = np.array(muapp)*1000
                print(filenames[j] + ': ' + str(k+1) + '|' + str(len(bps[j])))

            filename = filenames[j][:len(filenames[j])-4] + '_muapp_FRS_' + str(exp_data['fg_FR_exp'][i]) + '.csv'
            df = pd.DataFrame(vec)
            df.to_csv(filename, index=False) 
            print('Flow rate scan .csv file has been written successfully:\n'+filename)
    
            last_slash_index = filenames[j].rfind('/')
            ut_filename = filenames[j][:last_slash_index] + '/ut.csv'
            np.savetxt(ut_filename, xx_ut, delimiter=',')  

def generate_FoamFRS_plotData_v2(bp, filenames, core_params, exp_data):
    xx_sw = np.linspace(core_params[0],1.0-core_params[1], 20000, endpoint=False)
    xx_ut = np.linspace(2.39947600e-06, 5.99869001e-05, 50)

    for i in range(len(exp_data['fg_FR_exp'])):
        vec = np.zeros((len(bp),len(xx_ut)))
        for k in range(len(bp)):
            res = bp[k]
            theta = res.X
            muapp = []
            for h in range(len(xx_ut)):
                xx_ut_prov = xx_ut[h] * np.ones_like(xx_sw)
            
                model_fit = stars_full_class(core_params, xx_sw, xx_ut_prov)
                fg_fit, mu_app_fit   = model_fit.vec_func(theta[0] , theta[1] , theta[2], theta[3] , theta[4])

                aux = get_idx(fg_fit, [exp_data['fg_FR_exp'][i]], 1e-3)
                muapp.append(mu_app_fit[aux][0])
            vec[k,:] = np.array(muapp)*1000
            print(filenames + ': ' + str(k+1) + '|' + str(len(bp)))

        filename = filenames[:len(filenames)-4] + '_muapp_FRS_' + str(exp_data['fg_FR_exp'][i]) + '.csv'
        df = pd.DataFrame(vec)
        df.to_csv(filename, index=False) 
        print('Flow rate scan .csv file has been written successfully:\n'+filename)

        last_slash_index = filenames.rfind('/')
        ut_filename = filenames[:last_slash_index] + '/ut.csv'
        np.savetxt(ut_filename, xx_ut, delimiter=',')  

def plot_FoamFRS(filenames, colors, exp_data):
    assert len(filenames) == len(colors), "Input arrays must have the same length"

    last_slash_index = filenames[0].rfind('/')
    ut_filename = filenames[0][:last_slash_index] + '/ut.csv'
    ut = np.loadtxt(ut_filename, delimiter=',')

    for i in range(len(exp_data['fg_FR_exp'])):
        for j in range(len(filenames)):
            filename = filenames[j][:len(filenames[j])-4] + '_muapp_FRS_' + str(exp_data['fg_FR_exp'][i]) + '.csv'
            df = pd.read_csv(filename)

            last_slash_index = filename.rfind('/')
            label = filename[last_slash_index + 1:-18]

            for index, row in df.iterrows():
                np_array = row.values
                if index == df.shape[0]-1:
                    plt.plot(ut, np_array, color=colors[j], alpha=1.0, label=label)
                else:
                    plt.plot(ut, np_array, color=colors[j], alpha=1.0)
        
        plt.errorbar(exp_data['ut_FR'][i], exp_data['mu_app_FR'][i]*1000, yerr=exp_data['mu_app_FR_std'][i]*1000, fmt='bo', capsize=5, label='LASURF - $f_g\;=\;$'+str(exp_data['fg_FR_exp'][i]))
    
        plt.ylabel(r"$\mu_{app} \, [\mathrm{cP}]$")
        plt.xlabel("$u_t \, [ft/day]$")
        plt.legend( fontsize="6", loc ="upper right")
        plt.tight_layout()
        plt.savefig('post_proc/figs/FRS_' + str(exp_data['fg_FR_exp'][i]) + '.pdf', dpi=300)
        plt.close()

def plot_FoamFRS_CI(filenames, colors, fg_FR_exp = None):
    assert len(filenames) == len(colors), "Input arrays must have the same length"
    assert fg_FR_exp != None, "Provide 'fg_FR_exp'"

    core_params, exp_data = read_expData()

    last_slash_index = filenames[0].rfind('/')
    ut_filename = filenames[0][:last_slash_index] + '/ut.csv'
    ut = np.loadtxt(ut_filename, delimiter=',')
    ut *= 283465 # convert to ft/day

    for i in range(len(exp_data['fg_FR_exp'])):
        for j in range(len(filenames)):
            filename = filenames[j][:len(filenames[j])-4] + '_muapp_FRS_' + str(exp_data['fg_FR_exp'][i]) + '.csv'
            df = pd.read_csv(filename)

            last_slash_index = filename.rfind('/')
            # label = filename[last_slash_index + 1:-18]
            
            vec = np.zeros((df.shape[0],len(ut)))
            for index, row in df.iterrows():
                vec[index,:] = row.values

            pcer5 = []
            pcer95 = []
            for k in range(len(ut)):
                pcer5.append(np.percentile(vec[:,k], 5))
                pcer95.append(np.percentile(vec[:,k], 95))   

            if filenames[j] == 'post_proc/parameter_optimization/approach2.1/approach_2.1.csv': plt.fill_between(ut, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1), alpha=1.0, color='tab:cyan', label='Approach C - CI [5\%,95\%]')
            elif filenames[j] == 'post_proc/parameter_optimization/approach2.2/approach_2.2.csv': plt.fill_between(ut, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1),linestyle='--', edgecolor='dimgray',facecolor='none', label='Approach B - CI [5\%,95\%]')
            elif filenames[j] == 'post_proc/parameter_optimization/approach3/approach_3.csv': plt.fill_between(ut, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1), alpha=1.0, color='tab:red', label='Approach D - CI [5\%,95\%]')
            elif filenames[j] == 'post_proc/parameter_optimization/approach3.2/approach_3.2.csv': plt.fill_between(ut, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1),linestyle='--', edgecolor='orange',facecolor='none', label='Approach E - CI [5\%,95\%]')
         
        fit_2 = {
            'fmmob' :  229.2946839614043, 
            'SF'    :  0.4565260114073254, 
            'sfbet' :  494.7222375526418
        }
        muapp = []
        xx_sw = np.linspace(core_params[0],1.0-core_params[1], 20000, endpoint=False)
        for j in range(len(ut)):
            xx_ut_prov = ut[j] * np.ones_like(xx_sw)

            model_fit2 = stars_class(core_params, xx_sw, xx_ut_prov)
            fg_fit2, mu_app_fit2   = model_fit2.vec_func(fit_2['fmmob'] , fit_2['SF'] , fit_2['sfbet'])

            aux = get_idx(fg_fit2, [fg_FR_exp[i]], 1e-3)
            muapp.append(mu_app_fit2[aux][0])
            #print([xx_ut_prov[0]], fg_fit2[aux], mu_app_fit2[aux])
            #plt.scatter(xx_ut_prov[0]*283465, mu_app_fit2[aux]*1000, c='r')
        plt.plot(ut, np.array(muapp)*1000, c='k', linestyle='-', linewidth=1.5, label='Approach A')

        plt.errorbar(exp_data['ut_FR'][i]*283465, exp_data['mu_app_FR'][i]*1000, yerr=exp_data['mu_app_FR_std'][i]*1000, fmt='bo', capsize=5, label='LASURF - $f_g\;=\;$'+str(exp_data['fg_FR_exp'][i]))
    
        plt.ylabel(r"$\mu_{app} \, [\mathrm{cP}]$")
        plt.xlabel("$u_t \, [ft/day]$")
        plt.legend( fontsize="6", loc ="upper right")
        plt.tight_layout()
        plt.savefig('post_proc/figs/FRS_' + str(exp_data['fg_FR_exp'][i]) + '.pdf', dpi=300)
        plt.close()

def plot_FoamFRS_CI_v2(filenames, fg_FR_exp = None):
    assert fg_FR_exp != None, "Provide 'fg_FR_exp'"

    core_params, exp_data = read_expData()

    last_slash_index = filenames.rfind('/')
    ut_filename = filenames[:last_slash_index] + '/ut.csv'
    ut = np.loadtxt(ut_filename, delimiter=',')
    ut *= 283465 # convert to ft/day

    filename = filenames[:len(filenames)-4] + '_muapp_FRS_' + str(fg_FR_exp) + '.csv'
    df = pd.read_csv(filename)

    last_slash_index = filename.rfind('/')
    # label = filename[last_slash_index + 1:-18]
    
    vec = np.zeros((df.shape[0],len(ut)))
    for index, row in df.iterrows():
        vec[index,:] = row.values

    pcer5 = []
    pcer95 = []
    for k in range(len(ut)):
        pcer5.append(np.percentile(vec[:,k], 5))
        pcer95.append(np.percentile(vec[:,k], 95))   

    if filenames == './post_proc/parameter_optimization/approach2.1_wideRange/approach2.1_wideRange.csv': plt.fill_between(ut, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1), alpha=1.0, color='tab:cyan', label='Approach C - CI [5\%,95\%]')
    # elif filenames == './post_proc/parameter_optimization/approach2.2/approach2.2.csv': plt.fill_between(ut, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1),linestyle='--', edgecolor='dimgray',facecolor='none', label='Approach B - CI [5\%,95\%]')
    elif filenames == './post_proc/parameter_optimization/approach2.2/approach2.2.csv': plt.fill_between(ut, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1),alpha=1.0, color='tab:cyan', label='Approach B - CI [5\%,95\%]')
    elif filenames == './post_proc/parameter_optimization/approach3/approach3.csv': plt.fill_between(ut, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1), alpha=1.0, color='tab:red', label='Approach D - CI [5\%,95\%]')
    # elif filenames == './post_proc/parameter_optimization/approach3.2/approach3.2.csv': plt.fill_between(ut, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1),linestyle='--', edgecolor='orange',facecolor='none', label='Approach E - CI [5\%,95\%]')
    # elif filenames == './post_proc/parameter_optimization/approach3.2/approach3.2.csv': plt.fill_between(ut, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1),alpha=1.0, color='tab:red', label='Approach E - CI [5\%,95\%]')

    elif filenames == './post_proc/parameter_optimization/approach3.2/approach3.2.csv': plt.fill_between(ut, np.array(pcer5,ndmin=1), np.array(pcer95,ndmin=1),alpha=1.0, color='tab:red', label='Approach C - CI [5\%,95\%]')
    
    ###
    print('\n')
    # print(pcer5,'\n',pcer95)
    if fg_FR_exp == 0.6: i = 0
    elif fg_FR_exp == 0.8: i = 1
    ut_exp = exp_data['ut_FR'][i]*283465
    # indxs = [i for i in np.argmin(abs(ut_exp[i]-ut))]
    indxs = [np.argmin(abs(ut - ut_exp[i])) for i in range(len(ut_exp))]
    pcer5 = np.array(pcer5)
    pcer95 = np.array(pcer95)
    indxs = np.array(indxs)
    print(ut_exp,pcer5[indxs])
    print(ut_exp,pcer95[indxs])
    # print(exp_data['mu_app_FR'][i]*1000,pcer5[indxs])
    errors_5 = np.abs(100*(pcer5[indxs]-exp_data['mu_app_FR'][i]*1000)/(exp_data['mu_app_FR'][i]*1000))
    errors_95 = np.abs(100*(pcer95[indxs]-exp_data['mu_app_FR'][i]*1000)/(exp_data['mu_app_FR'][i]*1000))
    print(exp_data['mu_app_FR'][i]*1000)
    print(errors_5,errors_95)
    ###

    # if it is the last plot, then ...
    if filenames == './post_proc/parameter_optimization/approach3.2/approach3.2.csv':
        fit_2 = {
            'fmmob' :  170.11309586607092, 
            'SF'    :  0.45656355172634727, 
            'sfbet' :  504.1431556457855
        }
        muapp = []
        xx_sw = np.linspace(core_params[0],1.0-core_params[1], 20000, endpoint=False)
        for j in range(len(ut)):
            xx_ut_prov = ut[j] * np.ones_like(xx_sw)

            model_fit2 = stars_class(core_params, xx_sw, xx_ut_prov)
            fg_fit2, mu_app_fit2   = model_fit2.vec_func(fit_2['fmmob'] , fit_2['SF'] , fit_2['sfbet'])

            aux = get_idx(fg_fit2, [fg_FR_exp], 1e-3)
            muapp.append(mu_app_fit2[aux][0])
            #print([xx_ut_prov[0]], fg_fit2[aux], mu_app_fit2[aux])
            #plt.scatter(xx_ut_prov[0]*283465, mu_app_fit2[aux]*1000, c='r')
        plt.plot(ut, np.array(muapp)*1000, c='k', linestyle='-.', label='Approach A')

        if fg_FR_exp == 0.6: i = 0
        elif fg_FR_exp == 0.8: i = 1
        plt.errorbar(exp_data['ut_FR'][i]*283465, exp_data['mu_app_FR'][i]*1000, yerr=exp_data['mu_app_FR_std'][i]*1000, fmt='bo', capsize=5, label=r'$\textrm{Exp. Data}$ - $f_g\;=\;$'+str(fg_FR_exp))
    
        plt.ylabel(r"$\mu_{app} \, [\mathrm{cP}]$")
        plt.xlabel("$u_t \, [ft/day]$")
        # plt.legend( fontsize="6", loc ="upper right")

        # reordering the labels 
        handles, labels = plt.gca().get_legend_handles_labels() 
        # specify order 
        order = [2, 0, 1, 3] 
        # order = [0,1,2,3] 
        # pass handle & labels lists along with order as below 
        plt.legend([handles[i] for i in order], [labels[i] for i in order], loc='best',fontsize="6") 

        plt.tight_layout()
        plt.savefig('post_proc/figs/FRS_' + str(fg_FR_exp) + '.pdf', dpi=300)
        plt.close()

def ident_profiles(ranges, res, figname=None):
    assert figname != None, "'figname' parameter must be provided"

    #plt.rc('font', size=20)
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))     

    axs[0].plot(ranges[0],res[0])
    axs[0].set_title('$fmmob$')
    # axs[0].set_ylim([0.25,0.4])
    axs[0].set_ylabel('frequency')
    axs[0].set_yscale('log')

    axs[1].plot(ranges[1],res[1])
    axs[1].set_title('$SF$')
    axs[1].set_yscale('log')

    axs[2].plot(ranges[2],res[2])
    axs[2].set_title('$sfbet$')
    axs[2].set_yscale('log')

    axs[3].plot(ranges[3],res[3])
    axs[3].set_title('$epcap$')
    axs[3].set_ylim([0.1,1.3])
    axs[3].set_yscale('log')
 
    axs[4].plot(ranges[4],res[4])
    axs[4].set_xscale('log')
    axs[4].set_title('$fmcap$')
    axs[4].set_ylim([0.1,1.3])
    axs[4].set_yscale('log')
    # axs[4].set_ylim([0.32801,0.32804])

    # idx = np.argmin(np.array(res[4]))
    # print(idx)
    # print(ranges[4][idx], res[4][idx])

    plt.tight_layout()
    plt.savefig('post_proc/figs/' + figname + '.pdf', dpi=300)
    plt.close()

def ident_profiles_v2(ranges, res, figname=None, plot_epcap = True, plot_fmcap = True):
    assert figname != None, "'figname' parameter must be provided"

    plt.rc('font', size=15)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3))     

    # axs[0].plot(ranges[0],res[0])
    axs[0].semilogy(ranges[0],res[0],linewidth=2)
    axs[0].set_title('$fmmob$')
    # axs[0].set_ylim([0.25,0.4])
    axs[0].set_ylabel(r'$\mathcal{X}^2$')
    print('min fmmob: ', ranges[0][np.argmin(np.array(res[0]))])

    axs[1].set_yscale("log")
    axs[1].plot(ranges[1],res[1],linewidth=2)
    axs[1].set_title('$SF$')
    print('min SF: ', ranges[1][np.argmin(np.array(res[1]))])

    axs[2].set_yscale("log")
    axs[2].plot(ranges[2],res[2],linewidth=2)
    axs[2].set_title('$sfbet$')
    print('min sfbet: ', ranges[2][np.argmin(np.array(res[2]))])

    if plot_epcap:
        axs[3].set_yscale("log")
        axs[3].plot(ranges[3],res[3],linewidth=2)
        axs[3].set_title('$epcap$')
        axs[3].set_ylim([0.1,2.3])
        print('min epcap: ', ranges[3][np.argmin(np.array(res[3]))])
    else: 
        axs[3].axis('off') 
    
    # if plot_fmcap: 
    #     axs[4].plot(ranges[4],res[4])
    #     axs[4].set_xscale('log')
    #     axs[4].set_title('$fmcap$')
    #     axs[4].set_ylim([0.1,1.3])
    #     print('min fmcap: ', ranges[4][np.argmin(np.array(res[4]))])
    # else: 
    #     axs[4].axis('off') 

    # idx = np.argmin(np.array(res[4]))
    # print(idx)
    # print(ranges[4][idx], res[4][idx])

    # plt.tight_layout()
    plt.show()
    plt.savefig('post_proc/figs/' + figname + '.pdf', dpi=300)
    plt.close()

def muApp_to_Pdrop(core_params, exp_data, fit):
    ut_QS = core_params[8] * np.ones_like(exp_data['sw_QS'])
    xx_sw = np.linspace(core_params[0],1.0-core_params[1], 2000, endpoint=False)
    xx_ut = core_params[8] * np.ones_like(xx_sw)

    model_fit = stars_full_class(core_params, xx_sw, xx_ut)
    fg_fit, mu_app_fit   = model_fit.vec_func(fit['fmmob'] , fit['SF'] , fit['sfbet'], fit['epcap'], fit['fmcap'])
    
    aux = get_idx(fg_fit, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], 1e-2)  
    PDrop = (np.multiply(mu_app_fit[aux], ut_QS) * core_params[-1] / (core_params[-2])) * 0.000145038/1.0000018082621 # convert [Pa] to [Psi]
    print(PDrop.tolist())

def log_to_file_2(log_message):
    with open('ranges_UQ.txt', 'a') as file:
        file.write(log_message + '\n')

def filter_to_UQpropag(ranges, res, writeFile = False):

        ranges_UQ = []
        res_UQ = []

        filter_value = [0.5, 0.5, 0.5, 0.5, 0.5]

        for i in range(len(res)):
            idx = [int(j) for j, value in enumerate(res[i]) if value < filter_value[i]]
                        
            
            vec = ranges[i]
            aux = [vec[j] for j in idx]
            ranges_UQ.append(np.array(aux))
            
            vec = res[i]
            aux = [vec[j] for j in idx]
            res_UQ.append(np.array(aux))

        # Write ranges on file
        if writeFile:
            for i in range(len(res_UQ)):
                log_message = f"[{min(ranges_UQ[i])}, {max(ranges_UQ[i])}]"
                print(log_message)
                log_to_file_2(log_message) 
        
        return ranges_UQ, res_UQ