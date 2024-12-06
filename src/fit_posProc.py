import scienceplots

from opt_def import *
from foam_fit import *
from post_proc_funcs import *

if __name__ == '__main__':
    # plt.style.use('science')
    core_params, exp_data = read_expData()
    print(core_params, exp_data)

    # # Read results
    bp  = Read_postProcfiles(filesDir='./post_proc/parameter_optimization/approach2.1/result_seed_', n_results = 50, root_seed = 1)   
    bp2 = Read_postProcfiles(filesDir='./post_proc/parameter_optimization/approach2.2/result_seed_', n_results = 50, root_seed = 1)   
    bp3 = Read_postProcfiles(filesDir='./post_proc/parameter_optimization/approach3/result_seed_', n_results =50, root_seed = 1)       
    bp4 = Read_postProcfiles(filesDir='./post_proc/parameter_optimization/approach3.2/result_seed_', n_results =50, root_seed = 1)      
    bps = [bp, bp2, bp3, bp4]
    # bps = [bp]

    # # Gather all parameters results
    all_thetas, Fs = get_thetas(bp)
    all_thetas2, Fs = get_thetas(bp2)
    all_thetas3, Fs = get_thetas(bp3)
    all_thetas4, Fs = get_thetas(bp4)

    # print(np.min(Fs), np.max(Fs), np.std(Fs), np.median(Fs), np.mean(Fs))
    # plt.hist(Fs,bins=50)
    # plt.show()
    # plt.close()
    # teste

    # # Histogram
    summary_of_results(all_thetas, core_params, 'hist_approach2.1', fmcap_plot = True)
    summary_of_results(all_thetas2, core_params, 'hist_approach2.2', fmcap_plot = False)
    summary_of_results(all_thetas3, core_params, 'hist_approach3', fmcap_plot = True)
    summary_of_results(all_thetas4, core_params, 'hist_approach3.2', fmcap_plot = False)
    
    # display_corr(all_thetas3)
    # teste

    # C = []
    # fmmobs = []
    # fmcaps = []
    # for i in range(len(all_thetas3)):
    #     fmcap = all_thetas3[i][4]
    #     if fmcap < 1e-6:
    #         fmmob = all_thetas3[i][0]
    #         fmmobs.append(fmmob)
    #         fmcaps.append(fmcap)
    #         epcap = all_thetas3[i][3]
    #         C.append(fmmob * np.power(fmcap,epcap))
    #         plt.scatter(fmmob, fmcap, linewidth=0.1, edgecolor='steelblue', facecolors='none', s=10)
    # df = pd.DataFrame([fmmobs ,fmcaps])
    # df = df.transpose()
    # csv_file = "correlation.csv"
    # df.to_csv(csv_file, index=False)
    # print(df.head())
    # print(np.mean(C), np.std(C),np.min(C), np.max(C))

    # cte = np.median(C)
    # print("C = ", cte)
    # x = np.linspace(min(fmmobs), max(fmmobs), 200)
    # y = np.exp((1/epcap) * np.log(cte/x))
    # plt.plot(x,y,c='k',linewidth=1.5)
    # plt.xlabel(r'$fmmob$')
    # plt.ylabel(r'$fmcap$')
    # plt.savefig('expPlot.pdf', dpi=300)
    # plt.close()

    # plt.plot(range(len(C)), C)
    # plt.ylabel(r'$fmmob \times (fmcap)^{epcap}$')
    # plt.xlabel('Independent run')
    # plt.ylim([0.45, 0.5])
    # plt.savefig('CPlot.pdf', dpi=300)


    # Plot fg vs. mu_app graph for the 3 approaches
    postProc_files = ['post_proc/parameter_optimization/approach2.1/approach_2.1.csv',
                      'post_proc/parameter_optimization/approach2.2/approach_2.2.csv',
                      'post_proc/parameter_optimization/approach3/approach_3.csv',
                      'post_proc/parameter_optimization/approach3.2/approach_3.2.csv'  
    ]
    # postProc_files = ['post_proc/parameter_optimization/approach2.1/approach_2.1.csv'  
    # ]
    colors = ['tab:cyan', 'tab:gray', 'tab:red', 'darkorange']
    # colors = ['tab:cyan']


    # Generate data for plots
    generate_data = True

    # print(np.concatenate((exp_data['ut_FR'][0],exp_data['ut_FR'][1])))
    # Nca = (core_params[6] * np.concatenate((exp_data['ut_FR'][0],exp_data['ut_FR'][1])))/0.03
    # print(Nca)

    if generate_data:
        # If data has not been generated through optimization results, otherwise it is read from .csv
        generate_FoamQS_plotData(bps, filenames=postProc_files, core_params=core_params)
        generate_FoamFRS_plotData(bps, filenames=postProc_files, core_params=core_params, exp_data=exp_data)

    plot_FoamQS_CI(postProc_files, colors, exp_data)
    plot_FoamFRS_CI(postProc_files, colors, fg_FR_exp = (0.6, 0.8))
    # plot_FoamQS(postProc_files, colors, exp_data)
    # plot_FoamFRS(postProc_files, colors, exp_data)

    # fit = {
    #     'fmmob':    1386.0731, 
    #     'SF':       0.4578, 
    #     'sfbet':    144.2921, 
    #     'epcap':    0.5237, 
    #     'fmcap':    2.5594e-07
    # }
    # muApp_to_Pdrop(core_params, exp_data, fit)

    # fit = {
    #     'fmmob':    229.2946839614043, 
    #     'SF':       0.4565260114073254, 
    #     'sfbet':    494.7222375526418, 
    #     'epcap':    0.0, 
    #     'fmcap':    1.0
    # }
    # muApp_to_Pdrop(core_params, exp_data, fit)