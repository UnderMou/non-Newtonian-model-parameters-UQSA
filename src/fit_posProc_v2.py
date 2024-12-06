import scienceplots

from opt_def import *
from foam_fit import *
from post_proc_funcs import *

if __name__ == '__main__':
    plt.style.use('science')
    core_params, exp_data = read_expData()
    print(core_params, exp_data)

    # bp  = Read_postProcfiles(filesDir='./post_proc/parameter_optimization/approach1/result_seed_', n_results = 2000, root_seed = 1)   
    # all_thetas, Fs = get_thetas(bp)
    # summary_of_results(all_thetas, core_params, 'hist_approach1', fmcap_plot = False, epcap_plot = False)

    cases = ['approach2.2', 'approach3.2']
    post_proc_dirs = [['./post_proc/parameter_optimization/'+cases[0], 2000, 1],
                      ['./post_proc/parameter_optimization/'+cases[1], 2000, 1]]
    # post_proc_dirs = [post_proc_dirs[-1]]

    generate_data = False
    if generate_data:
        # Generate Data
        for i in range(len(post_proc_dirs)):
            # Read results
            bp  = Read_postProcfiles(filesDir=post_proc_dirs[i][0]+'/result_seed_', n_results = post_proc_dirs[i][1], root_seed = post_proc_dirs[i][2])   

            # Gather all parameters results
            all_thetas, Fs = get_thetas(bp)

            # Histogram
            fmcap_plot = False
            if i==0 : epcap_plot = False
            else : epcap_plot = True
            summary_of_results(all_thetas, core_params, 'hist_'+cases[i], fmcap_plot = fmcap_plot, epcap_plot = epcap_plot)

            # # Generate data for plots
            # generate_FoamQS_plotData_v2(bp, filenames=post_proc_dirs[i][0]+'/'+cases[i]+'.csv', core_params=core_params)
            # generate_FoamFRS_plotData_v2(bp, filenames=post_proc_dirs[i][0]+'/'+cases[i]+'.csv', core_params=core_params, exp_data=exp_data)

    plot_graph = True
    if plot_graph:
        # Plot Foam quality scan data
        for i in range(len(post_proc_dirs)):
            # plot_FoamQS_CI_v2(post_proc_dirs[i][0]+'/'+cases[i]+'.csv', exp_data)
            pass

        # Plot flow rate scan data
        # fg = 0.6
        for i in range(len(post_proc_dirs)):    
            plot_FoamFRS_CI_v2(post_proc_dirs[i][0]+'/'+cases[i]+'.csv', fg_FR_exp = 0.6)
        # fg = 0.8
        for i in range(len(post_proc_dirs)):    
            plot_FoamFRS_CI_v2(post_proc_dirs[i][0]+'/'+cases[i]+'.csv', fg_FR_exp = 0.8)

    # # NN
    # fit = {
    #     'fmmob':    688.7437945055274, 
    #     'SF':       0.45783802263044193, 
    #     'sfbet':    144.38275160971176, 
    #     'epcap':    0.5246370864805638, 
    #     'fmcap':    8.4e-08
    # }
    # # N
    # # fit = {
    # #     'fmmob':    170.113, 
    # #     'SF':       0.45656, 
    # #     'sfbet':    504.143, 
    # #     'epcap':    0, 
    # #     'fmcap':    1
    # # }
    # muApp_to_Pdrop(core_params, exp_data, fit)   
