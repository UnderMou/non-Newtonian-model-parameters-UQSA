from foam_fit import *
import matplotlib.pyplot as plt
import multiprocessing
import csv
import ast

def process_item(item):
    core_params, exp_data, opt_approach, par_fixed, par_range = item[0], item[1], item[2], item[3], item[4]
    
    XPL_arr = run_identifiability_analysis(core_params, exp_data, opt_approach, par_fixed, par_range)
    
    return XPL_arr

def plot_ranges(values, nt):

    n1 = int(0.1*nt)
    n2 = nt
    n3 = int(0.9*nt)

    y1,y2,y3=0*np.ones(n1), 1*np.ones(n2), 2*np.ones(n3)
    ys = np.concatenate((y1,y2,y3))

    plt.figure(figsize=(25,5))
    #plt.scatter(values, ys, s=0.25)
    plt.scatter(values, np.zeros_like(values), s=0.25)
    #plt.ylim([-0.001, 0.001])
    plt.savefig('range.pdf', dpi=500)
    plt.show()


def fmmob_spacing(nt, opt_approach, ref_region=[500, 3000]):
    
    vi = opt_approach['p0'][0]  
    vf = opt_approach['p0'][1]

    n1 = int(0.1*nt)
    n2 = nt
    n3 = int(0.9*nt)

    vec_1 = np.linspace(vi, ref_region[0], n1, endpoint=False)
    vec_2 = np.linspace(ref_region[0], ref_region[1], n2, endpoint=False)
    vec_3 = np.linspace(ref_region[1], vf, n3, endpoint=False)
    values = np.concatenate((vec_1, vec_2, vec_3))

    plot_ranges(values, nt)
    
    return values

def fmcap_spacing(nt, opt_approach):
    
    vi = opt_approach['p4'][0]  
    vf = opt_approach['p4'][1]

    values = np.zeros(nt)

    q = np.power(vf/vi,1/(nt-1))

    for i in range(nt):
        values[i] = (vi*np.power(q,i)).astype(float)
    
    return values

def log_to_file(log_message):
    with open('logfile.txt', 'a') as file:
        file.write(log_message + '\n')

def run_identifiability_analysis(core_params, exp_data, opt_approach, par_fixed, par_range):

    # OPTIMIZATION ALGORITHM
    # Sets the evolutionary algorithm
    algorithm = DE(
        pop_size=200,
        sampling= FloatRandomSampling(),
        variant="DE/rand/1/bin",
        CR=0.6,
        dither="vector",
        jitter=False
    )
    # Sets the stop criterion
    termination = DefaultSingleObjectiveTermination(
                        n_max_gen=2000,
                        n_max_evals=400000,
                        period=200)

    n_runs = 10  # number of independent runs of the optimization algorithm

    XPL_arr = []
    for j in range(len(par_range)):

        opt_approach[par_fixed] = [par_range[j], par_range[j]]

        NNFittingSTARS_problem = NNFittingSTARS(core_params, exp_data, opt_approach)

        # Run optimizations
        results = []
        root_seed = 1   # Sets seed for reproductibility of the results

        obj_func_values = []
        for i in range(n_runs):
            seed = root_seed + i
            # print(par_fixed, ' = ', par_range[j], j+1, '|', len(par_range), 'run: ', i+1, '|', n_runs)
            log_message = f"{par_fixed} = {par_range[j]} {j+1}/{len(par_range)} run: {i+1}|{n_runs}"
            print(log_message)
            log_to_file(log_message) 
    
            res = minimize(NNFittingSTARS_problem,
                    algorithm,
                    termination,
                    seed=seed,
                    verbose=False)

            obj_func_values.append(res.F)
        
        obj_func_values = np.array(obj_func_values)
        # print('median:', np.median(obj_func_values),
        #       'mean: ', np.mean(obj_func_values),
        #       'std: ', np.std(obj_func_values)           
        #      )
        XPL_arr.append(np.mean(obj_func_values))
    
    return XPL_arr

if __name__ == '__main__':

    core_params, exp_data = read_expData()

    """
    approach_id = 0   : approach 1
    approach_id = 1   : approach 2.1
    approach_id = 2   : approach 2.2
    approach_id = 3   : approach 3
    approach_id = 4   : approach 3.2
    approach_id = 5   : approach 4
    """
    approach_id = 3
    approaches = opt_approaches(core_params)
    opt_approach = approaches[approach_id]

    # Define ranges for indentifiability analysis
    ranges = []
    Np = 500
    for i in range(5):  # 5 parameters
        array = np.linspace(opt_approach['p'+str(i)][0], opt_approach['p'+str(i)][1], Np)

        ranges.append(array.tolist())
    
    with open('post_proc/parameter_identifiability/approach3/ranges_approach3_rand.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(ranges)


    data = [[core_params, exp_data, opt_approach, 'p'+str(i), ranges[i]] for i in range(5)]

    num_workers = 5

    with multiprocessing.Pool(processes=num_workers) as pool:
    
        results = pool.map(process_item, data)

    # print(results)
    
    filename = 'post_proc/parameter_identifiability/approach3/output_approach3_rand.csv'

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(results)
