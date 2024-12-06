
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import corner # https://github.com/dfm/corner.py/blob/main/corner/corner.py

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

#sample_file = 'stars_mcmc.arv'
#fid = open(sample_file, "w")
#p0 , p1 , p2 , p3 = trace['fmmob'] , trace['SF'] , trace['sfbet'] , trace['epcap']
#for k in range(len(p0)):
#fid.write("%d \t %0.8e \t %0.8e \t %0.8e \t %0.8e\n" % (k+1,p0[k],p1[k],p2[k],p3[k]))
#fid.close()


if __name__ == "__main__":

    data = np.loadtxt('stars_mcmc.arv', comments="#")
    param_list  = ['fmmob','SF','sfbet','epcap']
    param_label = ['$fmmob$','$SF$','$sfbet$','$epcap$']
    
    df = pd.DataFrame(data=data[:,1:],index=data[:,0],columns=param_list)
    df.info()

    limits = [[400, 850],
              [0.454, 0.462],
              [70, 240],
              [0.2, 0.8]]
    df_filtered = df
    for i in range(len(limits)):
        df_filtered = df_filtered[df_filtered[param_list[i]] > limits[i][0]]
        df_filtered = df_filtered[df_filtered[param_list[i]] < limits[i][1]]
    
    fig = corner.corner(df_filtered, show_titles=True, bins=100, labels=param_label, 
                            plot_datapoints=False,  #quantiles=[0.05, 0.95], 
                            title_fmt="0.1e")
    plt.plot()
    plt.tight_layout()
    
    name_fig = 'mcmc_stars.pdf'
    plt.savefig(name_fig, dpi=300)

    plt.clf()
    