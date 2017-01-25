import copy as cp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#cmap = plt.cm.Set1
#cseq = [plt.cm.Set1(i) for i in np.linspace(0, 1, 9, endpoint=True)]
#plt.rcParams['axes.color_cycle'] = cseq

"""
Plot thickened volume
"""


def getxsyszs(path2csv):
    df=pd.read_csv(path2csv, sep=',')
    xs = df.values[:,1]
    ys = df.values[:,2]
    zs = df.values[:,0]
    return xs,ys,zs

def plot_thick(X,Y,Z,out,mi,ma):
    obstacle = plt.Circle((1.5, .5), 0.2, color='k',alpha = 1, fill = False)
    obstacle2 = plt.Circle((1.5, .5), 0.2, color='k', alpha = .3)
    obstacle3 = plt.Circle((1.5, .5), 0.2, color='w')

    fig, ax = plt.subplots()
    #mi, ma = np.min(Z), np.max(Z)
    mi, ma = 0., np.max(Z)
    #lvl = np.array([0,.4*ma])
    #lvl = np.hstack([lvl,np.arange(ma*.5,ma,10)])
    
    mesh = ax.tripcolor(X,Y,Z, edgecolors = 'none',vmin=mi, vmax=ma)
    #ax.tricontourf(X,Y,Z,40,shading = 'gouraud',levels = lvl)
    ax.add_artist(obstacle3)
    ax.add_artist(obstacle2)
    ax.add_artist(obstacle)
    
    plt.xlim(1.,2.0)
    plt.ylim(0.5,1.)
    plt.gca().set_aspect('equal', adjustable='box')
    
    fig.colorbar(mesh,fraction=0.023, pad=0.02)

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.savefig(out+"ThickVol.pdf")

    return

X1, Y1, Z1 = getxsyszs("025Thick.csv")
X2, Y2, Z2 = getxsyszs("050Thick.csv")
X3, Y3, Z3 = getxsyszs("20Thick.csv")

mi,ma = 0, 30 
plot_thick(X1,Y1,Z1,"025",mi,ma)
plot_thick(X2,Y2,Z2,"050",mi,ma)
plot_thick(X3,Y3,Z3,"20",mi,ma)

"""
Zt = cp.deepcopy(Z)

thick = 10.
Zt[np.where(Z > thick)] = 1.
Zt[np.where(Z <= thick)] = 0
"""
