import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cmap = plt.cm.Set1
cseq = [plt.cm.Set1(i) for i in np.linspace(0, 1, 9, endpoint=True)]
plt.rcParams['axes.color_cycle'] = cseq

"""
Construncting the streamlines in Paraview, save them as .csv fiels
and this extracts the data and plots the streamlines of three 
files
"""


def getxsys(path2csv):
    df=pd.read_csv(path2csv, sep=',')
    xs = df.values[:,-3]
    ys = df.values[:,-2]
    return xs,ys

def individualLines(xs,ys):
    t = np.where(xs == 0)[0]
    
    spt = []
    ept = []

    for i,ele in enumerate(t):
        if ele == 0:
            spt.append(t[0])
        elif i == len(t)-1:
            ept.append(t[-1])
        elif ele - 1 == t[i-1]:
            continue
        else:
            spt.append(ele)
            ept.append(ele-1)

    return ept, spt

xs0,ys0 = getxsys('streamlines05.csv')
ept0, spt0 = individualLines(xs0,ys0)

xs1,ys1 = getxsys('streamlines10.csv')
ept1, spt1 = individualLines(xs1,ys1)

xs2,ys2 = getxsys('streamlines20.csv')
ept2, spt2 = individualLines(xs2,ys2)

obstacle = plt.Circle((1.5, .5), 0.2, color='k',alpha =.3)

fig, ax = plt.subplots()
ax.add_artist(obstacle)

ax.plot(xs0[int(spt0[0]):int(ept0[0]+1)], \
            ys0[int(spt0[0]):int(ept0[0]+1)], '--',color = cseq[0], label = '$n = 0.5$')
ax.plot(xs1[int(spt1[0]):int(ept1[0]+1)], \
            ys1[int(spt1[0]):int(ept1[0]+1)], '-',color = cseq[1], label = '$n = 1.0$')
ax.plot(xs2[int(spt2[0]):int(ept2[0]+1)], \
            ys2[int(spt2[0]):int(ept2[0]+1)], '-.',color = cseq[3], label = '$n = 2.0$')


for i in xrange(len(ept0)//2, len(ept0)):
    ax.plot(xs0[int(spt0[i]):int(ept0[i]+1)], \
            ys0[int(spt0[i]):int(ept0[i]+1)], '--',color = cseq[0])

for i in xrange(len(ept1)//2, len(ept1)):
    ax.plot(xs1[int(spt1[i]):int(ept1[i]+1)], \
            ys1[int(spt1[i]):int(ept1[i]+1)], '-',color = cseq[1])

for i in xrange(len(ept2)//2, len(ept2)):
    ax.plot(xs2[int(spt2[i]):int(ept2[i]+1)], \
            ys2[int(spt2[i]):int(ept2[i]+1)], '-.',color = cseq[3])

ax.legend(['$n = 0.5$','$n = 1.0$','$n = 2.0$'],loc = 1)

plt.title('Streamlines for different choices of $n$', style = 'italic')
plt.xlabel('$x$', style = 'italic')
plt.ylabel('$y$', style = 'italic')
plt.xlim(.8,2.0)
plt.ylim(0.5,1.)
plt.gca().set_aspect('equal', adjustable='box')

fig.savefig('streamlines.pdf',orientation='landscape')

plt.close()

t = np.where(xs0 == 0)
d = np.diff(t)



ept, spt = individualLines(xs0,ys0)