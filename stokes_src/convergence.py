import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cmap = plt.cm.Set1
cseq = [plt.cm.Set1(i) for i in np.linspace(0, 1, 9, endpoint=True)]
plt.rcParams['axes.color_cycle'] = cseq

#1/4
ys0 = np.array([2.646284e+01, 1.169537e+01, 1.212114e+01, 7.789829e+00, 2.524910e+00,\
                2.728099e+00, 8.462916e-01, 2.614338e-01, 7.878814e-02, 1.946373e-02,\
                2.694979e-03, 3.001377e-04, 1.028377e-04, 3.518043e-05, 7.595351e-06, \
                5.524088e-07, 5.823987e-09])
#1/2
ys1 = np.array([2.329780e+00, 3.234193e-01, 1.331347e-01, 2.818738e-02, 4.571224e-03,\
                4.362849e-04, 1.998988e-05,8.015426e-08]) 


#3/4
ys2 = np.array([2.020358e+00, 5.639447e-02, 2.506581e-03, 6.590976e-05, 2.383281e-07,\
                1.026535e-11])

#2
ys3 = np.array([1.968851e+00, 5.043988e-02, 4.125207e-02, 2.056054e-02, 1.404624e-02,\
                9.153138e-03, 6.939208e-03, 4.736409e-03, 3.356202e-03, 2.201741e-03,\
                1.001184e-03, 4.092674e-04, 1.541586e-04, 3.190032e-05, 1.114820e-06,\
                1.832605e-09])

#3
ys4 = np.array([1.967423e+00, 6.713000e-02, 1.376282e-01, 1.162109e-01, 7.196204e-02,\
                5.429133e-02, 3.582088e-02, 3.253243e-02, 1.894180e-02, 1.622079e-02,\
                8.338439e-03, 7.832097e-03, 3.612157e-03, 4.315618e-03, 3.463704e-03,\
                7.793350e-04, 8.456676e-05, 3.359668e-06, 7.517352e-09])

#5
ys5 = np.array([1.967331e+00, 8.596344e-02, 1.666219e-01, 1.383496e-01, 1.244257e-01,\
                9.651761e-02, 7.842875e-02, 6.023933e-02, 4.629783e-02, 3.462088e-02,\
                3.594031e-02, 3.202958e-02, 2.790631e-02, 2.475807e-02, 2.166990e-02,\
                2.007560e-02, 1.950751e-02, 1.870532e-02, 8.006745e-03, 5.089805e-03,\
                3.399298e-03, 2.179031e-03, 1.054564e-03, 1.447646e-04, 2.054432e-06])

fig, ax = plt.subplots()
ax.semilogy(np.arange(len(ys0)), ys0, '-o', label = '$n=0.25$')
ax.semilogy(np.arange(len(ys1)), ys1, '-o', label = '$n=0.5$')
ax.semilogy(np.arange(len(ys2)), ys2, '-o', label = '$n=0.75$')
ax.semilogy(np.arange(len(ys3)), ys3, '-o', label = '$n=2.0$')
ax.semilogy(np.arange(len(ys4)), ys4, '-o', label = '$n=3.0$')
ax.semilogy(np.arange(len(ys5)), ys5, '-o', label = '$n=5.0$')

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.485, -0.05),
          fancybox=True, shadow=False, ncol=5)

#plt.title('Convergence for different $n$', style = 'italic')
plt.ylabel('$||g_k||$', style = 'italic')
plt.xlabel('iteration k', style = 'italic')
plt.ylim(1e-12,1e2)
plt.xlim(-1,25)
plt.grid(True)
#plt.gca().set_aspect('equal', adjustable='box')

fig.savefig('convergence.pdf',orientation='landscape')

plt.close()