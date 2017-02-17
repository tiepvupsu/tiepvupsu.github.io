# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np 

import matplotlib.pyplot as plt

x0 = np.linspace(0.001, 0.999, 1000)




def ce(p, q):
	return -(p*np.log(q) + (1-p)*np.log(1 - q))

def dist(p, q):
	return (p - q)**2 

fig = plt.figure()
# fig = plt.figure(num=None, figsize=(4, 4), dpi=300)
# fig = plt.gcf()
# fig.set_size_inches(7, 10.5)

# fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')



# fig.suptitle('Example', fontsize=14, fontweight='bold')
ax = fig.add_subplot(131)
# y0 = -(.5*np.log(x0) + .5*np.log(1 - x0))
p = .5 
y0 = ce(p, x0)
plt.plot(x0, y0, 'b-')
ax.text(0.2, 1.1, r'$-(0.5\log(q) + 0.5\log(1 - q)$', fontsize=14, color = 'blue')

z0 = (p - x0)**2
plt.plot(x0, z0, 'r-')

plt.axis([0, 1, -.5, 6])
ax.text(0.2, .3, r'$(q - 0.5)^2$', fontsize=14, color = 'red')
ax.set_title("$p = 0.5$")
plt.plot(p, ce(p, p), 'go', markersize= 5)
plt.plot(p, dist(p, p), 'go', markersize= 5)

#######################
ax = fig.add_subplot(132)
# y0 = -(.5*np.log(x0) + .5*np.log(1 - x0))
p = .1
y0 = ce(p, x0)
plt.plot(x0, y0, 'b-')
ax.text(0.1, 1.3, r'$-(0.1\log(q) + 0.9\log(1 - q)$', fontsize=14, color = 'blue')

z0 = (p - x0)**2
plt.plot(x0, z0, 'r-')

plt.axis([0, 1, -.5, 6])
ax.text(0.5, .4, r'$(q - 0.1)^2$', fontsize=14, color = 'red')
ax.set_title("$p = 0.1$")
plt.plot(p, ce(p, p), 'go', markersize= 5)
plt.plot(p, dist(p, p), 'go', markersize= 5)



#######################
ax = fig.add_subplot(133)
# y0 = -(.5*np.log(x0) + .5*np.log(1 - x0))
p = .8
y0 = ce(p, x0)
plt.plot(x0, y0, 'b-')
ax.text(0.3, 1.2, r'$-(0.8\log(q) + 0.2\log(1 - q)$', fontsize=14, color = 'blue')

z0 = (p - x0)**2
plt.plot(x0, z0, 'r-')

plt.axis([0, 1, -.5, 6])
ax.text(0.2, .5, r'$(q - 0.8)^2$', fontsize=14, color = 'red')
ax.set_title("$p = 0.8$")
plt.plot(p, ce(p, p), 'go', markersize= 5)
plt.plot(p, dist(p, p), 'go', markersize= 5)
plt.tight_layout()
plt.show()
plt.savefig('crossentropy.png', bbox_inches='tight', dpi = 300)