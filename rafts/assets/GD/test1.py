import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_line(num, data, line, img):
    line.set_data(data[...,:num])
    if num == 24:
        img.set_visible(True)
    return line, img

fig1 = plt.figure()

data = np.random.rand(2, 25)
ax1=plt.subplot(211)
l, = plt.plot([], [], 'rx')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')
ax2=plt.subplot(212)
nhist, xedges, yedges = np.histogram2d(data[0,:], data[1,:])
img = plt.imshow(nhist, aspect='auto', origin='lower')
img.set_visible(False)
line_ani = animation.FuncAnimation(fig1, update_line, 25, 
                                   fargs=(data, l, img),
                                   interval=50, blit=True)
line_ani.repeat = False
plt.show()