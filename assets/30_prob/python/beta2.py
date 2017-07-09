
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.stats import beta
fig, ax = plt.subplots(1, 1)

with PdfPages('beta1.pdf') as pdf:
    a, b = .5, 1.5
    x = np.linspace(beta.ppf(0.00, a, b), beta.ppf(1, a, b), 100)
    ax.plot(x, beta.pdf(x, a, b),'r-', lw=2, alpha=1, label='beta pdf')

    a, b = 4, 12
    x = np.linspace(beta.ppf(0., a, b), beta.ppf(1, a, b), 100)
    ax.plot(x, beta.pdf(x, a, b),'k-', lw=2, alpha=1, label='beta pdf')


    a, b = 1, 3
    x = np.linspace(beta.ppf(0.05, a, b), beta.ppf(0.95, a, b), 100)
    ax.plot(x, beta.pdf(x, a, b),'b-', lw=2, alpha=1, label='beta pdf')

    a, b = 2, 6
    x = np.linspace(beta.ppf(0.00, a, b), beta.ppf(1, a, b), 1000)
    ax.plot(x, beta.pdf(x, a, b),'c-', lw=2, alpha=1, label='beta pdf')

    a, b = 0.25, .75
    x = np.linspace(beta.ppf(0.3, a, b), beta.ppf(.96, a, b), 100)
    ax.plot(x, beta.pdf(x, a, b),'g-', lw=2, alpha=0.6, label='beta pdf')

    plt.text(0.1, .12, r'(0.25, 0.75)', fontsize=12, color = 'green')
    plt.text(0.2, 1.94, r'(1, 3)', fontsize=12, color = 'blue')
    plt.text(0.2, 3.4, r'(4, 12)', fontsize=12, color = 'black')
    plt.text(0.1, 0.92, r'(.5, 1.5)', fontsize=12, color = 'red')
    plt.text(0.2, 2.84, r'(2, 6)', fontsize=12, color = 'c')

    ax.set_yticks([])
    plt.xlim([0,1])
    plt.ylim([0,6])
    ax.set_yticks([])
#     plt.axis('equal')
    plt.xlabel('$\lambda$', fontsize = 15)
    plt.ylabel('$(\lambda)$', fontsize = 15)

    # pdf.savefig()
    plt.show()