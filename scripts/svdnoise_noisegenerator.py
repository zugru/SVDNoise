# coding: utf-8

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import ticker as ticker
from scipy.stats import norm
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline, PchipInterpolator
from scipy.optimize import brentq
from svdnoise_pdfmap import *

class NoiseGenerator:
    """
    The class takes threshold, sigma and width as input and generates
    pdf, cdf, margins, partial copula and approximation splines to provide pdf norm and random variates from the pdf. Unlike NoiseFitter, no fits are made. The class is meant to provide data for a machine learning engine.
    """
    def __init__(self, t0_range = (-250, 150, 2), amp_range = (0.0, 8.0, 0.05)):
        """This just constructs the basic t0/amplitude grid."""
        self.t0_range = t0_range
        self.amp_range = amp_range
        self.t0s, self.amplitudes = np.mgrid[\
            self.t0_range[0]:self.t0_range[1]:self.t0_range[2],\
            self.amp_range[0]:self.amp_range[1]:self.amp_range[2]\
            ]
        self.make_plots = True
        self.plotdir = ""

    def generate_pdf(self, threshold, sigma, width, n_samples = 6):
        """ Takes basically threshold, sigma, width as input.
        Number of APV samples and grids for t0 and amplitude can be
        specified as well.
        """
        self.threshold = threshold
        self.sigma = sigma
        self.width = width
        self.n_samples = n_samples

        # Generate signals
        s = np.empty(self.t0s.shape + (self.n_samples,))
        for i in range(self.n_samples):
            s[:,:,i] = self.amplitudes *\
                betaprime_wave(-self.t0s + (i-1)*apv_dt, self.width)
        # Calculate pdf
        self.pdf = (norm.cdf(self.threshold/self.sigma)**self.n_samples - np.product(norm.cdf((self.threshold - s)/self.sigma), axis = 2))*norm.pdf(self.amplitudes)
        self.P_xy = RectBivariateSpline(self.t0s[:,-1], self.amplitudes[-1,:], self.pdf)
        # Calculate cdf
        dx, dy = self.t0s[1,0] - self.t0s[0,0], self.amplitudes[0,1] - self.amplitudes[0,0]
        self.cdf = self.pdf.copy()
        self.cdf[1:,:] += self.pdf[:-1,:]
        self.cdf[:,1:] += self.cdf[:,:-1]
        self.cdf = np.cumsum(np.cumsum(self.cdf, axis = 0), axis = 1)*dx*dy
        cdf_norm = self.cdf[-1,-1]
        self.cdf /= cdf_norm
        # Calculate pdf margins and precise norms
        self.pdfu = np.apply_along_axis(simps,1,self.pdf,*[self.amplitudes[-1,:]])
        u_norm = simps(self.pdfu, self.t0s[:,-1])
        self.pdfu /= u_norm
        self.pdfv = np.apply_along_axis(simps,0,self.pdf,*[self.t0s[:,-1]])
        v_norm = simps(self.pdfv, self.amplitudes[-1,:])
        self.pdfv = self.pdfv / v_norm
        self.norm = u_norm
        if abs(u_norm - v_norm) > 1.0e-10:
            print("ERROR in normalization!")
            print("Norms:\ncdf:\t{0}\nu:\t{1}\nv:\t{2}".format(cdf_norm, u_norm, v_norm))
        # Construct copula
        increasing_u = self._increasing_subsequence(self.cdf[:,-1])
        self.u_to_t0 = PchipInterpolator(self.cdf[increasing_u,-1], self.t0s[increasing_u,-1], extrapolate = True)
        increasing_v = self._increasing_subsequence(self.cdf[-1,:])
        self.v_to_amp = PchipInterpolator(self.cdf[-1,increasing_v], self.amplitudes[-1,increasing_v], extrapolate = True)
        F_xy = RectBivariateSpline(self.t0s[:,-1], self.amplitudes[-1,:], self.cdf)
        self.u, self.v = np.mgrid[0.005:1.0:0.01, 0.005:1.0:0.01]
        self.t0_q = self.u_to_t0(self.u)
        self.amp_q = self.v_to_amp(self.v)
        self.copula = F_xy(self.t0_q[:,0], self.amp_q[0,:])
        # Calculate reconstructed density
        # Calculate the derivative copula
        F_uv = RectBivariateSpline(self.u[:,0], self.v[0,:], self.copula, bbox = (0.0, 1.0, 0.0, 1.0))
        self.dcopula = F_uv(self.u[:,0], self.v[0,:], dx = 1, dy = 0)
        dcopula_bottom = F_uv(self.u[:,0], 0, dx = 1, dy = 0)
        dcopula_top = F_uv(self.u[:,0], 1.0, dx = 1, dy = 0)
        self.dcopula = (self.dcopula - dcopula_bottom)/(dcopula_top - dcopula_bottom)
        # Make exact dcopula inverter:
        self.dcopula_spline = RectBivariateSpline(self.u[:,0], self.v[0,:], self.dcopula, bbox = (0.0, 1.0, 0.0, 1.0))

    @staticmethod
    def _increasing_subsequence(seq):
        """Return indices into input array such that the resulting subsequence is strictly increasing."""
        if seq is None:
            return seq
        M = [None] * len(seq)    # offset by 1 (j -> j-1)
        P = [None] * len(seq)
        # Since we have at least one element in our list, we can start by
        # knowing that the there's at least an increasing subsequence of length one:the first element.
        L = 1
        M[0] = 0
        # Looping over the sequence starting from the second element
        for i in range(1, len(seq)):
            # Binary search: we want the largest j <= L
            #  such that seq[M[j]] < seq[i] (default j = 0),
            #  hence we want the lower bound at the end of the search process.
            lower = 0
            upper = L
            # Since the binary search will not look at the upper bound value,
            # we'll have to check that manually
            if seq[M[upper-1]] < seq[i]:
                j = upper
            else:
                # actual binary search loop
                while upper - lower > 1:
                    mid = (upper + lower) // 2
                    if seq[M[mid-1]] < seq[i]:
                        lower = mid
                    else:
                        upper = mid
                j = lower    # this will also set the default value to 0
            P[i] = M[j-1]
            if j == L or seq[i] < seq[M[j]]:
                M[j] = i
                L = max(L, j+1)
        # Building the result: [seq[M[L-1]], seq[P[M[L-1]]], seq[P[P[M[L-1]]]], ...]
        result = []
        pos = M[L-1]
        for _ in range(L):
            result.append(pos)
            pos = P[pos]
        return result[::-1]    # reversing

    def reconstruct_pdf(self):
        """Re-calculate pdf on a quantile-based mesh"""
        # Generate signals
        s = np.empty(self.t0_q.shape + (self.n_samples,))
        for i in range(self.n_samples):
            s[:,:,i] = self.amp_q *\
                betaprime_wave(-self.t0_q + (i-1)*apv_dt, self.width)
        # Calculate pdf
        self.pdf_q = (norm.cdf(self.threshold/self.sigma)**self.n_samples - np.product(norm.cdf((self.threshold - s)/self.sigma), axis = 2))*norm.pdf(self.amp_q)

    def uv_transform(self, u, v):
        """Transform (u,v) to (u,v')."""
        vprime = np.zeros_like(v)
        for i, av in enumerate(v):
            vprime[i] = brentq(lambda vp: self.dcopula_spline(u[i], vp) - av, 0, 1)
        return {'u':u, 'vprime':vprime}

    def random_transform(self, u, v):
        """Transform (u,v) pair from (0,1)**2 to the corresponding (t0, amp) pair using only interpolation and no fits. u and v are expected to be 1d and same size."""
        vprime = np.zeros_like(v)
        for i, av in enumerate(v):
            vprime[i] = brentq(lambda vp: self.dcopula_spline(u[i], vp) - av, 0, 1)
        amp = self.v_to_amp(vprime)
        t0 = self.u_to_t0(u)
        return {'t0' : t0, 'amplitude' : amp, 'vprime' : vprime}

    @staticmethod
    def _fmt(x, pos):
        """Helper function to format colorbar ticks"""
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    def plot(self):
        """This makes a vignette with P_over pdf/cdf distribution,
        margin pdfs and the partial copula."""
        fig = plt.figure(figsize = (12, 12))
        # pdf and cdf go to bottom right, margins on appropriate sides
        ax1 = plt.subplot2grid((12,12),(4,0), colspan = 9, rowspan = 8)
        pdf_map = ax1.contourf(self.t0s, self.amplitudes, self.pdf, 10, cmap = 'Blues')
        #ax1.contour(self.t0_q, self.amp_q, self.pdf_rec, 10, cmap = 'Reds', linewidth = 0.5)
        ax1.set_title('Probability of at least one of {0} samples over threshold'.format(self.n_samples))
        ax1.set_xlabel('t0 [ns]')
        ax1.set_ylabel('amplitude [S/N]')
        ax1c = plt.subplot2grid((12,12), (1,9), rowspan = 3, colspan = 2)
        plt.colorbar(pdf_map, cax = ax1c, format = ticker.FuncFormatter(self._fmt))
        ax2 = plt.subplot2grid((12,12),(1,0), colspan = 9, rowspan = 3, sharex = ax1)
        ax2.plot(self.t0s[:,-1], self.pdfu, label = 'data')
        ax2.set_title('t0 margin distribution')
        ax2.set_ylabel('P(1 over)')
        plt.setp(ax2.get_xticklabels(), visible = False)
        ax3 = plt.subplot2grid((12,12),(4,9), rowspan = 8, colspan = 3, sharey = ax1)
        ax3.plot(self.pdfv, self.amplitudes[-1,:], label = 'data')
        ax3.set_title('Amplitude margin distribution')
        ax3.set_xlabel('P(1 over)')
        plt.setp(ax3.get_yticklabels(), visible = False)
        ax4 = plt.subplot2grid((12,12),(0,0), colspan = 9)
        ax4.text(0.5, 1.0, 'Probability of at least one sample over threshold\nthreshold : {0}, sigma : {1}, width : {2}'.format(self.threshold, self.sigma, self.width), horizontalalignment = 'center', verticalalignment = 'top', fontsize = 18)
        ax4.set_axis_off()
        plt.tight_layout()
        plt.savefig('{0}/margin_fits_{1}s_thr{2}_sig{3}_w{4}.png'.format(self.plotdir, self.n_samples, self.threshold, self.sigma, self.width))

    def plot_rng(self, n_random_samples = 10000):
        """ Plot P_1over pdf and some random samples over it."""
        u = np.random.uniform(0,1,n_random_samples)
        v = np.random.uniform(0,1,n_random_samples)
        samples = self.random_transform(u, v)
        fig = plt.figure(figsize = (12, 12))
        # pdf and random samples go to bottom right, margins on appropriate sides
        ax1 = plt.subplot2grid((12,12),(4,0), colspan = 9, rowspan = 8)
        pdf_map = ax1.contourf(self.t0s, self.amplitudes, self.pdf, 10, cmap = 'Blues')
        ax1.scatter(samples['t0'], samples['amplitude'], s = 0.03, c = 'y')
        ax1.set_title('{0} random samples from P_over distribution'.format(n_random_samples))
        ax1.set_xlabel('t0 [ns]')
        ax1.set_ylabel('amplitude [S/N]')
        ax1c = plt.subplot2grid((12,12), (1,9), rowspan = 3, colspan = 2)
        plt.colorbar(pdf_map, cax = ax1c, format = ticker.FuncFormatter(self._fmt))
        ax2 = plt.subplot2grid((12,12),(1,0), colspan = 9, rowspan = 3, sharex = ax1)
        ax2.plot(self.t0s[:,-1], self.pdfu)
        ax2.hist(samples['t0'], bins = self.t0s[:,0], normed = True)
        ax2.set_title('t0 margin distribution')
        ax2.set_ylabel('P(1 over)')
        plt.setp(ax2.get_xticklabels(), visible = False)
        ax3 = plt.subplot2grid((12,12),(4,9), rowspan = 8, colspan = 3, sharey = ax1)
        ax3.plot(self.pdfv, self.amplitudes[-1,:])
        ax3.hist(samples['amplitude'], bins = self.amplitudes[0,:], normed = True, orientation = 'horizontal')
        ax3.set_title('Amplitude margin distribution')
        ax3.set_xlabel('P(1 over)')
        plt.setp(ax3.get_yticklabels(), visible = False)
        ax4 = plt.subplot2grid((12,12),(0,0), colspan = 9)
        ax4.text(0.5, 1.0, '{0} random samples from P(one over) distribution\nthreshold : {1}, sigma : {2}, width : {3}'.format(n_random_samples, self.threshold, self.sigma, self.width), horizontalalignment = 'center', verticalalignment = 'top', fontsize = 18)
        ax4.set_axis_off()
        plt.tight_layout()
        plt.savefig('{0}/rng_test_{1}s_thr{2}_sig{3}_w{4}_n{5}.png'.format(self.plotdir, self.n_samples, self.threshold, self.sigma, self.width, n_random_samples))

    def plot_uv(self):
        fig = plt.figure(figsize = (12,6))
        plot_u, plot_v = np.mgrid[0.005:0.995:0.01, 0.005:0.995:0.01]
        samples = self.uv_transform(plot_u.flatten(), plot_v.flatten())
        plot_vprime = samples['vprime'].reshape(plot_v.shape)
        plt.title("Half-copula: v' as function of u and v\nthreshold : {0}, sigma : {1}, width : {2}".format(self.threshold, self.sigma, self.width))
        plt.contourf(plot_u, plot_v, plot_vprime, levels = np.arange(-0.05, 1.05, 0.05), cmap = 'Blues')
        plt.xlabel('u')
        plt.ylabel('v')
        plt.colorbar()
        plt.savefig('{0}/half_copula_thr{1}_sig{2}_w{3}.png'.format(self.plotdir, self.threshold, self.sigma, self.width))

    def set_make_plots(self, option = True):
        self.make_plots = option

    def set_plotdir(self, plotdir):
        self.plotdir = plotdir

    def get_norm(self):
        return self.norm

    def get_pdf(self, u, v):
        return self.P_xy(u,v)

if (__name__ == '__main__'):
    #%matplotlib inline
    print('NoiseGenerator demo:')
    generator = NoiseGenerator()
    generator.set_plotdir('../pictures/diags')
    threshold = 3.0
    sigma = 0.1
    width = 200
    n_samples = 6
    print('Fitting noise pdf, threshold = {0}, sigma = {1}, width = {2}...'.format(threshold, sigma, width))
    generator.generate_pdf(threshold, sigma, width, n_samples)
    #generator.reconstruct_pdf()
    print('Some plots...')
    generator.plot()
    generator.plot_rng()
    generator.plot_uv()
