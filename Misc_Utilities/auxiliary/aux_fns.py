import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as spopt
import scipy.linalg
import scipy.stats as spstat
from auxiliary.utils import convert_tuple
from numba import njit
import numba

# conda install -c plotly plotly=4.8.2
# conda install -c plotly plotly-orca==1.2.1 psutil requests
import plotly.graph_objects as go
from plotly.offline import iplot
from plotly.subplots import make_subplots

"""
This file contains several functions that are useful when creating plots based on 
models:

    write_to_file: writes data (list of lists) to a csv file

    transp: transposes a list of lists

    sort: sorts a list

    make_plot: creates a series of plots based on a dict of dicts. Designed to plot macro variables with possibly multiple lines each

    equalize_lengths: takes a dict and makes sure all items have equal length by possibly duplicating the last entries

    plot_IRFs: plots IRFs of one or multiple models for one or multiple outcome variables

    comp_statics: runs IRFs for various parameters but then saves specific statistics from those runs

"""

mydir_source = 'Graphs/Source/'
mydir_tikz = 'Graphs/Tikz/'
mydir_png = 'output/'


def write_to_file(filename, data, write_csv=True, sort_trans=False):
    if sort_trans:
        data = transp(data)  # transpose
    if write_csv:
        # write to csv
        # in that case, data is best an array
        with open(mydir_source + filename + '.txt', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(data)
    else:
        # just write any string to file (data is a string!)
        file = open(mydir_source + filename + '.txt', 'w')
        file.write(data)
        file.close()
        # print('Written to ' + filename + '.txt !')


def transp(mylist):
    return list(map(list, zip(*mylist)))


def sort(mylist):
    return list(zip(*sorted(zip(*mylist))))


###


# % % Prepare for IRFs
def make_plot2(filename, plot_data, plot_models_legend, plot_fields, T_plot,
                       title=None, colors=None, show=False, print_png=True, field_percent=[],
                       tickformat=',.1', title_font_size=22, font_size=18, legend_font_size=16,
                       legend_index=1, subtitlegap=0.01, legend_y=0.99, legend_x=0.01, legend_orient='v', scale=1):

    """legend_index : which subplot gets legend. if none ==> put legend at bottom
    plot_data : dic of fields, and in each field, models
    plot_fields: of the form 'field id': 'field label' """
    # sanity checks
    assert type(filename) is str
    assert type(plot_data) is dict
    assert type(plot_fields) is dict, 'make_plot found type error'

    # dimensions of the figure panels
    no_series = len(plot_data)
    if no_series == 4:
        n_horiz = 4
        n_vert = 1
    else:
        n_horiz = int(np.ceil(np.sqrt(no_series)))
        n_vert = int(np.ceil(no_series / n_horiz))

    # initialize figure
    fig = make_subplots(rows=n_vert, cols=n_horiz,
                        subplot_titles=tuple(plot_fields.values()),
                        horizontal_spacing=0.06 * 4/n_horiz,
                        vertical_spacing=1/n_vert/3)

    # figure setup
    lines = [None, 'dash', None, 'dot']
    if colors is None:
        colors = ['black', 'firebrick', 'rgba(105,105,105,0.5)', 'rgba(105,105,105,0.5)']
    if title is None:
        title = ''

    # make figure
    i = 0
    for field in plot_fields.keys():
        i += 1
        r = int(np.ceil(i / n_horiz))
        c = int(np.mod(i - 1, n_horiz) + 1)

        if i == legend_index:  # display legend for only the first subplot
            legend_display = True
        else:
            legend_display = False

        j = 0
        for model in plot_data[field].keys():

            data = plot_data[field][model] * scale
            if field in field_percent:
                data /= 100  # need to scale down because units scale it up again

            fig.add_trace({
                'x': np.linspace(0, T_plot, T_plot + 1),
                'y': data,
                'line': dict(color=colors[j], dash=lines[j]),
                'name': plot_models_legend[model],
                'legendgroup': model,
                'showlegend': legend_display,
            }, row=r, col=c)
            j += 1

        ## update subplot layout
        fig.update_xaxes(ticks='outside', showline=True, linecolor='black', mirror=True, row=r, col=c)
        fig.update_yaxes(ticks='outside', showline=True, linecolor='black', mirror=True, row=r, col=c)
        fig['layout']['annotations'][i-1]['font']=dict(family="Computer Modern Roman", size=title_font_size)

        # if r == n_vert:
        #     fig.update_xaxes(title_text=xlabel, row=r, col=c)

        # adjust subplot title position
        fig.layout.annotations[i-1].update(y=fig.layout.annotations[i-1]['y'] + subtitlegap)

        if field in field_percent:
            fig.update_yaxes(tickformat=tickformat + '%', row=r, col=c)
        else:
            fig.update_yaxes(tickformat=tickformat + 'f', row=r, col=c)

        # if not y_annot == None:
        #     fig['layout']['annotations'][i - 1]['y'] = y_annot[i - 1]
        # if not x_annot == None:
        #     fig['layout']['annotations'][i - 1]['x'] = x_annot[i - 1]
        # # fig['layout']['annotations'][i - 1]['font'] = dict(size=15)
        # # fig.layout.annotations[i-1].update(y=fig.layout.annotations[i-1]['y'] + 0.01)

    fig.update_layout(
        height=250 * n_vert + 80,
        width=250 * n_horiz + 150 + 60,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        title='',
        title_x=0.5,
        yaxis1=dict(),
        font=dict(family="Computer Modern Roman", size=font_size),
        title_font=dict(family="Computer Modern Roman", size=title_font_size),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    if legend_index is not None:
        fig.update_layout(legend=dict(
            yanchor="auto",
            y=legend_y,
            x=legend_x,
            xanchor="auto",
            orientation=legend_orient,
            bgcolor='rgba(0,0,0,0)'
        ))
    else:
        fig.update_layout(legend=dict(font=dict(family="Computer Modern Roman", size=legend_font_size),
                                      orientation="h", xanchor='center',
                                      x=0.5, yanchor='top', bgcolor = 'rgba(0,0,0,0)'))

    # export figure
    if show:
        iplot(fig)
    if print_png:
        fig.write_image(mydir_png + filename + '.png', scale=2)


def make_plot(filename, title, x, out, print_png=True, Tplot=None, colors=None, show=True, legend=True, ncol=2):
    assert type(filename) is str
    assert type(title) is str
    assert type(x) is np.ndarray
    assert type(out) is dict, 'make_plot found type error'
    # shape of out:
    # e.g. out = {'c':{'high psi':[0,1,2,3],'low psi':[1,2,3,4]},'a':...} etc
    no_series = len(out)

    # turn off automatic plots
    plt.ioff()

    # create new figure
    fg = plt.figure()
    fg.suptitle(title, y=1)

    # dimensions of the figure panels
    n_horiz = np.ceil(np.sqrt(no_series))
    n_vert = np.ceil(no_series / n_horiz)

    i = 0
    fgs = {}
    axs = {}
    for var, paths in out.items():
        i += 1

        # create a separate plot
        fgs[i] = plt.figure()
        axs[i] = fgs[i].add_subplot(1, 1, 1)

        ax = fg.add_subplot(n_vert, n_horiz, i)
        ax.set_title(var)

        # store data
        data = [x]

        for label, path in paths.items():

            kwargs = {}
            if colors is not None:
                kwargs.update({'color': colors[label]})

            temp = [x, path]
            #            temp = sort(temp)
            x = temp[0]
            path = temp[1]

            # ensure both have same size
            Tmin = min(len(x), len(path))
            x_here = x[:Tmin]
            path = path[:Tmin]

            # plot on figures 0 and i
            ax.plot(x_here, path, label=label, **kwargs)
            if Tplot is not None:
                ax.set_xlim(left=0, right=Tplot)
            axs[i].plot(x_here, path, label=label, **kwargs)
            data.append(path)
        if i == no_series and legend:  # n_horiz:
            ax.legend(loc=8, bbox_to_anchor=(0.5, -0.1), ncol=ncol)
        fg.tight_layout()

        # plot to png file
        cur_filename = filename + '_' + str(var)
        if print_png:
            #            fgs[i].legend(loc=2,bbox_to_anchor=(1.05, 1))
            if str(var) == 'i':
                cur_filename = cur_filename + 'i'
            fgs[i].savefig(mydir_png + cur_filename + '.png')
            # print('Saved to ' + cur_filename + '.png !')

        data = sort(data)
        data = transp(data)  # transpose
        write_to_file(cur_filename, data)

        plt.close(fgs[i])
    if show:
        plt.show()
    else:
        plt.close()
    return None


def make_fill_plot(filename, xaxis, Ds, labels_sh=None, labels_se=None, colors=None, extra_line=None, vertlines=[],
                   show=True,
                   **kwargs):
    # prepare:
    if labels_sh is None:
        labels_sh = [k for k in range(Ds.shape[1])]
    if colors is None:
        colors = {labels_sh[i]: f'C{i}' for i in range(len(labels_sh))}
    T, len_se, len_sh = Ds.shape

    for i_se in range(len_se):
        b = []

        y_offset_pos = np.zeros_like(Ds[:, i_se, 0])  # positive offset, collecting positive terms
        y_offset_neg = np.zeros_like(Ds[:, i_se, 0])  # negative offset, collecting negative terms

        # start storing data to save as txt
        data = [xaxis]

        for i_sh in range(len_sh):
            sh = labels_sh[i_sh]
            Ds_here = Ds[:, i_se, i_sh]
            y_offset = (Ds_here > 0) * y_offset_pos + (Ds_here < 0) * y_offset_neg
            # b.append(plt.bar(xaxis,Ds_here,bottom=y_offset,width=0.24))
            y_offset_pos_ = y_offset_pos + np.maximum(Ds_here, 0)
            y_offset_neg_ = y_offset_neg - np.maximum(-Ds_here, 0)
            plt.fill_between(xaxis, y_offset_pos, y_offset_pos_, color=colors[sh], label=sh, **kwargs)
            plt.fill_between(xaxis, y_offset_neg, y_offset_neg_, color=colors[sh], **kwargs)
            y_offset_pos = y_offset_pos_
            y_offset_neg = y_offset_neg_
            data += [y_offset_pos, y_offset_neg]
        # plt.legend((k[0] for k in b), shocks_list)
        plt.legend()
        plt.title(labels_se[i_se])

        # add extra line?
        if extra_line is not None:
            extra_line_kwargs = {k: extra_line[k] for k in extra_line if k != 'y'}
            plt.plot(xaxis, extra_line['y'][:, i_se], **extra_line_kwargs)
            data.extend([extra_line['y'][:, i_se]])

        cur_filename = filename + '_' + labels_se[i_se]

        # somehow the 'i' label overwrites the I label file??? # TODO FIX THIS
        if labels_se[i_se] == 'i':
            cur_filename = cur_filename + 'i'

        # add vertical line?
        for x in vertlines:
            plt.axvline(x=x, color='black', linestyle='--')

        # save figure
        plt.savefig(mydir_png + cur_filename + '.png', format='png', dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

        # write to txt file
        data = sort(data)
        data = transp(data)  # transpose
        write_to_file(cur_filename, data)


def equalize_lengths(mydict):
    def lenlist(mylist):
        assert type(mylist) is list or np.ndarray, 'Parameter lists must be lists, even shocks!'
        return len(mylist)

    lengths = [lenlist(mydict[key]) for key in mydict.keys()]
    N = np.max(lengths)
    for key, val in mydict.items():
        mydict[key] = val + (N - len(mydict[key])) * [val[-1]]
    return (mydict, N)


def shock_block(block, i, t=None, shock=None, Tplot=20, T=100, out=None, plot=True, plotname='block_shock', **kwargs):
    """ Function allows to shock an individual block directly"""
    T = max(Tplot, T)
    calib_here = block.ssin
    calib_here.update({**kwargs})
    block.eval_ss(calib_here)

    # what is the shock?
    if t is not None:
        shock = np.arange(T) == t

    # compute the TD
    if out is None:
        td = block.eval_td({i: shock})
    else:
        td = {o: block.eval_td({i: shock})[o] for o in out}

    # plot?
    if plot:
        make_plot(plotname, plotname, np.arange(Tplot), {'a': {o: td[o][:Tplot] for o in td}})
    return td


@njit
def stacked_det(A):
    """ This function computes complex determinant of stack of 3 x 3 matrices. """
    n, m = A.shape[:2]
    assert n == m and n <= 3

    if n == 1:
        det = A[0, 0, ...]
        return det
    elif n == 2:
        det = A[0, 0, ...] * A[1, 1, ...] - A[0, 1, ...] * A[1, 0, ...]
        return det
    elif n == 3:
        det = A[0, 0, ...] * A[1, 1, ...] * A[2, 2, ...] + A[0, 1, ...] * A[1, 2, ...] * A[2, 0, ...] + A[0, 2, ...] * \
              A[1, 0, ...] * A[2, 1, ...] \
              - A[0, 0, ...] * A[1, 2, ...] * A[2, 1, ...] - A[0, 1, ...] * A[1, 0, ...] * A[2, 2, ...] - A[0, 2, ...] * \
              A[1, 1, ...] * A[2, 0, ...]
        return det


@njit
def stacked_adjunct(A):
    """ This function computes the transpose of the inverses of stacked 3 x 3 matrices. """
    """ Right now, only works for 3-dimensional array"""
    """ Somehow the transpose() function below didnt work"""
    n, m, N = A.shape
    assert n == m and n <= 3
    if n == 1:
        return np.ones_like(A)
    elif n == 2:
        inv = np.empty_like(A)
        inv[0, 0, ...] = A[1, 1, ...]  # A[0,0]*A[1,1] - A[0,1]*A[1,0]
        inv[0, 1, ...] = -A[0, 1, ...]
        inv[1, 0, ...] = -A[1, 0, ...]
        inv[1, 1, ...] = A[0, 0, ...]
        # dets2 = stacked_det(A)
        return inv
    elif n == 3:
        adj = np.empty_like(A)
        adj[0, 0, ...] = A[1, 1, ...] * A[2, 2, ...] - A[1, 2, ...] * A[2, 1, ...]  # 1,2 & 1,2
        adj[0, 1, ...] = - A[1, 0, ...] * A[2, 2, ...] + A[1, 2, ...] * A[2, 0, ...]  # 1,2 & 0,2
        adj[0, 2, ...] = A[1, 0, ...] * A[2, 1, ...] - A[1, 1, ...] * A[2, 0, ...]  # 1,2 & 0,1
        adj[1, 0, ...] = - A[0, 1, ...] * A[2, 2, ...] + A[0, 2, ...] * A[2, 1, ...]  # 0,2 & 1,2
        adj[1, 1, ...] = A[0, 0, ...] * A[2, 2, ...] - A[0, 2, ...] * A[2, 0, ...]  # 0,2 & 0,2
        adj[1, 2, ...] = - A[0, 0, ...] * A[2, 1, ...] + A[0, 1, ...] * A[2, 0, ...]  # 0,2 & 0,1
        adj[2, 0, ...] = A[0, 1, ...] * A[1, 2, ...] - A[0, 2, ...] * A[1, 1, ...]  # 0,1 & 1,2
        adj[2, 1, ...] = - A[0, 0, ...] * A[1, 2, ...] + A[0, 2, ...] * A[1, 0, ...]  # 0,1 & 0,2
        adj[2, 2, ...] = A[0, 0, ...] * A[1, 1, ...] - A[0, 1, ...] * A[1, 0, ...]  # 0,1 & 0,1
        return adj


def determinacy_fast_compute_A(m_td=None, jac=None, T=600, unknowns=('y', 'w', 'r'),
                               targets=('asset_clearing', 'fisher', 'real_wage')):
    # Compute jacobian from model and read out A
    if jac is None:
        assert m_td is not None
        jac = m_td.own_jac(inputs=unknowns, outputs=targets, T=T)

    N = T + (T - 1)  # how many Aj's we'll have
    N_t = len(targets)
    N_u = len(unknowns)

    # Compute the Aj's, i.e. the blocks that make up the asymptotic column of the model
    A = np.empty((N_t, N_u, N))
    for i in range(N_t):
        for j in range(N_u):
            ta = targets[i]
            un = unknowns[j]
            # Truncate first to T x T (to get rid of boundary problems)
            A[i, j, :T] = jac[ta][un][0:T, T - 1]  # first take last column of jacobian (T elements, incl diagonal)
            A[i, j, T:] = np.flip(
                jac[ta][un][T - 1, 0: T - 1], axis=0)  # now take last row of jacobian (T-1 elements, excl diagonal)
    N_middle = T - 1  # A[N_middle, ...] is the block on the diagonal

    return A


def determinacy_fast(m_td=None, jac=None, A=None, T=600, unknowns=('y', 'w', 'r'),
                     targets=('asset_clearing', 'fisher', 'real_wage'), N_fft=1000, computeA0=False):
    if A is None:
        # Compute jacobian from model and read out A
        A = determinacy_fast_compute_A(m_td=m_td, jac=jac, T=T, unknowns=unknowns, targets=targets)

    N_t, N_u, N = A.shape
    assert N_t == N_u
    N_middle = (N + 1) / 2 - 1

    # 2. Second step: compute path integral
    # To do this, evaluate the argument of the integral for points on the top half unit circle (since real!)
    # A_t = np.fft.fftn(A,s=(N_fft,),axes=(2,))  # A(t) on a grid of t's
    # A_T_inv = stacked_adjunct(A_t)/stacked_det(A_t)
    # Aprime = np.empty_like(A)
    # Aprime[...,:N-1] = A[...,:N-1] * np.arange(N-1,0,-1)
    # Aprime[...,N-1] = 0
    # A_t_der = np.fft.fftn(Aprime,s=(N_fft,),axes=(2,))
    # # z_factor = np.exp(2 * np.pi * 1j * np.arange(N_fft) / N_fft)
    # # f_fft = np.einsum('ijk,ijk->k',A_T_inv,A_t_der)  #* z_factor
    #
    # # no_roots_inside = np.sum(A_T_inv*A_t_der)/N_fft
    # no_roots_inside = np.dot(A_T_inv.ravel(),A_t_der.ravel())/N_fft
    # winding_no = np.round(no_roots_inside) - N_t * N_middle

    A_t = np.fft.rfftn(A, s=(N_fft,), axes=(2,))  # A(t) on a grid of t's
    A_T_inv = stacked_adjunct(A_t) / stacked_det(A_t)
    Aprime = np.empty_like(A)
    Aprime[..., :N - 1] = A[..., :N - 1] * np.arange(N - 1, 0, -1)
    Aprime[..., N - 1] = 0
    A_t_der = np.fft.rfftn(Aprime, s=(N_fft,), axes=(2,))

    no_roots_inside = 2 * np.dot(A_T_inv.ravel(), A_t_der.ravel()).real
    no_roots_inside -= np.dot(A_T_inv[:, :, 0].ravel(), A_t_der[:, :, 0].ravel())
    if N_fft % 2 == 0:
        no_roots_inside -= np.dot(A_T_inv[:, :, -1].ravel(), A_t_der[:, :, -1].ravel())
    no_roots_inside /= N_fft

    winding_no = np.round(no_roots_inside) - N_t * N_middle

    if computeA0:
        A0_distance = np.real(np.linalg.det(A_t[:, :, 0]))
        return np.real(winding_no), A0_distance
    else:
        return np.real(winding_no)


def weed_out(A, tol_vec=None):
    """ Go through array A and kick out rows where all elements are too close to the previous one. """
    if tol_vec is None:
        tol_vec = 1e-4 * np.ones_like(A[0, ...])
    N = A.shape[0]
    B = A[0, ...][np.newaxis, ...]
    for n in range(N - 1):
        if np.any(np.abs(A[n + 1, ...] - A[n, ...]) > tol_vec):
            B = np.concatenate((B, A[n + 1, ...][np.newaxis, ...]), axis=0)
    return B


def determinacy_nd(m_td, filename='test', jac=None, T0=100, T1=500, T_jac=600, unknowns=('y', 'w', 'r'),
                   targets=('asset_clearing', 'fisher', 'real_wage'),
                   plot_winding=False, plot_vec=False, points_along_path=4, write_txt=False, N_path=10000, xlim=None,
                   ylim=None,
                   sparse_lam=True, sparse_lam_par=3, weed_par=1):
    """ This function implements the winding number criterion in n dimensions, as in Onatski. """
    # Set up
    T = T1 - T0
    N = T + (T - 1)  # how many Aj's we'll have
    N_t = len(targets)
    N_u = len(unknowns)

    # Compute jacobian from model
    if jac is None:
        jac = m_td.own_jac(inputs=unknowns, outputs=targets, T=T_jac)

    # Compute the Aj's, i.e. the blocks that make up the asymptotic column of the model
    A = np.empty((N_t, N_u, N))
    for i in range(N_t):
        for j in range(N_u):
            ta = targets[i]
            un = unknowns[j]
            # Truncate first to T x T (to get rid of boundary problems)
            A[i, j, :T] = jac[ta][un][T0:T0 + T,
                          T1 - 1]  # first take last column of jacobian (T elements, incl diagonal)
            A[i, j, T:] = np.flip(
                jac[ta][un][T1 - 1, T0:T1 - 1])  # now take last row of jacobian (T-1 elements, excl diagonal)
            # jac_here = jac[ta][un][T0:T1, T0:T1]
            # if n < T:  # first take last column of jacobian (T elements, incl diagonal)
            #     A[i, j, n] = jac_here[n, -1]
            # elif n >= T:  # now take last row of jacobian (T-1 elements, excl diagonal)
            #     A[i, j, n] = jac_here[-1, T-2-n]
    N_middle = T - 1  # A[N_middle, ...] is the block on the diagonal

    # Set up the characteristic function of the Aj's
    def A_lam_fn(lam):
        weights = np.exp(1j * lam * (np.arange(N) - N_middle))
        return np.linalg.det(A @ weights)  # np.einsum('i,ijk->jk', weights, A)

    # Compute points for integral
    eps = 1e-8  # so as winding number calc does not get confused below
    lams = np.linspace(eps, 2 * np.pi + eps, N_path)
    if sparse_lam:
        par = sparse_lam_par
        half = np.pi
        lams = np.tanh(par * (lams - half)) * half / np.tanh(par * half) + half
        # par = 0.2
        # lams = np.sign(lams-half)*(np.sign(lams-half)*(lams - half))**par * half**(1-par) + half

    A_lam = np.empty_like(lams, dtype=np.complex)
    for i in range(N_path):
        A_lam[i] = A_lam_fn(lams[i])
    A_lam_real = np.real(A_lam)
    A_lam_imag = np.imag(A_lam)

    # Compute winding number
    # Count number of times A_lam crosses ray from origin to real infinity
    up_crossings = np.sum(
        np.all((A_lam_imag[:-1] < 0, A_lam_imag[1:] > 0, A_lam_real[:-1] > 0, A_lam_real[1:] > 0), axis=0))
    down_crossings = np.sum(
        np.all((A_lam_imag[:-1] > 0, A_lam_imag[1:] < 0, A_lam_real[:-1] > 0, A_lam_real[1:] > 0), axis=0))
    winding_no = up_crossings - down_crossings

    # Store real part of A_lam(0) from 0 (often times a useful metric of determinacy)
    A0_distance = A_lam_real[0]

    # Plot ?
    if plot_winding:
        plt.plot(A_lam_real, A_lam_imag)
        plt.plot(0, 0, '*')
        for i in np.arange(points_along_path) / points_along_path:
            x = np.real(A_lam_fn(i * 2 * np.pi))
            y = np.imag(A_lam_fn(i * 2 * np.pi))
            plt.plot(x, y, '.')
            plt.annotate(f'{i}', (x, y))
            if ylim is not None:
                plt.ylim(ylim)
            if xlim is not None:
                plt.xlim(xlim)
        plt.title('Complex path associated with model asymptotics')
        plt.show()

    # write to txt file?
    if write_txt:
        # weed out observations that are too close to each other
        A_real_imag = np.concatenate((A_lam_real[:, np.newaxis], A_lam_imag[:, np.newaxis]), axis=1)
        A_real_imag = weed_out(A_real_imag, tol_vec=[weed_par, weed_par])
        data = [A_real_imag[:, 0], A_real_imag[:, 1]]
        # print(A_real_imag.shape)
        write_to_file(filename, data, sort_trans=True)
        np.save(mydir_source + filename + '_symbol.npy', A)

    # Plot A itself?
    if plot_vec:
        for i in range(N_t):
            for j in range(N_u):
                plt.plot(np.arange(N) - N_middle, A[i, j, :])
            plt.legend(unknowns)
            plt.title('Effects on ' + targets[i])
            plt.show()

    return winding_no, A0_distance


def determinacy(m, T=100, plot=False, target='goods_clearing', get_roots=False, use_col=True,
                fft_N=50000, plot_winding=False):
    # 1. Compute far out col or row vector
    r = m['ss'].ssout['r']
    Thalf = np.int(T / 2)
    if use_col:
        Y_shock = {'y': np.arange(T) == Thalf}
        td_Yshock = m['td'].solve_td(Y_shock, unknowns=('w', 'r'), targets=('fisher', 'real_wage'))
        net_demand_response = td_Yshock[target]
        det_assets = np.sum(td_Yshock['asset_clearing'])
    else:  # use row (doesn't work somehow...)
        m_Yshock = m['td'].solved(unknowns=('w', 'r'), targets=('fisher', 'real_wage'))
        jac = m_Yshock.own_jac(inputs=('y',), outputs=(target,), T=T)[target]['y']
        net_demand_response = np.flip(jac[Thalf, :])
        det_assets = 0
    net_demand_response_PV = net_demand_response * (1 + r) ** (-np.arange(T) + Thalf)
    det_PV = np.sum(net_demand_response_PV)  # to check that it's close to 0!

    # 2. Assuming convexity, the following works.
    det = np.sum(net_demand_response)
    determinate_convex = det > 0

    # 3. Not assuming convexity, compute FFT path integral along unit disk
    v = net_demand_response
    vprime = np.concatenate(([0], np.polyder(v, 1)))  # derivative with 0 padding at beginning
    z_factor = np.exp(2 * np.pi * 1j * np.arange(fft_N) / fft_N)  #
    f_fft = np.fft.fft(vprime, fft_N) / np.fft.fft(v, fft_N) * z_factor
    no_roots_inside = np.sum(f_fft) / fft_N
    determinate_general = np.round(no_roots_inside) < np.floor(v.size / 2) - 1
    if plot_winding:
        # Compute function v(lam) = sum_j  v_j e^(i j lam)
        # and plug in lambdas between 0 and 2pi
        def v_lam_fn(lam):
            return np.vdot(v, np.exp(1j * lam * (np.arange(T) - Thalf + 1)))

        lams = np.linspace(0, 2 * np.pi, 10000)
        v_lam = np.empty_like(lams, dtype=np.complex)
        for i in range(len(lams)):
            v_lam[i] = v_lam_fn(lams[i])
        v_lam_real = np.real(v_lam)
        v_lam_complex = np.imag(v_lam)
        plt.plot(v_lam_real, v_lam_complex)
        plt.plot(0, 0, '*')
        n_special_lams = 4
        for i in np.arange(n_special_lams) / n_special_lams:
            x = np.real(v_lam_fn(i * 2 * np.pi))
            y = np.imag(v_lam_fn(i * 2 * np.pi))
            plt.plot(x, y, '.')
            plt.annotate(f'{i}', (x, y))
        plt.title('Complex path associated with model asymptotics')
        plt.show()

    if get_roots:
        roots = np.roots(net_demand_response)
    else:
        roots = None
    if plot:
        make_plot('determinacy', 'determinacy', np.arange(T) - Thalf, {' ': {' ': net_demand_response}})
    return det, det_PV, det_assets, net_demand_response, roots, no_roots_inside, determinate_convex, determinate_general


def inspect_block_jac_G(title, jac, G, out, shock_var, shock_path, Tplot=20, inputs=None, plot=True):
    """
        Similar to inspect_block, but works with jacobians directly
        title:  string denoting block?
        jac:    dic of dic's with jacobians
        G:      G matrix input
        out:    outcome variable to split up
    """

    inputs_jac = list(jac[out].keys())
    td = {var: G[var][shock_var] @ shock_path for var in [out] + inputs_jac if var in G.keys()}
    td.update({var: np.zeros_like(shock_path) for var in [out] + inputs_jac if var not in G.keys()})
    td.update({shock_var: shock_path})

    if inputs is None:
        input_td = [i for i in td if i in inputs_jac]  # inputsall
    else:
        input_td = [i for i in inputs if i in inputs_jac and i in td]

    td_decomp = {'all': td[out]}
    td_decomp.update({i: jac[out][i] @ td[i] for i in input_td})
    if plot:
        make_plot(title, title, np.arange(Tplot), {out: {k: td_decomp[k][:Tplot] for k in td_decomp}})
    return td_decomp


def inspect_block_jac(title, jac, td, out, shock=None, Tplot=20, inputs=None, plot=True):
    """
        Similar to inspect_block, but works with jacobians directly
        title:  string denoting block?
        jac:    dic of dic's with jacobians
        td:     transitional dynamics being investigated
        out:    outcome variable to split up
    """

    if shock is not None:  # important to feed shock in !!
        td.update(shock)
    inputs_jac = jac[out].keys()

    if inputs is None:
        input_td = [i for i in td if i in inputs_jac]  # inputsall
    else:
        input_td = [i for i in inputs if i in inputs_jac and i in td]

    td_decomp = {'all': td[out]}
    td_decomp.update({i: jac[out][i] @ td[i] for i in input_td})
    if plot:
        make_plot(title, title, np.arange(Tplot), {out: {k: td_decomp[k][:Tplot] for k in td_decomp}})
    return td_decomp


def inspect_block(title, block, td, out, shock=None, Tplot=20, inputs=None, plot=True):
    if shock is not None:  # important to feed shock in !!
        td.update(shock)
    if inputs is None:
        input_td = [i for i in td if i in block.inputs]  # inputsall
    else:
        input_td = [i for i in inputs if i in block.inputs and i in td]
    td_decomp = {'all': block.eval_td(td)[out]}
    td_decomp.update({i: block.eval_td({i: td[i]})[out] for i in input_td})
    if plot:
        make_plot(title, title, np.arange(Tplot), {out: {k: td_decomp[k][:Tplot] for k in td_decomp}})
    return td_decomp


def jacobian(f, x0, f0=None, dx=1e-4):  # compute Jacobian
    if f0 is None:
        f0 = f(x0)

    m = f0.shape[0]
    n = x0.shape[0]
    I = np.eye(n)
    jac = np.empty((m, n))
    for i in range(n):
        jac[:, i] = (f0 - f(x0 - dx * I[i, :])) / dx
    return jac


def match_IRFs(model, params, series, shock_fn, file, unknowns=('y', 'w', 'r'), optimize=True, bounds=None,
               levels=True, fake_data=None, exog_shocks=None, first_datapoint=1, qs=4, sd=False, param_equals={},
               weights=None,
               targets=('fisher', 'real_wage', 'asset_clearing'), global_opt=False, **kwargs):
    """This function estimates the parameters in list 'params', with initial guesses,
     to match 'model' IRFs from list 'series' with the ones in string 'file'
      bounds in format {'param1': (0.1, 0.9), 'param2': (0.3, 0.5), ... } """
    # prepare
    series = convert_tuple(series)
    params = convert_tuple(params)
    if exog_shocks is not None:  # exog_shocks has to be dict with keys = shock_params, and values = initial values
        shock_params = tuple([k for k in exog_shocks])
    else:
        shock_params = shock_fn.__code__.co_varnames
    n_p = len(params)
    n_sp = len(shock_params)
    ssvals = model['ss'].ssout
    if weights is None:
        weights = {k: 1 for k in series}

    # set groups of variables
    var_perc = ['i', 'r', 'pi']

    # set stock variables (will have to be multiplied by 4, for 4 Quarters!)
    qs = 4
    var_stock = ['Bqss']

    # use fake data set (for debugging)?
    if fake_data is not None:
        print('Using manual data.')
        # fake_data must be a tuple consisting of (data_dict, sd_dict)
        d_mean = fake_data[0]
        sd = fake_data[1]
        t_max = len(fake_data[0][[*fake_data[0]][0]])  # length of the value of the first item
        d = d_mean.copy()
        d.update({k + '_lo': d_mean[k] - 2 * sd[k] for k in d_mean})
        d.update({k + '_hi': d_mean[k] + 2 * sd[k] for k in d_mean})
    else:
        # read data into d_mean, d_lo, d_hi
        df = pd.read_csv(file, index_col=0)
        t_max = len(df.index.tolist()) - first_datapoint
        d_dict = df.to_dict('list')
        d = {k: np.array(d_dict[k][first_datapoint:]) for k in d_dict}  # everything is 1% format
        if not levels:
            d.update({k: d[k] * ssvals[k.split('_')[0]] for k in d.keys() if
                      k.split('_')[0] not in var_perc and k.split('_')[
                          0] in ssvals})  # scale by ss level except if % variable
        if levels:
            d.update({k: d[k] * qs for k in d.keys() if k.split('_')[:-1] in var_stock})
        d.update({k: d[k] / qs for k in d.keys() if k.split('_')[0] in var_perc and k.split('_')[
            0] in ssvals})  # divide interest rates by 4 to have it be quarterly
        # compute r by hand if not otherwise present
        if 'r' not in d:
            d.update({'r' + k: d['i' + k] - np.array([*d['pi'][1:], d['pi'][-1]]) for k in ['', '_lo', '_hi']})
        if 'W' not in d:
            d.update({'W' + k: d['w' + k] + d['p' + k] for k in ['', '_lo', '_hi']})
        # scale_quantities = 1
        # d.update({k: scale_quantities * d[k] for k in d.keys() if k[0] in ['y', 'c', 'I']})
        d_mean = {k: d[k] for k in series}
        sd = {k: np.abs(d[k + '_hi'] - d[k + '_lo']) / 4 for k in series}

    # prepare optimization
    x0_params = [model['ss'].ssin[p] for p in params]
    if exog_shocks is not None:
        x0_shocks = [exog_shocks[k] for k in exog_shocks]
    else:
        x0_shocks = shock_fn.__defaults__
    if x0_shocks is None:
        x0_shocks = ()
    assert len(x0_shocks) == n_sp
    x0 = np.concatenate((x0_params, x0_shocks))
    if bounds is not None:
        bounds_tuple = (np.array([bounds[p][0] for p in params + shock_params]),
                        np.array([bounds[p][1] for p in params + shock_params]))
        kwargs.update({'bounds': bounds_tuple})

    # if targets is None and np.abs(model['ss'].ssout['goods_clearing']) < np.abs(model['ss'].ssout['asset_clearing']):
    #     targets = ('fisher', 'real_wage', 'goods_clearing')
    # do not override targets!
    # else:
    #    targets = ('fisher', 'real_wage', 'asset_clearing')

    # define objective function
    def objective(x, opt_mode=True):  # TODO optmode
        # print(x)
        x_params = x[:n_p]
        x_shocks = x[n_p:]
        assert len(x_params) == n_p
        calib_here = model['ss'].ssout.copy()
        calib_here.update({params[i]: x_params[i] for i in range(n_p)})
        calib_here.update({pa1: calib_here[pa2] for pa1, pa2 in param_equals.items()})
        shock_args = {shock_params[i]: x_shocks[i] for i in range(n_sp)}
        shocks = shock_fn(**shock_args)
        model['td_reval'].eval_ss(calib_here)
        td_all = model['td'].solve_td(shocks, unknowns=unknowns,
                                      targets=targets)
        # diff = {k: (td_all[k][:t_max] - d_mean[k])/sd[k] for k in series}
        # diff_stacked = np.array([item for k in series for item in diff[k].tolist()])
        diff_stacked = np.array(
            [(td_all[k][i] - d_mean[k][i]) / sd[k][i] * np.sqrt(weights[k]) for k in series for i in range(t_max) if
             np.abs(sd[k][i]) > 1e-7])
        if opt_mode:
            return diff_stacked
        else:
            return calib_here, td_all, shock_args

    # minimize the objective
    if optimize:
        if not global_opt:
            result = spopt.least_squares(objective, x0, **kwargs)
        else:
            def objective2(x):
                return objective(x) @ objective(x)

            bounds_list = list(zip(bounds_tuple[0], bounds_tuple[
                1]))  # [(bounds_tuple[0][i], bounds_tuple[1][i]) for i in range(len(bounds_tuple[0]))]
            result = spopt.dual_annealing(objective2, bounds=bounds_list)
        x = result.x
        # compute std deviations
        # Note, since we divided by sd above, the std.dev of our reference data d_mean/sd is 1!
        # Thus, we automatically did "optimal weighting" and the variance of x is just jac.T * jac !
        # Either compute jacobian separately or use the approximation from solution
        if sd:
            jac = jacobian(objective, x, f0=result.fun)  # Note: jac already has sqrt-weights in it!
            print('Computing sd deviations manually...')
        else:
            jac = result.jac
        W = np.diag(np.array([weights[k] for k in series for i in range(t_max) if
                              np.abs(sd[k][i]) > 1e-7]))  # weighting matrix (symmetric since diagonal)
        var_x = np.linalg.inv(jac.T @ jac) @ jac.T @ W @ jac @ np.linalg.inv(
            jac.T @ jac)  # var-covar of x. W only appears in middle since weighting is already part of jac!
        x_sd = np.sqrt(np.diagonal(var_x))
        nfev = result.nfev
    else:
        result = objective(x0)
        x = x0
        nfev = 1
        x_sd = np.zeros_like(x)
    calib_here, td_all, shock_args = objective(x, opt_mode=False)
    return x, x_sd, calib_here, result, nfev, td_all, d, t_max, shock_args  # calib_here, shock_args, result, td_all, d, t_max, x_sd


""" Likelihood based estimation """


@njit
def stacked_complex_mult(A, B):
    n, m, _ = A.shape
    C = np.empty((n, m, B.shape[2]), numba.complex128)
    for i in range(n):
        C[i, :, :] = A[i, :, :] @ B[i, :, :]
    return C


def fft_autocov(As):
    """Takes in T*O*E array containing MA(infinity),
    spits out autocovariance matrices E[x_tx_{t+u}'] for all u=0,...,T"""
    T = As.shape[0]
    dft = np.fft.rfftn(As, s=(2 * T - 2,), axes=(0,))
    total = stacked_complex_mult(dft.conjugate(), dft.swapaxes(1, 2))
    return np.fft.irfftn(total, s=(2 * T - 2,), axes=(0,))[:T]


@njit
def build_full_autocovariance_matrix(gammas, T):
    """Takes in tau*O*O array with observable autocovariances at each horizon tau, assembles them into T*O*T*O matrix"""
    tau, O, O = gammas.shape
    Sigma = np.empty((T, O, T, O))
    for t1 in range(T):
        for t2 in range(T):
            if abs(t1 - t2) >= tau:
                Sigma[t1, :, t2, :] = np.zeros((O, O))
            else:
                if t1 < t2:
                    Sigma[t1, :, t2, :] = gammas[t2 - t1, :, :]
                elif t1 > t2:
                    Sigma[t1, :, t2, :] = gammas[t1 - t2, :, :].T
                else:
                    # want exactly symmetric!
                    Sigma[t1, :, t2, :] = (gammas[0, :, :] + gammas[0, :, :].T) / 2
    return Sigma


def log_likelihood(Sigma, y):
    """Implements multivariate normal log-likelihood formula given O observables for T periods, with T*O y
    and T*O*T*O (auto)covariance matrix Sigma"""
    T, O = y.shape
    Sigma = Sigma.reshape((T * O, T * O))
    y = y.reshape(T * O)

    Sigma_factored = scipy.linalg.cho_factor(Sigma)
    quadratic_form = np.dot(y, scipy.linalg.cho_solve(Sigma_factored, y))

    log_determinant = 2 * np.sum(np.log(np.diag(Sigma_factored[0])))

    return -(T * O * np.log(2 * np.pi) + log_determinant + quadratic_form) / 2


def log_likelihood_from_MA(As, y, sigma_e=None, sigma_o=None):
    """Efficiently calculates log-likelihood given MA(T) As and data y

    Parameters
    ----------
    As : array (Tm*O*E) giving the O*E matrix mapping shocks to observables at each of Tm lags in the MA(infty),
            e.g. As[6, 3, 5] gives the impact of shock 5, 6 periods ago, on observable 3 today
    y : array (To*O) giving the data (already assumed to be demeaned, though no correction is made for this in the log-likelihood)
            each of the To rows t is the vector of observables at date t (earliest should be listed first)
    sigma_e : [optional] array (E) giving sd of each shock e, assumed to be 1 if not provided
    sigma_o : [optional] array (O) giving sd of iid measurement error for each observable o, assumed to be 0 if not provided

    Returns
    ----------
    log_likelihood : float
    """

    # to implement sigmas for the shocks, just rescale the MA array
    if sigma_e is not None:
        As = As * sigma_e

    gammas = fft_autocov(As)

    # to implement iid measurement error, just add to diagonal of contemporaneous covariance matrix, i.e. gammas[0]
    if sigma_o is not None:
        gammas[0, ...] += np.diag(sigma_o ** 2)

    Sigma = build_full_autocovariance_matrix(gammas, y.shape[0])

    return log_likelihood(Sigma, y)


# # @njit # TODO JIT THIS
# def generate_ARMA_matrix(ar_coeff, ma_coeff, T):
#     AR = np.eye(T,T)
#     for i in range(min(len(ar_coeff),T-1)):
#         AR += - np.diag(ar_coeff[i] * np.ones(T-i-1),-i-1)
#     MA = np.eye(T,T)
#     for i in range(min(len(ma_coeff),T-1)):
#         MA += np.diag(ma_coeff[i] * np.ones(T-i-1),-i-1)
#     return np.linalg.inv(AR) @ MA

@njit
def arma_irf(ar_coeff, ma_coeff, T):  # NOTE: these need to be numpy.ndarrays
    x = np.empty((T,))
    n_ar = ar_coeff.size
    n_ma = ma_coeff.size
    sign_ma = -1  # This means all MA coefficients are multiplied by -1 (it's what SW etc all have)
    for t in range(T):
        if t == 0:
            x[t] = 1
        else:
            ar_sum = 0
            for i in range(min(n_ar, t)):
                ar_sum += ar_coeff[i] * x[t - 1 - i]
            ma_term = 0
            if 0 < t <= n_ma:
                ma_term = ma_coeff[t - 1]
            x[t] = ar_sum + ma_term * sign_ma
    return x


def map_to_x(model, params, shocks, meas_error):
    x0_params = [model['ss'].ssin[p] for p in params]
    x0_shocks = [shocks[k][0] for k in shocks]  # initial std devations
    x0_shcoef = [shocks[k][1 + i][j] for k in shocks for i in range(len(shocks[k]) - 1) for j in
                 range(len(shocks[k][1 + i]))]
    return np.concatenate((x0_shocks, x0_shcoef, x0_params, meas_error))


def set_bounds(params, param_bounds, shocks, shocks_bounds, meas_error_bounds, n_me):
    bounds_tuple = [shocks_bounds[sh][0] for sh in shocks] + \
                   [shocks_bounds[k][1 + i][j] for k in shocks for i in range(len(shocks[k]) - 1) for j in
                    range(len(shocks[k][1 + i]))] + \
                   [param_bounds[p] for p in params] + \
                   [meas_error_bounds] * n_me
    return bounds_tuple


def map_from_x(x, n_p, n_sh, shocks, reparametrizeAR2=False):
    n_shcoef = sum([len(shocks[k][1]) + len(shocks[k][2]) for k in shocks])  # number of additional coefficients

    sigmas = x[:n_sh]  # these are variances
    shcoef = x[n_sh:n_sh + n_shcoef]
    x_params = x[n_sh + n_shcoef:n_sh + n_shcoef + n_p]
    meas_error = x[n_sh + n_shcoef + n_p:]

    # extract ARMA coefficients from shcoef
    j = 0
    shocks_coeff = {}
    for sh in shocks:
        len_ar = len(shocks[sh][1])
        len_ma = len(shocks[sh][2])
        arcoef = shcoef[j:j + len_ar]
        macoef = shcoef[j + len_ar:j + len_ar + len_ma]
        if reparametrizeAR2 and len_ar == 2:
            # reparameterize AR2, see description in MLE function
            arcoef = np.array([arcoef[0] + arcoef[1], - arcoef[0] * arcoef[1]])
        shocks_coeff[sh] = (arcoef, macoef)
        j += len_ar + len_ma
    return x_params, sigmas, shocks_coeff, meas_error


def hessian(f, x0, f_x0=None, dx=1e-4):
    """Function to compute Hessian of generic function"""
    n = x0.shape[0]
    I = np.eye(n)

    # check if function value is given
    if f_x0 is None:
        f_x0 = f(x0)

    # compute Jacobian
    jac = np.empty(n)
    for i in range(n):
        jac[i] = (f_x0 - f(x0 - dx * I[i, :])) / dx

    # compute the hessian)
    hess = np.empty((n, n))
    for i in range(n):
        f_xi = f(x0 + dx * I[i, :])
        hess[i, i] = ((f_xi - f_x0) / dx - jac[i]) / dx
        for j in range(i):
            jac_j_at_xi = (f(x0 + dx * I[i, :] + dx * I[j, :]) - f_xi) / dx
            hess[i, j] = (jac_j_at_xi - jac[j]) / dx - hess[j, j]
            hess[j, i] = hess[i, j]

    return hess


def log_priors(x, priors_list):
    """This function computes a sum of log prior distributions that are stored in priors_list.
    Example: priors_list = {('Normal', 0, 1), ('Invgamma', 1, 2)}
    and x = np.array([1, 2])"""
    assert len(x) == len(priors_list)
    sum_log_priors = 0
    # dx = np.empty_like(x)
    for n in range(len(x)):
        dist = priors_list[n][0]
        mu = priors_list[n][1]
        sig = priors_list[n][2]
        if dist == 'Normal':
            sum_log_priors += - 0.5 * ((x[
                                            n] - mu) / sig) ** 2  # np.log( 1/(sig * np.sqrt(2 * np.pi)) * np.exp(- 0.5 * ((x[n] - mu)/sig)**2) )  #np.log(spstat.norm.pdf(x[n],loc=mu, scale=sig))
            # dx[n] = - 0.5 * ((x[n] - mu)/sig)**2
        elif dist == 'Invgamma':
            alpha = (mu / sig) ** 2 + 2
            beta = mu * (alpha - 1)
            sum_log_priors += (-alpha - 1) * np.log(x[n]) - beta / x[
                n]  # np.log( 1/beta * (x[n]/beta)**(-alpha-1)/gamma(alpha) * np.exp(-beta/x[n]) )  # np.log(spstat.invgamma.pdf(x[n], a=alpha, scale=beta))
            # assert np.isclose(spstat.invgamma.mean(a=alpha,scale=beta), mu) and np.isclose(sig**2, spstat.invgamma.var(a=alpha,scale=beta))
            # dx[n] =
        elif dist == 'Gamma':
            theta = sig ** 2 / mu
            k = mu / theta
            sum_log_priors += (k - 1) * np.log(x[n]) - x[
                n] / theta  # np.log( 1/theta * (x[n]/theta)**(k-1) * np.exp(-x[n]/theta) / gamma(k) )  #np.log(spstat.gamma.pdf(x[n], a=k, scale=theta))
            # assert np.isclose(spstat.gamma.mean(a=k, scale=theta), mu) and np.isclose(sig ** 2, spstat.gamma.var(a=k, scale=theta))
            # dx[n] =
        elif dist == 'Beta':
            alpha = (mu * (1 - mu) - sig ** 2) / (sig ** 2 / mu)
            beta = alpha / mu - alpha
            sum_log_priors += (alpha - 1) * np.log(x[n]) + (beta - 1) * np.log(1 - x[
                n])  # np.log( gamma(alpha+beta)/(gamma(alpha)*gamma(beta)) * x[n]**(alpha-1) * (1-x[n])**(beta-1) )  #np.log(spstat.beta.pdf(x[n], a=alpha, b=beta))
            # assert np.isclose(spstat.beta.mean(a=alpha, b=beta), mu) and np.isclose(sig ** 2, spstat.beta.var(a=alpha, b=beta))
            # dx[n] =
        else:
            raise ValueError('Distribution provided is not implemented in log_priors!')

    if np.isinf(sum_log_priors) or np.isnan(sum_log_priors):
        print(x)
        raise ValueError('Need tighter bounds to prevent prior value = 0')
    return sum_log_priors


def extrap(As, T_extrap, lag=10):
    """ This function extrapolates array As along the first dimension. uses T-lag and T-lag+1 for that """
    T = As.shape[0]
    assert T_extrap > T and lag < T
    ratios = As[-lag - 1, ...] / As[-lag - 2, ...]
    ratios[ratios >= 1] = 0
    ratios[ratios < 0] = 0
    As_extra = np.empty((T_extrap - T + lag,) + As.shape[1:])
    As_extra[0, ...] = As[-lag - 1, ...] * ratios
    for t in range(T_extrap - T - 1):
        As_extra[t + 1, ...] = As_extra[t, ...] * ratios
    As = np.concatenate((As[:T - lag], As_extra), axis=0)
    return As


def MLE(model, params, series, shocks, file, T=100, T_trunc=None, unknowns=('y', 'w', 'r'),
        targets=('fisher', 'real_wage', 'asset_clearing'), param_bounds=None, shocks_bounds=None,
        optimize=True, meas_error0=None, meas_error_bounds=(1e-3, 10), sd=False, priors=None, qs=4, T_extrap=None,
        percent_vars_already_quarterly=False,
        start_date='1966Q1', end_date='2006Q4', fake_data=None, levels=True, reparametrizeAR2=False, exog_G=None,
        global_opt=False, **kwargs):
    """ reparametrizeAR2: if true, any AR2 paths fed in are read as [rho, index] to be formed into an
                          AR2 with coefficients ar1=rho+index, ar2=-rho*index. """

    # prepare
    n_p = len(params)
    n_se = len(series)
    n_sh = len(shocks)
    shocks_list = [*shocks]

    # Check whether we need measurement error or not
    if n_sh > n_se:
        meas_error = True
    elif n_sh == n_se:
        meas_error = False
    else:
        raise ValueError('Cannot estimate with more shocks than series!')

    # set normalization for each series
    series_no_norm = ['i', 'pi', 'r']
    var_perc = ['i', 'pi', 'r']
    norm = {se: 1 for se in series}
    if not levels:
        norm.update({se: model['ss'].ssout[se] for se in series if se not in series_no_norm})

    # use fake data set (for debugging)?
    if fake_data is not None:
        print('Using fake data.')
        data = fake_data
    else:
        # load data
        df = pd.read_csv(file, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.to_period("Q")
        if start_date is not None and end_date is not None:
            df = df[start_date:end_date]
        df = df[list(series)]  # select the right series (this is ordered like series)
        if not percent_vars_already_quarterly:
            for se in var_perc:
                if se in df.columns:
                    df[se] = df[se] / qs  # interest variables need to be quarterly!!
        data = df.values  # this gives an array

    # set initial x for the optimization
    if meas_error:
        if meas_error0 is None:
            x0_meas_error = np.ones(n_se)
        else:
            x0_meas_error = meas_error0
        x0 = map_to_x(model, params, shocks, x0_meas_error)
        n_me = n_se
    else:
        n_me = 0
        x0 = map_to_x(model, params, shocks, [])

    # set bounds for the optimization
    if param_bounds is not None and shocks_bounds is not None:
        bounds_tuple = set_bounds(params, param_bounds, shocks, shocks_bounds, meas_error_bounds, n_me)
        assert len(bounds_tuple) == len(x0)
        kwargs.update({'bounds': bounds_tuple})

    # set priors for the optimization
    if priors is not None:
        priors_list = [priors[k][0] for k in shocks] + \
                      [priors[k][1][i] for k in shocks for i in range(len(shocks[k][1]))] + \
                      [priors[k][2][i] for k in shocks for i in range(len(shocks[k][2]))] + [priors[k] for k in
                                                                                             params] + \
                      [('Invgamma', 0.1, 2)] * n_me
        assert len(priors_list) == len(x0)

    # prepare TD
    # simulate model for each shock
    if T_trunc is not None:
        T_irf = T_trunc
    else:
        T_irf = T

    model_td_ini = model['td'].solved(unknowns=unknowns, targets=targets)
    calib_here_ini = model['ss'].ssout.copy()

    # Compute G matrix in case of no parameters
    if n_p == 0:
        if exog_G is not None:
            G = exog_G
        else:
            G = model_td_ini.own_jac(inputs=tuple(shocks_list), outputs=series, T=T)
    else:
        # prepare mutable object to store last evaluated parameters in
        x0_params, _, _, _ = map_from_x(x0, n_p, n_sh, shocks, reparametrizeAR2=reparametrizeAR2)
        last_x_params = [x0_params]
        last_model = [model_td_ini]
        counter = [0]

    def objective(x, opt_mode=True):
        # print(x)
        x_params, sigmas, shocks_coeff, meas_error = map_from_x(x, n_p, n_sh, shocks, reparametrizeAR2=reparametrizeAR2)
        assert len(meas_error) == n_me
        if n_me == 0:
            meas_error = np.zeros(n_se)

        # check if any parameters are to be assigned at all
        if n_p > 0:
            counter[0] += 1
            if counter[0] % 10 == 0:
                print(str(counter[0]) + '/', end='', flush=True)
                if counter[0] % 250 == 0:
                    print('\n')
                    print(x)
            # Decide whether any parameters have changed
            if not np.allclose(x_params, last_x_params[0], rtol=1e-10, atol=1e-10):
                # print(' *** Model parameters changed *** \n')
                calib_here = calib_here_ini
                calib_here.update({params[i]: x_params[i] for i in range(n_p)})
                model['td_reval'].eval_ss(calib_here)
                model_td = model['td'].solved(unknowns=unknowns, targets=targets)
                last_x_params[0] = x_params
                last_model[0] = model_td
            else:
                # print(' no model parameter changed\n')
                model_td = last_model[0]

        As = np.empty((T_irf, n_se, n_sh))
        shock_paths = {}
        for i_sh in range(n_sh):
            sh = shocks_list[i_sh]
            arma_shock = arma_irf(shocks_coeff[sh][0], shocks_coeff[sh][1], T)
            shock_paths.update({sh: arma_shock})

            if np.abs(arma_shock[-1]) > 1e20:
                raise Warning('ARMA shock misspecified, leading to explosive shock path!')

            if n_p > 0:
                # run simulation
                # print(i_sh)
                current_td = model_td.eval_td({sh: arma_shock})

            # store for each series
            for i_se in range(n_se):
                if n_p > 0:
                    As[:, i_se, i_sh] = current_td[series[i_se]][:T_irf] / norm[series[i_se]]

                else:
                    As[:, i_se, i_sh] = (G[series[i_se]][shocks_list[i_sh]] @ arma_shock)[:T_irf] / norm[series[i_se]]

        # possibly extrapolate IRFs
        if T_extrap is not None:
            if T_extrap > T_irf:
                As = extrap(As, T_extrap)

        if np.max(np.abs(As)) > 1e3:
            print(x)
            print('WARNING: Found too large values in IRFs! Might blow up cholesky decomposition. Check bounds for rho')
            Warning('Found too large values in IRFs! Might blow up cholesky decomposition. Check bounds for rho')
            assert opt_mode
            return 1e6

        # compute negative likelihood
        neg_likelihood = -log_likelihood_from_MA(As, data, sigma_e=sigmas, sigma_o=meas_error)

        # add negative log prior if activated
        if priors is not None:
            neg_likelihood += - log_priors(x, priors_list)

        if opt_mode:
            return neg_likelihood
        else:
            if n_p > 0:
                return neg_likelihood, calib_here, As, sigmas, meas_error, shock_paths
            else:
                return neg_likelihood, calib_here_ini, As, sigmas, meas_error, shock_paths

    # minimize the objective
    if optimize:
        if global_opt is False:
            result = spopt.minimize(objective, x0, **kwargs)
        else:
            # result = spopt.differential_evolution(objective, **kwargs)
            result = spopt.dual_annealing(objective, x0=x0, **kwargs)
        x = result.x
        nfev = result.nfev
        if sd:
            Hinv = np.linalg.inv(hessian(objective, x, result.fun))
        else:
            Hinv = np.zeros((x.size, x.size))
            print('Note: Set sd=True to compute std errors.')
            # if hasattr(result, 'hess_inv'):
            #     Hinv = result.hess_inv.todense()
            # else:
            #     Hinv = np.zeros((x.size,x.size))
            #     print('Warning: Still need to implement the standard errors for the optimization method you have used.')
        x_sd = np.sqrt(np.diagonal(Hinv))
    else:
        result = None  # = objective(x0)
        x = x0
        nfev = 1
        if sd:
            Hinv = np.linalg.inv(hessian(objective, x, objective(x0)))
            x_sd = np.sqrt(np.diagonal(Hinv))
        else:
            print('Note: Set sd=True to compute std errors.')
            x_sd = np.zeros_like(x)

    neg_likelihood, calib_here, As, sigmas, meas_error, shock_paths = objective(x, opt_mode=False)
    likelihood = - neg_likelihood
    return x, x_sd, result, nfev, likelihood, calib_here, x0, As, data, sigmas, meas_error, shock_paths


# @njit
def construct_stacked_A(As, To, To_out=None, sigma_e=None, sigma_o=None, reshape=True, long=False):
    # this can be jited, but I don't know how
    Tm, O, E = As.shape

    # how long should the IRFs be that we stack in A_full?
    if To_out is None:
        To_out = To
    if long:
        To_out = To + Tm  # store even the last shock's IRF in full!

    # allocate memory for A_full
    A_full = np.zeros((To_out, O, To, E))

    for o in range(O):
        for itshock in range(To):
            # if To > To_out, allow the first To - To_out shocks to happen before the To_out time periods
            if To <= To_out:
                iA_full = itshock
                iAs = 0

                shock_length = min(Tm, To_out - iA_full)
            else:
                # this would be the correct start time of the shock
                iA_full = itshock - (To - To_out)

                # since it can be negative, only start IRFs at later date
                iAs = - min(iA_full, 0)

                # correct iA_full by that date
                iA_full += - min(iA_full, 0)

                shock_length = min(Tm, To_out - iA_full)

            for e in range(E):
                A_full[iA_full:iA_full + shock_length, o, itshock, e] = As[iAs:iAs + shock_length, o, e]
                if sigma_e is not None:
                    A_full[iA_full:iA_full + shock_length, o, itshock, e] *= sigma_e[e]
                if sigma_o is not None:
                    A_full[iA_full:iA_full + shock_length, o, itshock, e] /= sigma_o[o]
    if reshape:
        A_full = A_full.reshape((To_out * O, To * E))
    return A_full


def back_out_shocks(As, y, sigma_e=None, sigma_o=None, preperiods=0):
    """Calculates most likely shock paths if As is true set of IRFs

    Parameters
    ----------
    As : array (Tm*O*E) giving the O*E matrix mapping shocks to observables at each of Tm lags in the MA(infty),
            e.g. As[6, 3, 5] gives the impact of shock 5, 6 periods ago, on observable 3 today
    y : array (To*O) giving the data (already assumed to be demeaned, though no correction is made for this in the log-likelihood)
            each of the To rows t is the vector of observables at date t (earliest should be listed first)
    sigma_e : [optional] array (E) giving sd of each shock e, assumed to be 1 if not provided
    sigma_o : [optional] array (O) giving sd of iid measurement error for each observable o, assumed to be 0 if not provided
    preperiods : [optional] integer number of pre-periods during which we allow for shocks too. This is suggested to be at
            least 1 in models where some variables (e.g. investment) only respond with a 1 period lag.
            (Otherwise there can be invertibility issues)

    Returns
    ----------
    eps_hat : array (To*E) giving most likely path of all shocks
    Ds : array (To*O*E) giving the level of each observed data series that is accounted for by each shock
    """

    # Step 1: Rescale As any y
    To, Oy = y.shape
    Tm, O, E = As.shape
    assert Oy == O
    To_with_pre = To + preperiods

    A_full = construct_stacked_A(As, To=To_with_pre, To_out=To, sigma_e=sigma_e, sigma_o=sigma_o)
    if sigma_o is not None:
        y = y / sigma_o
    y = y.reshape(To * O)

    # Step 2: Solve OLS
    eps_hat = np.linalg.lstsq(A_full, y, rcond=None)[0]  # this is To*E x 1 dimensional array
    eps_hat = eps_hat.reshape((To_with_pre, E))

    # Step 3: Decompose data
    for e in range(E):
        A_full = A_full.reshape((To, O, To_with_pre, E))
        Ds = np.sum(A_full * eps_hat, axis=2)

    # Cut away pre periods from eps_hat
    eps_hat = eps_hat[preperiods:, :]

    return eps_hat, Ds


def forecast(As, eps, t_forecast, decomp=False, sigma_e=None):
    """Calculates mean forecasts at various dates. Left to do: confidence intervals

    Parameters
    ----------
    As : array (Tm*O*E) giving the O*E matrix mapping shocks to observables at each of Tm lags in the MA(infty),
            e.g. As[6, 3, 5] gives the impact of shock 5, 6 periods ago, on observable 3 today
    eps : array (To*E) giving the paths of all shocks over the sample period
    t_forecast : index following which forecast is computed (last index with non-negative shock)

    Returns
    ----------
    forecasts : array ((To+Tm)*O)  if decomp == False, array ((To+Tm)*O*E) else
    """
    # Tm, O, E = As.shape
    To, E = eps.shape
    eps_here = eps.copy()
    for e in range(E):
        eps_here[t_forecast + 1:, e] = 0
    # eps = eps.reshape((To * E))
    A_full = construct_stacked_A(As, To, sigma_e=sigma_e, reshape=False, long=True)  # array: To+Tm, O, To, E
    if decomp:
        forecasts = np.einsum('ijkl,kl->ijl', A_full, eps_here)
    else:
        forecasts = np.einsum('ijkl,kl->ij', A_full, eps_here)
    return forecasts


def simulate_data(As, To, sigma_e=None, sigma_o=None, T_burn=100, store=False,
                  labels=['t', 'y', 'I', 'c', 'pi', 'i', 'n', 'w'], filename=None, firstyear=1966):
    """Simulates data from IRFs As, with length T

        Parameters
        ----------
        As : array (Tm*O*E) giving the O*E matrix mapping shocks to observables at each of Tm lags in the MA(infty),
                e.g. As[6, 3, 5] gives the impact of shock 5, 6 periods ago, on observable 3 today
        sigma_e : [optional] array (E) giving sd of each shock e, assumed to be 1 if not provided
        sigma_o : [optional] array (O) giving sd of iid measurement error for each observable o, assumed to be 0 if not provided

        Returns
        ----------
        y : array (To*O) giving the data (already assumed to be demeaned, though no correction is made for this in the log-likelihood)
                each of the To rows t is the vector of observables at date t (earliest should be listed first)


        # eps_hat : array (To*E) giving most likely path of all shocks
        # Ds : array (To*O*E) giving the level of each observed data series that is accounted for by each shock
        """

    Tm, O, E = As.shape
    T = To + T_burn

    # random shocks
    eps_hat = np.random.randn(T, E)
    eps_o = np.random.randn(T, O)  # measurement error

    if sigma_e is not None:
        eps_hat *= sigma_e
    if sigma_o is not None:
        eps_o *= sigma_o
    else:
        eps_o *= 0

    # full output
    y = np.zeros((T, O))
    data = [labels]
    for t in range(T):
        tmin = np.min((T - t, Tm))
        # print(tmin)
        y[t:t + tmin, :] += As[:tmin, :, :] @ eps_hat[t, :] + eps_o[t, :]
        # print(f'{t}: {y[t+tmin-1,3]/y[t+tmin-2,0]}')
        # print(As[tmin - 1, 3, 0] / As[tmin - 2, 0, 0])
        # print(y.shape)
        if store and t >= T_burn:
            s = t - T_burn
            year = int(np.floor(s / 4))
            quarter = int(s - 4 * year + 1)
            date = str(year + firstyear) + 'Q' + str(quarter)
            row = [date] + y[t, :].tolist()
            data.append(row)
            # data.append(t-T_burn)  #, y[t, :]])
            # data.append(y[t, :])  #(np.concatenate((np.array([t-T_burn]), y[t, :])))

    # cut y down
    y = y[T_burn:, :]

    # store as txt file?
    if store:
        assert filename is not None
        # data = sort(data)
        # data = transp(data)
        write_to_file(filename, data, write_csv=True, sort_trans=False)

    return y

def cov(v, w):
    return (v*w).mean() - (v.mean() * w.mean())

def var(v):
    return cov(v, v)

def sd(v):
    return np.sqrt(var(v))

def corr(v, w):
    return cov(v, w) / sd(v) / sd(w)

def auto(v, k=1):  # auto correlation
    return corr(v[:-k], v[k:])


def avg_fractile(p, x, p0, p1):
    """ p = vector with probabilities, x = vector with values
        function computes average value between percentiles p0 and p1 """
    assert len(p) == len(x)
    if np.isclose(p.sum(), 1):  # this can be violated when simulating a single firm ...

        # sort by value
        I = x.argsort()
        x = x[I]
        p = p[I]

        # CDF
        cum_prob = np.cumsum(p)

        # indices that straddle the percentiles
        i0 = np.argwhere(cum_prob>p0)[0][0]
        i1 = np.argwhere(cum_prob<=p1)[-1][0]
        x0 = np.interp(p0, cum_prob, x)
        x1 = np.interp(p1, cum_prob, x)

        # construct selected x and p
        x_selec = np.empty(i1-i0 + 3)
        x_selec[1:-1] = x[i0:i1+1]
        x_selec[0] = x0
        x_selec[-1] = x1
        cum_prob_selec = np.empty(i1-i0 + 3)
        cum_prob_selec[1:-1] = cum_prob[i0:i1 + 1]
        cum_prob_selec[0] = p0
        cum_prob_selec[-1] = p1

        # compute avg
        avg = np.trapz(x_selec, cum_prob_selec) / (p1 - p0)
        return avg
    else:
        return 0