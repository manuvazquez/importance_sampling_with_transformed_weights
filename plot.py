import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


def setup_axes(figure_id, clear_figure=True):

	# interactive mode on
	plt.ion()

	# a new figure is created...
	fig = plt.figure(figure_id)

	# ...and, if requested,...
	if clear_figure:
		# ...cleared, just in case this method is called several times with the same "figure_id"
		plt.clf()

	# ...and the corresponding axes created
	axes = plt.axes()

	return axes, fig


def plain_vs_tiw(x, ys, parameters, id, output_file=None, axes_properties={}):

	ax, fig = setup_axes(id)

	for y, parameters in zip(ys.T, parameters):

		ax.loglog(x, y, **parameters)

	# the labels are shown
	ax.legend()

	ax.set(**axes_properties)

	fig.show()

	if output_file:

		plt.savefig(output_file)

	return ax, fig


def plain_vs_tiw_with_max_weight(x, ys, max_weight, parameters, id, output_file=None, axes_properties={}):

	parameters_without_labels = [{k: par[k] for k in par if k!='label'} for par in parameters]
	for par in parameters_without_labels:
		par['marker'] = 'None'

	ax, fig = plain_vs_tiw(x, ys, parameters_without_labels, id, output_file=None, axes_properties=axes_properties)

	cm = plt.cm.get_cmap('RdYlBu_r')

	leg = []

	for y, w, par in zip(ys.T, max_weight.T, parameters):

		leg.append(matplotlib.lines.Line2D(
			[], [], color=par['color'], marker=par['marker'], markersize=8, markerfacecolor='None',
			markeredgecolor=par['color'], label=par['label']))

		sc = ax.scatter(x, y, c=w, cmap=cm, s=70, marker=par['marker'])

	ax.legend(handles=leg)

	# the x axis is adjusted so that no empty space is left before the beginning of the plot
	ax.set_xbound(lower=x[0], upper=x[-1])

	fig.colorbar(sc)
	fig.show()

	if output_file:

		plt.savefig(output_file)

	return ax, fig


def variance(x, ys, output_file=None):

	ax, fig = setup_axes('variance')

	for alg_y, label, color, marker in zip(np.rollaxis(ys, 2), ['plain IW', 'TIW'], ['black', 'blue'], ['s', 'o']):

		for y, coeff_label, linestyle in zip(alg_y, ['$\\theta_1$', '$\\theta_2$'], ['dotted', 'dashed']):

			ax.loglog(x, y, label='{} for {}'.format(coeff_label, label), color=color, marker=marker, linestyle=linestyle)

	# the labels are shown
	ax.legend()

	ax.set_xlabel('number of particles')
	ax.set_ylabel('variance estimates')

	if output_file:

		plt.savefig(output_file)


def single_curve(x, y, id, output_file=None, axes_properties={}):

	ax, fig = setup_axes(id)

	ax.plot(x, y, marker='s')

	ax.set(**axes_properties)

	fig.show()

	if output_file:
		plt.savefig(output_file)


def vs_two_variables_3d(x, y, z, output_file=None):

	ax, fig = setup_axes('two')

	ax = fig.gca(projection='3d')

	ax.plot_surface(x, y, z)

	fig.show()

	if output_file:
		plt.savefig(output_file)


def vs_two_variables(x1, x2, ys, id, output_file=None, axes_properties={}):

	ax, fig = setup_axes(id)

	for y, color, N in zip(ys.T, ['blue', 'black', 'magenta', 'green', 'gray', 'brown'], x2):

		ax.plot(x1, y, color=color, label='N = {}'.format(N))

	# the labels are shown
	ax.legend()

	ax.set(**axes_properties)

	fig.show()

	if output_file:
		plt.savefig(output_file)