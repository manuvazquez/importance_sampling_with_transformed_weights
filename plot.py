import numpy as np
import matplotlib.pyplot as plt
import matplotlib


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


def plain_vs_tiw_aux(ax, x, ys, parameters, axes_properties={}):

	for y, parameters in zip(ys.T, parameters):

		ax.loglog(x, y, **parameters)

	# the labels are shown
	ax.legend()

	ax.set(**axes_properties)


def plain_vs_tiw(x, ys, parameters, id, output_file=None, axes_properties={}):

	ax, fig = setup_axes(id)

	plain_vs_tiw_aux(ax, x, ys, parameters, axes_properties)

	fig.show()

	if output_file:

		plt.savefig(output_file)

	return ax, fig


def plain_vs_tiw_with_max_weight_aux(
		ax, x, ys, max_weight, parameters, axes_properties={}, color_map=plt.cm.get_cmap('RdYlBu_r')):

	parameters_without_labels = [{k: par[k] for k in par if k !='label'} for par in parameters]

	for par in parameters_without_labels:

		par['marker'] = 'None'

	plain_vs_tiw_aux(ax, x, ys, parameters_without_labels, axes_properties=axes_properties)

	cm = color_map

	leg = []

	for y, w, par in zip(ys.T, max_weight.T, parameters):

		leg.append(matplotlib.lines.Line2D(
			[], [], color=par['color'], marker=par['marker'], markersize=8, markerfacecolor='None',
			markeredgecolor=par['color'], label=par['label']))

		sc = ax.scatter(x, y, c=w, cmap=cm, s=70, marker=par['marker'], vmin=0, vmax=1)

	ax.legend(handles=leg)

	# the x axis is adjusted so that no empty space is left before the beginning of the plot
	ax.set_xbound(lower=x[0], upper=x[-1])

	# the *last* scatter plot is returned
	return sc


def plain_vs_tiw_with_max_weight(x, ys, max_weight, parameters, id, output_file=None, axes_properties={}):

	ax, fig = setup_axes(id)

	sc = plain_vs_tiw_with_max_weight_aux(ax, x, ys, max_weight, parameters, axes_properties)

	color_bar = fig.colorbar(sc)

	# label for the color bar
	color_bar.ax.set_ylabel('max. weight', labelpad=25)

	fig.show()

	if output_file:

		plt.savefig(output_file)

	return ax, fig


def plain_vs_tiw_with_max_weight_multiple(
		x, ys1, ys2, max_weight, parameters, id, output_file=None,
		axes_properties1={}, axes_properties2={}):

	fig, axes = plt.subplots(nrows=1, ncols=2)

	plain_vs_tiw_with_max_weight_aux(axes[0], x, ys1, max_weight, parameters, axes_properties1)
	sc = plain_vs_tiw_with_max_weight_aux(axes[1], x, ys2, max_weight, parameters, axes_properties2)

	color_bar = fig.colorbar(sc)

	# label for the color bar
	color_bar.ax.set_ylabel('max. weight', labelpad=25)

	fig.show()

	fig.tight_layout()

	if output_file:

		plt.savefig(output_file)

	return axes, fig


def single_curve(x, y, id, output_file=None, axes_properties={}):

	ax, fig = setup_axes(id)

	ax.plot(x, y, marker='s')

	ax.set(**axes_properties)

	fig.show()

	if output_file:
		plt.savefig(output_file)


def vs_two_variables(x1, x2, ys, id, output_file=None, axes_properties={}):

	ax, fig = setup_axes(id)

	vs_two_variables_aux(ax, x1, x2, ys, axes_properties)

	fig.show()

	if output_file:
		plt.savefig(output_file)


def vs_two_variables_aux(
		ax, x1, x2, ys, axes_properties={},
		colors=['blue', 'black', 'magenta', 'green', 'gray', 'brown'],
		markers=['o', 's', 'd', 'h', '>', '*']):

	for y, color, marker, N in zip(ys.T, colors, markers, x2):

		ax.plot(x1, y, color=color, marker=marker, label='N = {}'.format(N))

	ax.set_xbound(lower=x1[0], upper=x1[-1])

	# the labels are shown
	ax.legend()

	ax.set(**axes_properties)


def vs_two_variables_multiple(x1, x2, ys1, ys2, id, output_file=None, axes_properties1={}, axes_properties2={}):

	fig, axes = plt.subplots(nrows=1, ncols=2)

	vs_two_variables_aux(axes[0], x1, x2, ys1, axes_properties1)
	vs_two_variables_aux(axes[1], x1, x2, ys2, axes_properties2)

	# the legend for the first axis is moved down right
	axes[0].legend(loc=4)

	fig.show()

	if output_file:

		plt.savefig(output_file)