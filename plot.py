import numpy as np
import matplotlib.pyplot as plt


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


def mmse(x, ys, output_file=None):

	ax, fig = setup_axes('MMSE')

	for y, label, color, marker in zip(ys.T, ['Standard', 'TIW'], ['black', 'blue'], ['s', 'o']):

		ax.loglog(x, y, label=label, color=color, marker=marker)

	# the labels are shown
	ax.legend()

	ax.set_xlabel('number of particles')
	ax.set_ylabel('MMSE')

	fig.show()

	if output_file:

		plt.savefig(output_file)


def variance(x, ys, output_file=None):

	ax, fig = setup_axes('variance')

	for alg_y, label, color, marker in zip(np.rollaxis(ys, 2), ['Standard', 'TIW'], ['black', 'blue'], ['s', 'o']):

		for y, label_suffix, linestyle in zip(alg_y, ['0', '1'], ['dotted', 'dashed']):

			ax.loglog(x, y, label=label + label_suffix, color=color, marker=marker, linestyle=linestyle)

	# the labels are shown
	ax.legend()

	if output_file:

		plt.savefig(output_file)

