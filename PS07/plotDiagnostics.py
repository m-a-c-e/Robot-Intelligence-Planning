import matplotlib.pyplot as plt
import numpy as np
def plotDiagnostics(L, winLossRecord, i, axes, fig, data=None, method=None):
	"""
	data : data of NN for saving the parameters in the name of plot.
	"""
	# Author: Matthew Gombolay <Matthew.Gombolay@cc.gatech.edu
	# Date: 26 JUN 2020

	# This function takes as input a vector, L, containing the loss at each
	# iteration of training, the win-loss record, winLossRecord, at each
	# iteration of training, and the number of iterations, i.

	# Plot the loss after each iteration vs. the iteration number. 
	# fig, axes = plt.subplots(1,3)
	y = L[:i]
	y_new = np.zeros((len(y), 1))
	# print(L)
	for j in range(len(y)):
		y_new[j] = np.mean(y[max(j-25, 0): min(j+25, len(y))])
	axes[1].plot(range(0,i), y_new, '-.g')  # Can only use log scale for positive values
	axes[1].set_xlabel('Iteration')
	axes[1].set_ylabel('Change in Params')
	fig.canvas.draw()
	fig.canvas.flush_events()
	fig.show()
	#plt.close(fig)

	winLossRecord_new = np.zeros((len(y), 1))
	for j in range(i):
		winLossRecord_new[j] = np.mean(winLossRecord[max(j-25, 0):min(j+25, len(winLossRecord))])

	axes[2].plot(range(0,len(winLossRecord_new)), winLossRecord_new, '-r')
	axes[2].set_ylim(0,1)
	axes[2].set_xlabel('Games Played')
	axes[2].set_ylabel('Win/Loss')
	fig.canvas.draw()
	fig.canvas.flush_events()
	fig.show()
	if method is None:
		fig.savefig('Pset7.png')
	elif method=='Reinforce':
		fig.savefig('Reinforce_plot.png')
	elif method=='A2C':
		fig.savefig('A2C_plot.png')
	else:
		fig.savefig('AC_plot.png')
	#plt.close(fig)
