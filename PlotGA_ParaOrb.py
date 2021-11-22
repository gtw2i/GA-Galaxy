# Graham West
from copy import deepcopy
import sys 
import time 
import random
import numpy as np
import numpy.linalg as LA
import math
import pandas as pd
from subprocess import call
from scipy import optimize
from scipy import misc
from matplotlib import pyplot as plt
from matplotlib import image as img

import pickle
#import cv2
import multiprocessing as mp
import threading as thr
from threading import Thread
import os
import time
import Queue

import socket as skt

import glob

from math import sin
from math import cos


##############
#    MAIN    #
##############

def main():
	
	#####################
	###   VARIABLES   ###
	#####################
	
	np.set_printoptions(precision=3, suppress=True)
	
	toPlot = 21
		
	toSave = 0
	
	velSph = 1
	
	# target ID
#	zooBase = "587722984435351614"
#	zooBase = "587736941981466667"
	zooBase = "587726033843585146"
	
	print zooBase
	
	ext     = "_combined.txt"
	pFile   = zooBase + ext
	
	fileInd   = "12"
#	fileInd   = "SeedTest"
	targetInd = 0
	
#	outBase = "Results_GA_" + zooBase + "_" + fileInd + "_"
#	outBase = "Results_GA-Paper_" + zooBase + "_" + fileInd + "_"
#	outBase = "Results_GA-Best_" + zooBase + "_" + fileInd + "_"
	
#	outBase = "Results_GA-Roughshod_" + zooBase + "_" + fileInd + "_"
#	outBase = "Results_GA-Roughshod-Steps_" + zooBase + "_" + fileInd + "_"
	outBase   = "Results_GA-Roughshod_ReuseThreads_" + zooBase + "_" + fileInd + "_"
	
	shrink = 1.0
	
	nHist = 35
	
	nBin = 35
	
#	bound = np.array([	# bin window
#		[-0.6,0.6],
#		[-0.4,0.8]])
#		[-19.0, 11.0],
#		[-18.0, 12.0]])
	
	# read zoo file
#	data, nModel, nCol = ReadAndCleanupData( pFile, velSph )
	data, psi, nModel, nCol = ReadAndCleanupData( pFile )
	psiReal = psi[targetInd,:]
	
	pReal = data[targetInd,0:-1]
	nParam = len(pReal)
	
	
	# get unique solutions
	unique = np.zeros((8,14))
	for i in range(8):
		unique[i,:] = pReal
		if(   i == 1 ):
			unique[i,13] = -unique[i,13]
		elif( i == 2 ):
			unique[i,12] = -unique[i,12]
		elif( i == 3 ):
			unique[i,13] = -unique[i,13]
			unique[i,12] = -unique[i,12]
		elif( i == 4 ):
			unique[i,5]  = -unique[i,5]
		elif( i == 5 ):
			unique[i,13] = -unique[i,13]
			unique[i,5]  = -unique[i,5]
		elif( i == 6 ):
			unique[i,12] = -unique[i,12]
			unique[i,5]  = -unique[i,5]
		elif( i == 7 ):
			unique[i,13] = -unique[i,13]
			unique[i,12] = -unique[i,12]
			unique[i,5]  = -unique[i,5]
		# end
	# end
	
	# read max xLim and window bounds
	XXX = np.loadtxt( "Bounds_" + zooBase + ".txt" )
	mmm = XXX[:nParam,:]
	bound = XXX[nParam:,:]
	"""
	# modify max xLim
	mmm[2,:]  = np.array([ -10.0, 10.0 ])
	mmm[3,:]  = np.array([ -10.0, 10.0 ])
	mmm[4,:]  = np.array([ -10.0, 10.0 ])
	mmm[5,:]  = np.array([ -10.0, 10.0 ])
	mmm[6,:]  = np.array([ 0.1, 0.9 ])
	mmm[7,:]  = np.array([ 10.0, 100.0 ])
	mmm[8,:]  = np.array([ 0.5, 10.0 ])
	mmm[9,:]  = np.array([ 0.1, 7.0 ])
	mmm[10,:] = np.array([ 0.0, 360.0 ])
	mmm[11,:] = np.array([ 0.0, 360.0 ])
	mmm[12,:] = np.array([ 0.0, 360.0 ])
	mmm[13,:] = np.array([ 0.0, 360.0 ])
	"""
	
	###############################
	###   DATA/INITIALIZATION   ###
	###############################
	
#	chain  = pickle.load( open("solutions.txt", "rb") )
#	scores = pickle.load( open("scores.txt",    "rb") )
	
	"""
	chain = []
	fileChain = open( outBase + "solutions.txt", "rb")
	while 1:
		try:
			chain.append( pickle.load( fileChain ) )
		except EOFError:
			break
		# end
	# end
	fileChain.close()
	chain = np.array(chain)
	
	scores = []
	fileScores = open( outBase + "scores.txt", "rb")
	while 1:
		try:
			scores.append( pickle.load( fileScores ) )
		except EOFError:
			break
		# end
	# end
	fileScores.close()
	scores = np.array(scores)
	
	orbitals = []
	fileOrbital = open( outBase + "elements.txt", "rb")
	while 1:
		try:
			orbitals.append( pickle.load( fileOrbital ) )
		except EOFError:
			break
		# end
	# end
	fileOrbital.close()
	orbitals = np.array(orbitals)
	"""
	
	chain    = pickle.load( open( outBase + "solutions.txt" , "rb") )
	scores   = pickle.load( open( outBase + "scores.txt"    , "rb") )
	orbitals = pickle.load( open( outBase + "elements.txt"  , "rb") )
	
	
	nGen, nPop, nParam = chain.shape
	
	indBest = np.unravel_index( np.argmax(scores, axis=None), scores.shape )
	pBest   = chain[indBest[0],indBest[1],:]
	
	print bound
	
	print "nGen / nPop / nParam"
	print chain.shape
	
	print "pBest:"
	print pBest
	print orbitals[indBest[0],indBest[1]]
	
	print "score:"
	print scores[indBest[0],indBest[1]]
	
	print "pReal:"
	print pReal
	
	thresh = 0.0
	c2 = []
	s2 = []
	for i in range(nGen):
		for j in range(nPop):
			if( scores[i,j] > thresh ):
				c2.append( chain[i,j,:] )
				s2.append( scores[i,j] )
			# end
		# end
	# end
	c2 = np.array(c2)
	s2 = np.array(s2)
	
	# std
#	print "std / width:"
	sig   = np.zeros(nParam)
	width = np.zeros(nParam)
	for i in range(nParam):
		sig[i]   = np.std(c2[:,i])
		width[i] = max(c2[:,i])-min(c2[:,i])
	# end
	
	
	####################
	###   PLOTTING   ###
	####################
	
#	labels = [ 'x', 'y', 'z', 'vx', 'vy', 'vz', 'mf', 'mt', 'rp', 'rs', 'pp', 'ps', 'tp', 'ts' ]
#	labels = [ r'$x$', r'$y$', r'$z$', r'$v_x$', r'$v_y$', r'$v_z$', r'$m_p/(m_p+m_s)$', '$m_p+m_s$', '$r_p$', '$r_s$', '$\phi_p$', '$\phi_s$', r'$\theta_p$', r'$\theta_s$' ]
	
	labels = [ r'$x$', r'$y$', r'$z$', r'$(K+U)/(K-U)$', r'$\phi_v$', r'$\theta_v$', r'$m_p/(m_p+m_s)$', '$m_p+m_s$', '$A_p$', '$A_s$', '$\phi_p$', '$\phi_s$', r'$\theta_p$', r'$\theta_s$' ]
	
	labels2 = [ r'$t_{min}$', r'$d_{min}$', r'$v_{min}$', r'$\beta$', r'$i$', r'$\omega$', r'$\theta_{AN}$' ]
	
#	plt.style.use('dark_background')
	
	if(   toPlot == 1 ):
#		fig, axes = plt.subplots(nrows=4, ncols=4)
		fig, axes = plt.subplots(nrows=3, ncols=4)
		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		ind = 0
#		for i in range(nParam+1):
		for i in [2,3,4,5,6,7,8,9,12,13,14]:
			if(   i < nParam ):
				axes[ind].plot( np.sort( chain[:,:,i], axis=1), 'b.' )
#				axes[ind].plot(  xLim[i,0]*np.ones(nGen+1), 'g' )
#				axes[ind].plot(  xLim[i,1]*np.ones(nGen+1), 'g' )
				axes[ind].plot( pReal[i]*np.ones(nGen+1), 'r-', linewidth=2 )
				axes[ind].plot( pBest[i]*np.ones(nGen+1), 'g--', linewidth=3 )
				axes[ind].set_ylabel( labels[i] )
				axes[ind].set_xlabel( "generation" )
			elif( i == nParam ):
				axes[ind].plot( np.sort( scores, axis=1), 'r.' )
				axes[ind].set_ylim( [0,       1      ] )
				axes[ind].set_ylabel( "scores" )
				axes[ind].set_xlabel( "generation" )
			# end
			ind += 1
		# end
	elif( toPlot == 2 ):
		fig, axes = plt.subplots(nrows=4, ncols=4)
		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
#		scores = (scores/np.amax(scores))**2
		
#		r    = chain[:,:,6]
#		t    = chain[:,:,7]
#		m2 = t/(r+1)
#		m1 = r*m2
#		mred = m1*m2/(m1+m2)
#		vred = chain[:,:,4]/mred**0.5
		
		for i in range(nParam):
			axes[i].plot( chain[:,:,i].flatten(), scores.flatten(), 'b.' )
			
			axes[i].plot( [ pReal[i], pReal[i] ], [0, 1], 'r--' )
			
			axes[i].plot( [ pBest[i], pBest[i] ], [0, 1], 'k--' )
			axes[i].plot( [ pBest[i]-sig[i], pBest[i]-sig[i] ], [0, 1], 'g--' )
			axes[i].plot( [ pBest[i]+sig[i], pBest[i]+sig[i] ], [0, 1], 'g--' )
			
	#		axes[i].set_xlim( [xLim[i,0], xLim[i,1]] )
			axes[i].set_ylim( [0,       1      ] )
			axes[i].set_xlabel( labels[i] )
		# end
		
#		axes[14].plot( vred.flatten(), scores.flatten(), 'b.' )
#		axes[14].plot( m1.flatten(), scores.flatten(), 'b.' )
#		axes[15].plot( m2.flatten(), scores.flatten(), 'b.' )
	elif( toPlot == 3 ):
		fig, axes = plt.subplots(nrows=3, ncols=4)
		fig.set_size_inches(12,8)
		
		Tc, Tv, Td, Tu, RVt, RVv = solve( pReal, nParam, nBin, bound, "00", psiReal )
		Mc, Mv, Md, Mu, RVm, RVu = solve( pBest, nParam, nBin, bound, "00", psiReal )
		
		
		twin = [ [min(RVt[:,0]),max(RVt[:,0])], [min(RVt[:,1]),max(RVt[:,1])] ]
		mwin = [ [min(RVm[:,0]),max(RVm[:,0])], [min(RVm[:,1]),max(RVm[:,1])] ]
		win2 = [ [min([twin[0][0],mwin[0][0]]),max([twin[0][1],mwin[0][1]])], [min([twin[1][0],mwin[1][0]]),max([twin[1][1],mwin[1][1]])] ]
		win2 = np.array(win2)
		
		T  = BinField( nBin, RVt, bound )[0]
		T1 = BinField( nBin, RVt, win2  )[0]
		M  = BinField( nBin, RVm, bound )[0]
		M1 = BinField( nBin, RVm, win2  )[0]
		U  = BinField( nBin, RVu, bound )[0]
		U1 = BinField( nBin, RVu, win2  )[0]
		
		print T.shape, M.shape, U.shape
		
		tmScore  = MachineScoreW( nBin,  T,  M, U, 8 )
		muScore  = MachineScore(  nBin, M1, U1,    8 )
		tuScore  = MachineScore(  nBin, T1, U1,    8 )
		muScoreX = perturb( muScore, tuScore )
		score    = tmScore*muScoreX	# fitting full perturbed model
		
		print "score, tmScore, muScoreX, muScore, tuScore"
		print score, tmScore, muScoreX, muScore, tuScore
		
		
		
		
		Tc = np.log(1+Tc)
		Tc = Tc
		Mc = np.log(1+Mc)
		Mc = Mc
		mm = np.max( np.abs( [ np.amax(Tv),np.amin(Tv),np.amax(Mv),np.amin(Mv) ] ) )
		Tv[0,0] = mm
		Tv[-1,-1] = -mm
		Mv[0,0] = -mm
		Mv[-1,-1] = mm
		mm = np.max( np.abs( [ np.amax(Td),np.amax(Md) ] ) )
		Td[0,0] = mm
		Md[-1,-1] = mm
		
		axes[0,0].imshow( Tc, interpolation="none", cmap="gray" )
		axes[0,0].set_title("T")
		
		axes[1,0].imshow( Mc, interpolation="none", cmap="gray" )
		axes[1,0].set_title("M")
		
		axes[2,0].imshow( Tc-Mc, interpolation="none", cmap="bwr" )
		axes[2,0].set_title("T-M")
		
		h        = 0
		Tc[Tc>h] = 1
		Mc[Mc>h] = 1
		
		axes[0,1].imshow( Tc, interpolation="none", cmap="gray" )
		axes[1,1].imshow( Mc, interpolation="none", cmap="gray" )
		axes[2,1].imshow( Tc-Mc, interpolation="none", cmap="bwr" )
		axes[2,1].set_title("T-M")
		
		axes[0,2].imshow( Tv, interpolation="none", cmap="bwr" )
		axes[0,2].set_title("Tv")
		axes[1,2].imshow( Mv, interpolation="none", cmap="bwr" )
		axes[1,2].set_title("Mv")
		axes[2,2].imshow( Tv-Mv, interpolation="none", cmap="bwr" )
		axes[2,2].set_title("Tv-Mv")
		
		axes[0,3].imshow( Td, interpolation="none", cmap="jet" )
		axes[0,3].set_title("T disp")
		axes[1,3].imshow( Md, interpolation="none", cmap="jet" )
		axes[1,3].set_title("M disp")
		axes[2,3].imshow( Td-Md, interpolation="none", cmap="bwr" )
		axes[2,3].set_title("Td-Md")
		
#	pBest   = chain[indBest[0],indBest[1],:]
#		plt.tight_layout( w_pad=-10.0, h_pad=0.5 )
	elif( toPlot == 4 ):
		fig, axes = plt.subplots(nrows=4, ncols=4)
		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		thresh = 0.78
		inds   = np.where( scores >= thresh )
		inds   = np.array(inds)
		
		params = []
		for i in [2,3,4,5,6,7,8,9,12,13]:
			params.append( chain[inds[0,:],inds[1,:],i] )
		# end
		params = np.array(params)
		
		rho = np.corrcoef( params )
		print2D( rho )
		
		w, v = LA.eig( rho )
		print w
		print2D( v )
		
		for i in range(nParam):
			axes[i].plot( chain[inds[0,:],inds[1,:],i].flatten(), scores[inds[0,:],inds[1,:]].flatten(), 'b.' )
			
			axes[i].plot( [ pReal[i], pReal[i] ], [0, 1], 'r--' )
			
			axes[i].set_xlim( [xLim[i,0], xLim[i,1]] )
			axes[i].set_ylim( [0,       1      ] )
			axes[i].set_xlabel( labels[i] )
		# end
	elif( toPlot == 5 ):
		fig, axes = plt.subplots(nrows=4, ncols=4)
		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		for i in range(nParam):
			x, y = np.histogram( chain[:,:,i].flatten(), nHist )
			M = max(x)
			axes[i].hist( chain[:,:,i].flatten(), nHist )
			
			axes[i].plot( [ pReal[i], pReal[i] ], [0, M], 'r-' )
			axes[i].plot( [ pBest[i], pBest[i] ], [0, M], 'k-' )
			
			axes[i].set_xlim( [xLim[i,0], xLim[i,1]] )
			axes[i].set_xlabel( labels[i] )
		# end
	elif( toPlot == 6 ):
		print np.histogram( scores.flatten(), nHist )
		plt.hist( scores.flatten(), nHist )
		plt.xlim( [0,       1      ] )
	elif( toPlot == 7 ):
#		pList = [2,3,4,5,6,7]
#		pList = [3,4,5]
#		pList = [2,3,4,5,6]
#		pList = [2,5]
#		pList = [2,3]
#		pList = [8,9]
#		pList = [2,5,6,7]
#		pList = [2,5,12,13]
#		pList = [2,3,4,5,6,12,13]
		pList = [4,5,10,11,12,13]
#		pList = [8,9,10,11,12,13]
#		pList = [3,4,5,7,8,9]
#		pList = [2,3,4,5,7,8,9,10,11,12,13]
		
		fig, axes = plt.subplots(nrows=len(pList), ncols=len(pList))
#		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		thresh = 0.0
#		thresh = 0.8
		inds   = np.where( scores >= thresh )
#		inds   = np.where( scores <= thresh )
		inds   = np.array(inds)
		
		print np.array(inds.shape)[1]
		
#		inds   = np.where( chain[:,:,12] < 100 )
#		inds   = np.array(inds)
		
#		scores = (scores/np.amax(scores))**2
		
		params = []
		for i in pList:
			params.append( chain[inds[0,:],inds[1,:],i] )
		# end
		params = np.array(params)
		
		rho = np.corrcoef( params )
		print2D( rho )
		
		w, v = LA.eig( rho )
		print w
#		print2D( v )
		
		print pReal
		
		"""
		nnn = 100
		vz = np.zeros(nnn)
		zz = np.linspace(0,30,nnn)
		for i in range(nnn):
			vz[i] = ( ( LA.norm(pReal[3:6]) * LA.norm(pReal[0:3]) )**2 / (pReal[0]**2+pReal[1]**2+zz[i]**2) - pReal[3]**2 - pReal[4]**2 )
		# end
		axes[1,0].plot(  zz,  vz, 'k', linewidth=3 )
		axes[1,0].plot( -zz,  vz, 'k', linewidth=3 )
		axes[1,0].plot(  zz, -vz, 'k', linewidth=3 )
		axes[1,0].plot( -zz, -vz, 'k', linewidth=3 )
		"""
		
		ii = 0
		for i in pList:
			jj = 0
			for j in pList:
				II = axes[jj,ii].scatter( params[ii,:], params[jj,:], c=scores[inds[0,:],inds[1,:]], s=4, cmap='jet' )
				
				axes[jj,ii].plot( pReal[i], pReal[j], 'kX' )
#				axes[jj,ii].plot(        0,        0, 'ko' )
				
				II.set_clim( [ 0.0, 1.0 ] )
				if( ii == len(pList)-1 ):
					cbar = fig.colorbar( II, ax=axes[jj,ii] )
				# end
				
#				axes[jj,ii].set_xlim( [xLim[i,0], xLim[i,1]] )
#				axes[jj,ii].set_ylim( [xLim[j,0], xLim[j,1]] )
				if( jj == len(pList)-1 ):
					axes[jj,ii].set_xlabel( labels[i] )
				else:
					axes[jj,ii].set_xticks([],[])
				# end
				if( ii == 0 ):
					axes[jj,ii].set_ylabel( labels[j] )
				else:
					axes[jj,ii].set_yticks([],[])
				# end
				
				jj += 1
			# end
			
			ii += 1
		# end
		
		plt.tight_layout( w_pad=-0.1, h_pad=-0.5 )
	elif( toPlot == 8 ):
		pList = [2,3,4,5,6,7]
		
		fig, axes = plt.subplots(nrows=len(pList), ncols=len(pList))
#		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		nn = 20
		
		for kk in range(nn):
			thresh = 1 - np.amax(scores)*kk/(nn-1.0)
			inds   = np.where( scores >= thresh )
#			inds   = np.where( scores <= thresh )
			inds   = np.array(inds)
			print np.array(inds.shape)[1], thresh
			
#			scores = (scores/np.amax(scores))**2
			
			params = []
			for i in pList:
				params.append( chain[inds[0,:],inds[1,:],i] )
			# end
			params = np.array(params)
			
			ii = 0
			for i in pList:
				jj = 0
				for j in pList:
					axes[jj,ii].clear()
					
#					II = axes[jj,ii].scatter( params[ii,:], params[jj,:], c=scores[inds[0,:],inds[1,:]], s=2, cmap='jet' )
					II = axes[jj,ii].scatter( params[ii,:], params[jj,:], c=(scores[inds[0,:],inds[1,:]]/np.amax(scores))**2, s=2, cmap='jet' )
					
					axes[jj,ii].plot( pReal[i], pReal[j], 'kX' )
					
					II.set_clim( [ 0.0, 1.0 ] )
#					if( ii == len(pList)-1 ):
#						cbar = fig.colorbar( II, ax=axes[jj,ii] )
					# end
					
					axes[jj,ii].set_xlim( [xLim[i,0], xLim[i,1]] )
					axes[jj,ii].set_ylim( [xLim[j,0], xLim[j,1]] )
					if( jj == len(pList)-1 ):
						axes[jj,ii].set_xlabel( labels[i] )
					# end
					if( ii == 0 ):
						axes[jj,ii].set_ylabel( labels[j] )
					# end
					
					jj += 1
				# end
				
				ii += 1
			# end
			plt.pause(3.0)
		# end
	elif( toPlot == 9 ):
		print Ms.shape
		
		h1         = 0
		Ms[Ms>h1]  = 1
		Ms[Ms<=h1] = 0
		
		h2         = 0.91
		
		tot = 0
		X = np.zeros((nBin,nBin))
		for i in range(nPop):
			for j in range(nGen):
				if( scores[j,i] >= h2 ):
					X += Ms[j,i,:,:]
					tot += 1
				# end
			# end
		# end
		X = X/tot
		print tot
		print X
		
		plt.imshow( X, interpolation="none", cmap="gray" )
	elif( toPlot == 10 ):
		fig, axes = plt.subplots(nrows=2, ncols=1)
		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		T, V = solve( pReal, nParam, nBin, bound )
		axes[0].imshow( np.log(1+T), interpolation="none", cmap="gray" )
		axes[0].set_title("Target (log scale)")
		
		M, U = solve( pBest, nParam, nBin, bound )
		axes[1].imshow( np.log(1+M), interpolation="none", cmap="gray" )
		axes[1].set_title("Best model found (log scale)")
	elif( toPlot == 11 ):
#		fig, axes = plt.subplots(nrows=4, ncols=4)
		fig, axes = plt.subplots(nrows=3, ncols=4)
		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		gens = []
		for i in range(nGen):
			gens.append( i*np.ones(nPop) )
		# end
		gens = np.array(gens)
		
		print gens.shape
		print chain[:,:,0].shape
		
		ind = 0
#		for i in range(nParam+1):
		for i in [2,3,4,5,6,8,9,10,11,12,13,14]:
			if(   i < nParam ):
#				axes[ind].plot( chain[:,:,i], 'b.' )
				II = axes[ind].scatter( gens.flatten(), chain[:,:,i].flatten(), c=scores.flatten(), s=4, cmap='jet' )
				
				II.set_clim( [ 0.0, 1.0 ] )
				
#				for j in range(8):
#					axes[ind].plot( unique[j,i]*np.ones(nGen+1), 'k-', linewidth=2 )
#				# end
				axes[ind].plot( pReal[i]*np.ones(nGen+1), 'k-',  linewidth=3, alpha=0.5 )
				axes[ind].plot( pBest[i]*np.ones(nGen+1), 'k--', linewidth=3, alpha=0.5 )
				axes[ind].set_ylabel( labels[i] )
				axes[ind].set_xlabel( "generation" )
			elif( i == nParam ):
				II = axes[ind].scatter( gens.flatten(), scores.flatten(), c=scores.flatten(), s=8, cmap='jet' )
				II.set_clim( [ 0.0, 1.0 ] )
				
				axes[ind].set_ylim( [0,       1      ] )
				axes[ind].set_ylabel( "scores" )
				axes[ind].set_xlabel( "generation" )
			# end
			ind += 1
		# end
	elif( toPlot == 12 ):
		pList = [2,5]
		
		fig, axes = plt.subplots(nrows=len(pList), ncols=len(pList))
#		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		gens = []
		for i in range(nGen):
			gens.append( i*np.ones(nPop) )
		# end
		gens = np.array(gens)
		
		for kk in range(nGen):
			ii = 0
			for i in pList:
				jj = 0
				for j in pList:
					axes[jj,ii].clear()
					
					II = axes[jj,ii].scatter( chain[0:kk,:,i].flatten(), chain[0:kk,:,j].flatten(), c=scores[0:kk,:].flatten(), s=8, cmap='jet' )
					
					axes[jj,ii].plot( pReal[i], pReal[j], 'kX' )
					
					II.set_clim( [ 0.0, 1.0 ] )
					
					axes[jj,ii].set_xlim( [xLim[i,0], xLim[i,1]] )
					axes[jj,ii].set_ylim( [xLim[j,0], xLim[j,1]] )
					if( jj == len(pList)-1 ):
						axes[jj,ii].set_xlabel( labels[i] )
					# end
					if( ii == 0 ):
						axes[jj,ii].set_ylabel( labels[j] )
					# end
					
					jj += 1
				# end
				
				ii += 1
			# end
			plt.pause(0.1)
		# end
	elif( toPlot == 13 ):
#		fig, axes = plt.subplots(nrows=4, ncols=4)
		fig, axes = plt.subplots(nrows=3, ncols=4)
		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		gens = []
		for i in range(nGen):
			gens.append( i*np.ones(nPop) )
		# end
		gens = np.array(gens)
		
		print gens.shape
		print chain[:,:,0].shape
		
		ind = 0
#		for i in range(nParam+1):
		for i in [2,3,4,5,6,8,9,10,11,12,13,14]:
			if(   i < nParam ):
#				axes[ind].plot( chain[:,:,i], 'b.' )
				II = axes[ind].scatter( gens.flatten(), chain[:,:,i].flatten(), c=scores.flatten(), s=8, cmap='jet' )
				
				II.set_clim( [ 0.0, 1.0 ] )
				
				axes[ind].plot( pReal[i]*np.ones(nGen+1), 'k-', linewidth=2 )
				axes[ind].plot( pBest[i]*np.ones(nGen+1), 'k--', linewidth=2 )
				axes[ind].set_ylabel( labels[i] )
				axes[ind].set_xlabel( "generation" )
			elif( i == 14 ):
				II = axes[ind].scatter( gens.flatten(), scores.flatten(), c=scores.flatten(), s=8, cmap='jet' )
				II.set_clim( [ 0.0, 1.0 ] )
				
				cbar = fig.colorbar( II, ax=axes[ind] )
				
				axes[ind].set_ylim( [0,       1      ] )
				axes[ind].set_ylabel( "scores" )
				axes[ind].set_xlabel( "generation" )
			# end
			ind += 1
		# end
	elif( toPlot == 14 ):
		fig, axes = plt.subplots(nrows=2, ncols=2)
#		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		T, V = solve( pReal, nParam, nBin, bound )
		
		axes[0,0].imshow( T,           interpolation="none", cmap="gray" )
		axes[0,0].set_title("Target")
		
		M, U = solve( pBest, nParam, nBin, bound )
		
		axes[0,1].imshow( M,           interpolation="none", cmap="gray" )
		axes[0,1].set_title("Best-fit model")
		
		axes[1,0].imshow( np.log(1+T), interpolation="none", cmap="gray" )
		axes[1,0].set_title("Target (log scale)")
		
		axes[1,1].imshow( np.log(1+M), interpolation="none", cmap="gray" )
		axes[1,1].set_title("Best-fit model (log scale)")
	elif( toPlot == 15 ):
		fig, axes = plt.subplots(nrows=5, ncols=5)
		fig.set_size_inches(12,8)
		
		T, V = solve( pReal, nParam, nBin, bound )
		T = np.log(1+T)
		V = np.log(1+V)
		M, U = solve( pBest, nParam, nBin, bound )
		M = np.log(1+M)
		U = np.log(1+U)
		
		imgs = [M,U,T,V]
		imgs = np.array(imgs)
		labs = ["M","U","T","V"]
		
		maxx = np.max(np.abs(imgs))
		imgs[:,0,0] = maxx
		imgs[:,-1,-1] = 0
		
		axes[0,0].imshow( np.zeros((10,10)), interpolation="none", cmap="gray" )
		axes[0,0].set_xticks([])
		axes[0,0].set_yticks([])
		
		for i in range(4):
			axes[i+1,0].imshow( imgs[i], interpolation="none", cmap="gray" )
			axes[i+1,0].set_ylabel(labs[i])
			axes[i+1,0].set_xticks([])
			axes[i+1,0].set_yticks([])
			axes[0,i+1].imshow( imgs[i], interpolation="none", cmap="gray" )
			axes[0,i+1].set_title(labs[i])
			axes[0,i+1].set_xticks([])
			axes[0,i+1].set_yticks([])
		# end
		
		imgs[:,-1,-1] = -maxx
		
		for i in range(4):
			for j in range(4):
				axes[i+1,j+1].imshow( np.abs(imgs[i]-imgs[j]), interpolation="none", cmap="bwr" )
				axes[i+1,j+1].set_xticks([])
				axes[i+1,j+1].set_yticks([])
			# end
		# end
		
		plt.tight_layout( w_pad=-45.0, h_pad=-15.0 )
	elif( toPlot == 16 ):
#		fig, axes = plt.subplots(nrows=4, ncols=4)
		fig, axes = plt.subplots(nrows=4, ncols=4)
		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
		gens = []
		for i in range(nGen):
			gens.append( i*np.ones(nPop) )
		# end
		gens = np.array(gens)
		
		print gens.shape
		print chain[:,:,0].shape
		
		ind = 0
#		for i in range(nParam+1):
		for i in [2,3,4,5,6,7,8,9,10,11,12,13]:
			II = axes[ind].scatter( data[:,i], data[:,-1], c=data[:,-1], s=4, cmap='jet' )
			
			II.set_clim( [ 0.0, 1.0 ] )
			
			axes[ind].set_xlabel( labels[i] )
			axes[ind].set_ylabel( "zoo score" )
			ind += 1
		# end
	elif( toPlot == 17 ):
		fig = plt.figure()
		fig.set_size_inches(11,11)
		ax = Axes3D(fig)
		
		orbitR = solveOrb( pReal, nParam )
		orbitB = solveOrb( pBest, nParam )
		
#		ax.plot( RVv[:nPts/2,0], RVv[:nPts/2,1], RVv[:nPts/2,2], 'k.'  )
#		ax.plot( RVv[nPts/2:,0], RVv[nPts/2:,1], RVv[nPts/2:,2], 'k.'  )
		
		ax.plot( orbitR[:,0], orbitR[:,1], orbitR[:,2], 'r', linewidth=6 )
		ax.plot( orbitR[:,0], orbitB[:,1], orbitB[:,2], 'b', linewidth=6 )
		
		ax.plot( [0], [0], [0], 'kX', markersize=8 )
		
		ax.set_xlabel( 'x' )
		ax.set_ylabel( 'y' )
		ax.set_zlabel( 'z' )
	elif( toPlot == 18 ):
		fig, axes = plt.subplots(nrows=2, ncols=2)
		fig.set_size_inches(12,8)
		
		Tc, Tv, Td, Tu, RVt, RVv = solve( pReal, nParam, nBin, bound, "00", psiReal )
		Mc, Mv, Md, Mu, RVm, RVu = solve( pBest, nParam, nBin, bound, "00", psiReal )
		
		nPts, xxx = RVt.shape
		
		a = [min(RVm[:,0]), max(RVm[:,0])]
		b = [min(RVm[:,1]), max(RVm[:,1])]
		
		axes[0,0].plot( RVt[0:nPts/2,0],  RVt[0:nPts/2,1],  'b.' )
		axes[0,0].plot( RVt[nPts/2:-1,0], RVt[nPts/2:-1,1], 'r.' )
		axes[0,0].set_xlim(bound[0,:])
		axes[0,0].set_ylim(bound[1,:])
		
		axes[0,1].plot( RVt[0:nPts/2,0],  RVt[0:nPts/2,1],  'b.' )
		axes[0,1].plot( RVt[nPts/2:-1,0], RVt[nPts/2:-1,1], 'r.' )
		axes[0,1].set_xlim(a)
		axes[0,1].set_ylim(b)
		
		axes[1,0].plot( RVm[0:nPts/2,0],  RVm[0:nPts/2,1],  'b.' )
		axes[1,0].plot( RVm[nPts/2:-1,0], RVm[nPts/2:-1,1], 'r.' )
		axes[1,0].set_xlim(bound[0,:])
		axes[1,0].set_ylim(bound[1,:])
		
		axes[1,1].plot( RVm[0:nPts/2,0],  RVm[0:nPts/2,1],  'b.' )
		axes[1,1].plot( RVm[nPts/2:-1,0], RVm[nPts/2:-1,1], 'r.' )
		axes[1,1].set_xlim(a)
		axes[1,1].set_ylim(b)
	elif( toPlot == 19 ):
		fig, axes = plt.subplots(nrows=2, ncols=3)
		fig.set_size_inches(12,8)
		
		Tc, Tv, Td, Tu, RVt, RVv = solve( pReal, nParam, nBin, bound, "00", psiReal )
		Mc, Mv, Md, Mu, RVm, RVu = solve( pBest, nParam, nBin, bound, "00", psiReal )
		
		
#		twin = [ [min(RVt[:,0]),max(RVt[:,0])], [min(RVt[:,1]),max(RVt[:,1])] ]
		mwin = [ [min(RVm[:,0]),max(RVm[:,0])], [min(RVm[:,1]),max(RVm[:,1])] ]
		win2 = [ [min([bound[0][0],mwin[0][0]]),max([bound[0][1],mwin[0][1]])], [min([bound[1][0],mwin[1][0]]),max([bound[1][1],mwin[1][1]])] ]
		win2 = np.array(win2)
		
		T  = BinField( nBin, RVt, bound )[0]
		T1 = BinField( nBin, RVt, win2  )[0]
		M  = BinField( nBin, RVm, bound )[0]
		M1 = BinField( nBin, RVm, win2  )[0]
		U  = BinField( nBin, RVu, bound )[0]
		U1 = BinField( nBin, RVu, win2  )[0]
		
		tmScore  = MachineScoreW( nBin,  T,  M, U, 8 )
		muScore  = MachineScore(  nBin, M1**0.02, U1**0.02,    8 )
		tuScore  = MachineScore(  nBin, T1**0.02, U1**0.02,    8 )
		muScoreX = perturb( muScore, tuScore )
		score    = tmScore*muScoreX	# fitting full perturbed model
		
		print "score, tmScore, muScoreX, muScore, tuScore"
		print score, tmScore, muScoreX, muScore, tuScore
		
		T = np.log(1+T)
		T1 = np.log(1+T1)
		M = np.log(1+M)
		M1 = np.log(1+M1)
		U = np.log(1+U)
		U1 = np.log(1+U1)
		
		axes[0,0].imshow( T, interpolation="none", cmap="gray" )
		axes[0,0].set_title("T")
		
		axes[1,0].imshow( T1, interpolation="none", cmap="gray" )
		axes[1,0].set_title("T1")
		
		axes[0,1].imshow( M, interpolation="none", cmap="gray" )
		axes[0,1].set_title("M")
		
		axes[1,1].imshow( M1, interpolation="none", cmap="gray" )
		axes[1,1].set_title("M1")
		
		axes[0,2].imshow( U, interpolation="none", cmap="gray" )
		axes[0,2].set_title("U")
		
		axes[1,2].imshow( U1, interpolation="none", cmap="gray" )
		axes[1,2].set_title("U1")
		
		plt.tight_layout( w_pad=-10.0, h_pad=0.5 )
	elif( toPlot == 20 ):	
#		fig, axes = plt.subplots(nrows=4, ncols=4)
		fig, axes = plt.subplots(nrows=4, ncols=5)
		axes = axes.flatten()
		fig.set_size_inches(12,8)
		
#		pReal2 = np.zeros(nParam+7)
#		pReal2[:nParam] = pReal
		
		tmin, dmin, vmin, rmin = solveMod( pReal, nParam, psiReal )
		inc, argPer, longAN = getOrbitalElements( pReal, rmin )
		
		orbReal = np.zeros(7)
		orbReal[0] = tmin
		orbReal[1] = dmin
		orbReal[2] = vmin
		orbReal[3] = (pReal[6]+pReal[7])/(vmin*dmin**2)
		orbReal[4] = inc
		orbReal[5] = argPer
		orbReal[6] = longAN
		
		orbBest    = orbitals[indBest[0],indBest[1],:]
		
		print "-----------"
		
		print "x y mt"
		print pReal
		print pBest
		print "orbssssssssss"
		print orbReal
		print orbBest
		
		
		gens = []
		for i in range(nGen):
			gens.append( i*np.ones(nPop) )
		# end
		gens = np.array(gens)
		
		j = 2
		for i in range(nParam-3):
			print labels[j]
			print j
			if( j == 7 ):
				j += 1
			# end
			II = axes[i].scatter( gens.flatten(), chain[:,:,j].flatten(), c=scores.flatten(), s=8, cmap='plasma' )
			
			II.set_clim( [ 0.0, 1.0 ] )
			
			axes[i].plot( pReal[j]*np.ones(nGen+1), 'g-',  linewidth=2 )
			axes[i].plot( pBest[j]*np.ones(nGen+1), 'c--', linewidth=2 )
			axes[i].set_ylabel( labels[j] )
			axes[i].set_xlabel( "generation" )
			j += 1
		# end
		
		print gens.shape, orbitals.shape, scores.shape
		
		for i in range(7):
			II = axes[i+nParam-3].scatter( gens.flatten(), orbitals[:,:,i].flatten(), c=scores.flatten(), s=8, cmap='plasma' )
			
			II.set_clim( [ 0.0, 1.0 ] )
			
			axes[i+nParam-3].plot( orbReal[i]*np.ones(nGen+1), 'g-',  linewidth=2 )
			axes[i+nParam-3].plot( orbBest[i]*np.ones(nGen+1), 'c--', linewidth=2 )
			axes[i+nParam-3].set_ylabel( labels2[i] )
			axes[i+nParam-3].set_xlabel( "generation" )
			
			if( i == 3 ):
				axes[i+nParam-3].set_yscale('log')
			# end
		# end
		
		II = axes[18].scatter( gens.flatten(), scores.flatten(), c=scores.flatten(), s=8, cmap='plasma' )
		II.set_clim( [ 0.0, 1.0 ] )
		cbar = fig.colorbar( II, ax=axes[18] )
		
		axes[18].set_ylim( [0,       1      ] )
		axes[18].set_ylabel( "scores" )
		axes[18].set_xlabel( "generation" )
	elif( toPlot == 21 ):
		fig, axes = plt.subplots(nrows=2, ncols=2)
#		fig.set_size_inches(10,10)
		
		Tc, Tv, Td, Tu, RVt, RVv = solve( pReal, nParam, nBin, bound, "00", psiReal )
		Mc, Mv, Md, Mu, RVm, RVu = solve( pBest, nParam, nBin, bound, "00", psiReal )
		
		nPts = RVt.shape[0]
		
		Tc = np.log(1+Tc)
		Tc = Tc
		Mc = np.log(1+Mc)
		Mc = Mc
		
		mm = np.max( np.abs( [ np.amax(Tv),np.amin(Tv),np.amax(Mv),np.amin(Mv) ] ) )
		Tv[0,0] = mm
		Tv[-1,-1] = -mm
		Mv[0,0] = -mm
		Mv[-1,-1] = mm
		mm = np.max( np.abs( [ np.amax(Td),np.amax(Md) ] ) )
		Td[0,0] = mm
		Md[-1,-1] = mm
		
		axes[0,0].imshow( np.flipud(Tc), interpolation="none", cmap="gray", aspect=(bound[1,1]-bound[1,0])/(bound[0,1]-bound[0,0]) )
		axes[0,0].set_title("Target")
		axes[0,0].set_xticks([])
		axes[0,0].set_yticks([])
		
		axes[1,0].imshow( np.flipud(Mc), interpolation="none", cmap="gray", aspect=(bound[1,1]-bound[1,0])/(bound[0,1]-bound[0,0]) )
		axes[1,0].set_title("Model")
		axes[1,0].set_xticks([])
		axes[1,0].set_yticks([])
		
		axes[0,1].plot( RVt[0:nPts/2,0],  RVt[0:nPts/2,1],  'b.', markersize=0.3 )
		axes[0,1].plot( RVt[nPts/2:-1,0], RVt[nPts/2:-1,1], 'r.', markersize=0.3 )
		axes[0,1].set_xlim(bound[0,:])
		axes[0,1].set_ylim(bound[1,:])
		axes[0,1].set_title("Target")
		axes[0,1].set_aspect((bound[1,1]-bound[1,0])/(bound[0,1]-bound[0,0]))
		
		axes[1,1].plot( RVm[0:nPts/2,0],  RVm[0:nPts/2,1],  'b.', markersize=0.3 )
		axes[1,1].plot( RVm[nPts/2:-1,0], RVm[nPts/2:-1,1], 'r.', markersize=0.3 )
		axes[1,1].set_xlim(bound[0,:])
		axes[1,1].set_ylim(bound[1,:])
		axes[1,1].set_title("Model")
		axes[1,1].set_aspect((bound[1,1]-bound[1,0])/(bound[0,1]-bound[0,0]))
		
		plt.tight_layout( w_pad=-5, h_pad=1.0 )
#		plt.tight_layout( w_pad=0, h_pad=5.0 )
	# end	
	
	if( toPlot not in [ 3, 7, 15, 19, 21 ] ):
		plt.tight_layout( w_pad=0.5, h_pad=0.5 )
#		plt.tight_layout( w_pad=-1.0, h_pad=0.5 )
	# end
	
	if( toSave == 1 ):
		if(   toPlot == 20 ):
			plt.savefig( outBase + "GAConvPlot.png" )
		elif( toPlot == 21 ):
			plt.savefig( outBase + "TMPlot.png" )
		# end
	else:
		plt.show()
	# end
	
	
##############
#  END MAIN  #
##############

def getOrbitalElements( pReal2, rmin ):
	
	p = deepcopy(pReal2)
	
	# convert mass units
	f    = p[6]
	t    = p[7]
	p[6] = f*p[7]
	p[7] = (1.0-f)*p[7]
	
	G = 1
	c = p[3]
	v = ( (1+c)/(1-c)*2*G*p[6]/(p[0]**2+p[1]**2+p[2]**2)**0.5 )**0.5
	
	vx = v*math.cos(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vy = v*math.sin(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vz = v*                           math.sin(p[5]*np.pi/180.0)
	p[3] = vx
	p[4] = vy
	p[5] = vz
	
	Ap = p[8]
	As = p[9]
	p[8] = np.abs(Ap/np.cos(p[12]*np.pi/180.0))**0.5
	p[9] = np.abs(As/np.cos(p[13]*np.pi/180.0))**0.5
	
	pReal = deepcopy(p)
	
	# ---------------
	
	pp = pReal[10]*np.pi/180.0
	sp = pReal[11]*np.pi/180.0
	pt = pReal[12]*np.pi/180.0
	st = pReal[13]*np.pi/180.0
	
	origin = np.zeros(3)
	
	secCen = pReal[0:3]
#	secCen = RVt[-1,:3]
	secVel = pReal[3:6]
#	secVel = RVt[-1,3:]
	secVm  = ( secVel[0]**2 + secVel[1]**2 + secVel[2]**2 )**0.5
	secVn  = secVel/secVm
	
	pVec   = np.array( [ sin(pt)*cos(pp), sin(pt)*sin(pp), cos(pt) ] )
	
	sVec   = np.array( [ sin(st)*cos(sp), sin(st)*sin(sp), cos(st) ] )
	
	oVec   = np.cross( secCen, secVel )
	oVec   = oVec/LA.norm(oVec)
	
	inc = math.acos( np.dot( oVec, pVec ) )/np.pi*180.0
	
	ascNode = np.cross( pVec, oVec )
	ascNode = ascNode/LA.norm(ascNode)
	
	argPer = math.acos( np.dot( ascNode, rmin[:3]/LA.norm(rmin[:3]) ) )*180.0/np.pi
	
	refDir  = np.array( [ cos(pt)*cos(pp), cos(pt)*sin(pp), -sin(pt) ] )
	
	xxx = np.cross( refDir, ascNode )
	yyy = np.dot( pVec, xxx )
	if( yyy >= 0 ):
		longAN = math.acos( np.dot( refDir, ascNode ) )*180.0/np.pi
	else:
		longAN = 360 - math.acos( np.dot( refDir, ascNode ) )*180.0/np.pi
	# end
	
	return inc, argPer, longAN
# end

def solveMod( param, nParam, psi ):
	
	p = deepcopy(param)
	
	# add for psi
	if( psi[0] == -1 ):
		p[12] += 180.0
	# end
	if( psi[1] == -1 ):
		p[13] += 180.0
	# end
	
	# convert mass units
	f    = p[6]
	t    = p[7]
	p[6] = f*p[7]
	p[7] = (1.0-f)*p[7]
	
	G = 1
	c = p[3]
	v = ( (1+c)/(1-c)*2*G*p[6]/(p[0]**2+p[1]**2+p[2]**2)**0.5 )**0.5
	
	vx = v*math.cos(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vy = v*math.sin(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vz = v*                           math.sin(p[5]*np.pi/180.0)
	p[3] = vx
	p[4] = vy
	p[5] = vz
	
	Ap = p[8]
	As = p[9]
	p[8] = np.abs(Ap/np.cos(p[12]*np.pi/180.0))**0.5
	p[9] = np.abs(As/np.cos(p[13]*np.pi/180.0))**0.5
	
	paramStr = ','.join( map(str, p[0:nParam]) ) + ",0"
	
	call("./mod_run " + paramStr + " > SolveMetro.out", shell=True)
	
	output = np.loadtxt("rmin.txt")
	
	tmin = output[0]
	dmin = output[1]
	rmin = output[2:]
	vmin = LA.norm(output[5:])
	
	return tmin, dmin, vmin, rmin
	
# end

def solveMod_parallel( XXX, nParam, qOut, psi ):
	
	index = XXX[0]
	param = XXX[1]
	
	p = deepcopy(param)
	
	# add for psi
	if( psi[0] == -1 ):
		p[12] += 180.0
	# end
	if( psi[1] == -1 ):
		p[13] += 180.0
	# end
	
	# convert mass units
	f    = p[6]
	t    = p[7]
	p[6] = f*p[7]
	p[7] = (1.0-f)*p[7]
	
	G = 1
	c = p[3]
	v = ( (1+c)/(1-c)*2*G*p[6]/(p[0]**2+p[1]**2+p[2]**2)**0.5 )**0.5
	
	vx = v*math.cos(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vy = v*math.sin(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vz = v*                           math.sin(p[5]*np.pi/180.0)
	p[3] = vx
	p[4] = vy
	p[5] = vz
	
	Ap = p[8]
	As = p[9]
	p[8] = np.abs(Ap/np.cos(p[12]*np.pi/180.0))**0.5
	p[9] = np.abs(As/np.cos(p[13]*np.pi/180.0))**0.5
	
	paramStr = ','.join( map(str, p[0:nParam]) ) + ",0"
	call("./mod_run " + paramStr + " > SolveMetro.out", shell=True)
	output = np.loadtxt("rmin.txt")
	
	tmin = output[0]
	dmin = output[1]
	rmin = output[2:]
	vmin = LA.norm(output[5:])
	beta = (p[6]+p[7])/(vmin*dmin**2)
	
	inc, argPer, longAN = getOrbitalElements( p, rmin )
	
	qOut.put( [index, tmin, dmin, vmin, beta, inc, argPer, longAN] )
	
	return 0
# end

def solveOrb( param, nParam ):
	
	p = deepcopy(param)
	
	# convert mass units
	f    = p[6]
	t    = p[7]
	p[6] = f*p[7]
	p[7] = (1.0-f)*p[7]
	
	G = 1
	c = p[3]
	# K/U
#	v = ( 2*G*p[6]*(np.exp(c)-1)/(p[0]**2+p[1]**2+p[2]**2)**0.5 )**0.5
	# K+U/K-U
	v = ( (1+c)/(1-c)*2*G*p[6]/(p[0]**2+p[1]**2+p[2]**2)**0.5 )**0.5
	
	"""
	c = p[2]
	v = p[3]
	z = ( ( 2*G*p[6]*(np.exp(c)-1)/p[3]**2 )**2 - p[0]**2 - p[1]**2 )**0.5
	if( np.isreal(z) ):
		p[2] = z
	else:
		p[2] = 0
	# end
	"""
	
	vx = v*math.cos(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vy = v*math.sin(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vz = v*                           math.sin(p[5]*np.pi/180.0)
	p[3] = vx
	p[4] = vy
	p[5] = vz
	
	Ap = p[8]
	As = p[9]
	p[8] = np.abs(Ap/np.cos(p[12]*np.pi/180.0))**0.5
	p[9] = np.abs(As/np.cos(p[13]*np.pi/180.0))**0.5
	
	paramStr = ','.join( map(str, p[0:nParam]) )
	
	call("./orb_run " + paramStr + " > SolveMetro.out", shell=True)
	
	orbit = np.loadtxt( "orbit.txt" )
	
	return orbit
	
# end

def perturb( pm, pt ):
	if( pm <= pt ):
		r = pm/pt
	else:
		r = (1.0-pm)/(1.0-pt)
	# end
	return r
# end

def rotMat( axis, theta ):
	theta = theta * np.pi / 180.0
	axis = axis/LA.norm(axis)
	a = math.cos( theta / 2.0 )
	b, c, d = -axis * math.sin( theta / 2.0 )
	aa, bb, cc, dd = a**2, b**2, c**2, d**2
	bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
	
	M = [	[ aa+bb-cc-dd, 2*(bc+ad), 2*(bc-ad) ],
		[ 2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab) ],
		[ 2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc ] ]
	
	return np.array(M)
# end

def sigm( x ):
	return 1.0 / ( 1.0 + np.exp(-x) )
# end

def relax( x, a ):
	return min( x/a, 1 )
# end

def relax2( dx, h ):
	if( abs(dx) < h ):
		return 0
	else:
		return abs(dx)-h
	# end
# end

def MachineScore( nBin, binCt, binCm, scr ):
	
	T = deepcopy(binCt)
	M = deepcopy(binCm)
	
	if( scr == 0 ):
		"""
		mm = max( 1.0*np.amax(binCt), 1.0*np.amax(M) )
		
		tmScore = 0.0
		for i in range(nBin):
			for j in range(nBin):
				tmScore += ( T[i,j] - M[i,j] )**2
			# end
		# end
		tmScore /= 1.0*nBin**2
		tmScore /= 1.0*mm**2
		tmScore = ( 1 - tmScore**0.5 )**2
		"""
		
		X = binCt/np.amax(binCt)
		Y = binCm/np.amax(binCm)
		
		X = np.log(1+X)
		Y = np.log(1+Y)
		
		tmScore = 0.0
		for i in range(nBin):
			for j in range(nBin):
				tmScore += ( X[i,j] - Y[i,j] )**2
			# end
		# end
		
#		s = 4
		s = 3
		tmScore = np.exp( - tmScore/(2*s**2) )
	elif( scr == 1 ):
		h = 0
		
		tmScore = 0
		for i in range(nBin):
			for j in range(nBin):
#				print T[i,j], M[i,j]
				tmScore += Delta( Step( T[i,j], h ) - Step( M[i,j], h ) )
			# end
		# end
		tmScore /= 1.0*nBin**2
	elif( scr == 2 ):
		h = 0
		
		x = 0
		y = 0
		z = 0
		for i in range(nBin):
			for j in range(nBin):
				if( T[i,j] > h ):
					x += 1
				# end
				if( M[i,j] > h ):
					y += 1
				# end
				if( T[i,j] > h and M[i,j] > h ):
					z += 1
				# end
			# end
		# end
		if( z > 0 ):
			tmScore = (z / (x + y - z*1.0) )
		else:
			tmScore = 0.0
		# end
		
#		tmScore = tmScore**2.0
	elif( scr == 3 ):
		T  = T.flatten()
		M  = M.flatten()
		
		tmScore = np.corrcoef( T, M )[0,1]
		if( tmScore < 0 ):
			tmScore = 0.0
		# end
	elif( scr == 4 ):
		h = 0
		
		T  = T.flatten()
		M  = M.flatten()
		
		T[T>h] = 1
		M[M>h] = 1
		
		tmScore = np.corrcoef( T, M )[0,1]
		if( tmScore < 0 ):
			tmScore = 0.0
		# end
	elif( scr == 5 ):
		h = 0
		T[T>h] = 1
		M[M>h] = 1
		
		# target moments
		MT = np.zeros(10)
		moments = cv2.moments(T)
		MT[0] = moments['m00']
		c = moments['m10']
		d = moments['m01']
		MT[1] = c/MT[0]
		MT[2] = d/MT[0]
#		MT[3:] = cv2.HuMoments(moments)[:,0]
		MT[3] = moments['mu20']
		MT[4] = moments['mu11']
		MT[5] = moments['mu02']
		MT[6] = moments['mu30']
		MT[7] = moments['mu21']
		MT[8] = moments['mu12']
		MT[9] = moments['mu03']
		
		# model moments
		MM = np.zeros(10)
		moments = cv2.moments(M)
		MM[0] = moments['m00']
		c = moments['m10']
		d = moments['m01']
		MM[1] = c/MM[0]
		MM[2] = d/MM[0]
#		MM[3:] = cv2.HuMoments(moments)[:,0]
		MM[3] = moments['mu20']
		MM[4] = moments['mu11']
		MM[5] = moments['mu02']
		MM[6] = moments['mu30']
		MM[7] = moments['mu21']
		MM[8] = moments['mu12']
		MM[9] = moments['mu03']
		
		tmScore = ( (MM[1] - MT[1])/(MT[1]))**2 + ( (MM[2] - MT[2])/(MT[2]))**2
		
		tmScore = np.exp(-tmScore**1/0.1**2)
	elif( scr == 6 ):
		h = 0
		T[T>h] = 1
		M[M>h] = 1
		
		# target moments
		MT = np.zeros(10)
		moments = cv2.moments(T)
		MT[0] = moments['m00']
		c = moments['m10']
		d = moments['m01']
		MT[1] = c/MT[0]
		MT[2] = d/MT[0]
#		MT[3:] = cv2.HuMoments(moments)[:,0]
		MT[3] = moments['mu20']
		MT[4] = moments['mu11']
		MT[5] = moments['mu02']
		MT[6] = moments['mu30']
		MT[7] = moments['mu21']
		MT[8] = moments['mu12']
		MT[9] = moments['mu03']
		
		# model moments
		MM = np.zeros(10)
		moments = cv2.moments(M)
		MM[0] = moments['m00']
		c = moments['m10']
		d = moments['m01']
		MM[1] = c/MM[0]
		MM[2] = d/MM[0]
#		MM[3:] = cv2.HuMoments(moments)[:,0]
		MM[3] = moments['mu20']
		MM[4] = moments['mu11']
		MM[5] = moments['mu02']
		MM[6] = moments['mu30']
		MM[7] = moments['mu21']
		MM[8] = moments['mu12']
		MM[9] = moments['mu03']
		
#		tmScore = ( (MM[3] - MT[3])/(MT[3]))**2 + 2*( (MM[4] - MT[4])/(MT[4]))**2 + ((MM[5] - MT[5])/(MT[5]))**2
		
		MatM = np.array( [[ MM[3], MM[4]], [MM[4], MM[5]]] )
		MatT = np.array( [[ MT[3], MT[4]], [MT[4], MT[5]]] )
		
		tmScore = ( (LA.det(MatM) - LA.det(MatT))/LA.det(MatT) )**2
		
		tmScore = np.exp(-tmScore**1/0.4**2)
	elif( scr == 7 ):
		h = 0
		T[T>h] = 1
		M[M>h] = 1
		
		# target moments
		MT = np.zeros(10)
		moments = cv2.moments(T)
		MT[0] = moments['m00']
		c = moments['m10']
		d = moments['m01']
		MT[1] = c/MT[0]
		MT[2] = d/MT[0]
#		MT[3:] = cv2.HuMoments(moments)[:,0]
		MT[3] = moments['mu20']
		MT[4] = moments['mu11']
		MT[5] = moments['mu02']
		MT[6] = moments['mu30']
		MT[7] = moments['mu21']
		MT[8] = moments['mu12']
		MT[9] = moments['mu03']
		
		# model moments
		MM = np.zeros(10)
		moments = cv2.moments(M)
		MM[0] = moments['m00']
		c = moments['m10']
		d = moments['m01']
		MM[1] = c/MM[0]
		MM[2] = d/MM[0]
#		MM[3:] = cv2.HuMoments(moments)[:,0]
		MM[3] = moments['mu20']
		MM[4] = moments['mu11']
		MM[5] = moments['mu02']
		MM[6] = moments['mu30']
		MM[7] = moments['mu21']
		MM[8] = moments['mu12']
		MM[9] = moments['mu03']
		
		tmScore = ( (MM[6] - MT[6])/(MT[6]))**2 + 3*( (MM[7] - MT[7])/(MT[7]))**2 + 3*((MM[8] - MT[8])/(MT[8]))**2 + ((MM[9] - MT[9])/(MT[9]))**2
		
		tmScore = np.exp(-tmScore**1/20**2)
	elif( scr == 8 ):
		T = T.flatten()
		M = M.flatten()
		
		tmScore = np.corrcoef( np.log(1+T), np.log(1+M) )[0,1]
		
		if( tmScore < 0 ):
			tmScore = 0.0
		# end
	# end
	
#	tmScore *= tmScore
	
	return tmScore
	
# end

def MachineScoreW( nBin, binCt, binCm, binCu, scr ):
	
	T = deepcopy(binCt)
	M = deepcopy(binCm)
	U = deepcopy(binCu)
	
	T = np.log( 1+T.flatten() )
	M = np.log( 1+M.flatten() )
	U = np.log( 1+U.flatten() )
	
	tm = np.abs(T-M)
	mu = np.abs(M-U)
	tu = np.abs(T-U)
	weights = ( tm + mu + tu ) + np.ones(len(tu))*np.mean(tm+mu+tu)
	
#	a = np.corrcoef( np.log(1+T), np.log(1+M) )[0,1]
	b = corrW( T, M, weights )
	
	return b
# end

def covW( x, y, w ):
	n = len(x)
	return np.sum( w * (x - np.average(x,weights=w)) * (y - np.average(y,weights=w)) )/np.sum(w)*n/(n-1)
# end

def corrW( x, y, w ):
	return covW(x,y,w)/( covW(x,x,w)*covW(y,y,w) )**0.5
# end

def ReadAndCleanupData( filePath ):
	
	print "Cleaning target file..."
	
	# read data into np array
	df = pd.read_csv( filePath, sep=',|\t', engine='python', header=None )
	data1 = df.values
	
	# remove unranked models
	ind = 0
	while( not math.isnan(data1[ind,-1]) ):
		ind += 1
	# end
	data2 = data1[0:ind,:]
	nModel = ind + 1
	
	# include human score and SPAM params
	cols = range(4,18) + [ 1 ]
	data2 = data2[:,cols]
	
	# ignore bad zoo models
#	data2 = data2[data2[:,-1]>=thresh,:]
	nModel = data2.shape[0]
	
	data2 = np.array( data2, dtype=np.float32 )
	

	print "Zoo file:"
	print data2[0,:]
	
#	data2[0,2]  = 10
#	data2[0,5]  = -10
#	data2[0,10] = 20
#	data2[0,11] = 20
#	data2[0,12] = 100
#	data2[0,13] = 100
	
	data3 = deepcopy(data2)
	
	psi = []
	for i in range(nModel):
		psi_p = 1
		psi_s = 1
		
		if( data2[i,2] < 0 ):
			data2[i,2]  =  -1 * data2[i,2]
			data2[i,5]  =  -1 * data2[i,5]
			data2[i,10] = 180 + data2[i,10]
			data2[i,11] = 180 + data2[i,11]
		# end
		data2[i,10] %= 360
		data2[i,11] %= 360
		if( data2[i,10] > 180 ):
			data2[i,10] = data2[i,10] - 180
			data2[i,12] = -1 * data2[i,12]
		# end
		if( data2[i,11] > 180 ):
			data2[i,11] = data2[i,11] - 180
			data2[i,13] = -1 * data2[i,13]
		# end
		data2[i,12] %= 360
		data2[i,13] %= 360
		if( data2[i,12] > 180 ):
			data2[i,12] = data2[i,12] - 360
		# end
		if( data2[i,13] > 180 ):
			data2[i,13] = data2[i,13] - 360
		# end
		
		if( data2[i,12] > 90 ):
			data2[i,12] = data2[i,12] - 180
			psi_p = -1
		elif( data2[i,12] < -90 ):
			data2[i,12] = data2[i,12] + 180
			psi_p = -1
		# end
		if( data2[i,13] > 90 ):
			data2[i,13] = data2[i,13] - 180
			psi_s = -1
		elif( data2[i,13] < -90 ):
			data2[i,13] = data2[i,13] + 180
			psi_s = -1
		# end
		psi.append( [psi_p,psi_s] )
	# end
	psi = np.array(psi)
	
	# energy
	G = 1
	r = ( data2[:,0]**2 + data2[:,1]**2 + data2[:,2]**2 )**0.5
	U = -G*data2[:,6]*data2[:,7]/r
	v = ( data2[:,3]**2 + data2[:,4]**2 + data2[:,5]**2 )**0.5
	K = 0.5*data2[:,7]*v**2
#	c = np.log(1-K/U)
	c = (K+U)/(K-U)
	
	# convert p,s mass to fraction,total mass
	t = data2[:,6] + data2[:,7]
	f = data2[:,6] / t
	data2[:,6] = f
	data2[:,7] = t
	
	# spherical velocity
	phi   = ( np.arctan2( data2[:,4], data2[:,3] ) * 180.0 / np.pi ) % 360
	theta = ( np.arcsin( data2[:,5] / v ) * 180.0 / np.pi )
	
#	data2[:,2] = c
#	data2[:,3] = v
	data2[:,3] = c
#	data2[:,2] = K
#	data2[:,3] = U
	
	data2[:,4] = phi
	data2[:,5] = theta
	
#	"""
	Ap = np.abs(data2[:,8]**2*np.cos(data2[:,12]*np.pi/180.0))
	As = np.abs(data2[:,9]**2*np.cos(data2[:,13]*np.pi/180.0))
	
	data2[:,8] = Ap
	data2[:,9] = As
#	"""
	
	return data2, psi, nModel, len(cols)
	
# end

def solve( param, nParam, nBin, bound, fileInd, psi ):
	
	p = deepcopy(param)
	
	# add for psi
	if( psi[0] == -1 ):
		p[12] += 180.0
	# end
	if( psi[1] == -1 ):
		p[13] += 180.0
	# end
	
	# convert mass units
	f    = p[6]
	t    = p[7]
	p[6] = f*p[7]
	p[7] = (1.0-f)*p[7]
	
	G = 1
	c = p[3]
	# K/U
#	v = ( 2*G*p[6]*(np.exp(c)-1)/(p[0]**2+p[1]**2+p[2]**2)**0.5 )**0.5
	# K+U/K-U
	v = ( (1+c)/(1-c)*2*G*p[6]/(p[0]**2+p[1]**2+p[2]**2)**0.5 )**0.5
	
	"""
	c = p[2]
	v = p[3]
	z = ( ( 2*G*p[6]*(np.exp(c)-1)/p[3]**2 )**2 - p[0]**2 - p[1]**2 )**0.5
	if( np.isreal(z) ):
		p[2] = z
	else:
		p[2] = 0
	# end
	"""
	
	vx = v*math.cos(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vy = v*math.sin(p[4]*np.pi/180.0)*math.cos(p[5]*np.pi/180.0)
	vz = v*                           math.sin(p[5]*np.pi/180.0)
	p[3] = vx
	p[4] = vy
	p[5] = vz
	
	Ap = p[8]
	As = p[9]
	p[8] = np.abs(Ap/np.cos(p[12]*np.pi/180.0))**0.5
	p[9] = np.abs(As/np.cos(p[13]*np.pi/180.0))**0.5
	
#	p[2] = 1000
	
	print "SPAM"
	print p
	
#	print p
	paramStr = ','.join( map(str, p[0:nParam]) ) + ',0'
#	print paramStr
	
	# with flag
	call("./basic_run_unpreturbed -o " + fileInd + " " + paramStr + " > SolveMetro.out", shell=True)
	
	# with flag
	RV   = np.loadtxt("basic_"     + fileInd + ".out")
	RV_u = np.loadtxt("basic_unp_" + fileInd + ".out")
#	print RV_u[0,:]
	
	dr = RV[-1,0:3]-RV_u[-1,0:3]
	for i in range(len(RV)/2+1):
		j = i + len(RV)/2
		RV_u[j,0:3] = RV_u[j,0:3] + dr
		RV_u[j,3:] = 0
	# end
	
	binC,   binV,   binD   = BinField( nBin, RV,   bound )
	binC_u, binV_u, binD_u = BinField( nBin, RV_u, bound )
	
	return binC, binV, binD, binC_u, RV, RV_u
	
# end

def BinField( nBin, RV, bound ):
	
	nPts = np.size(RV,0)-1
	
	binC = np.zeros((nBin,nBin))
	binV = np.zeros((nBin,nBin))
	binD = np.zeros((nBin,nBin))
	
	xmin = bound[0,0]
	xmax = bound[0,1]
	ymin = bound[1,0]
	ymax = bound[1,1]
	
	dx = (xmax-xmin)/nBin
	dy = (ymax-ymin)/nBin
	
	inds = []
	for i in range(nBin):
		inds.append([])
		for j in range(nBin):
			inds[i].append([])
		# end
	# end
	
	for i in range(nPts):
		x  = float(RV[i,0])
		y  = float(RV[i,1])
		vz = float(RV[i,5])
		
		ii = int( (x - xmin) / (xmax - xmin) * nBin )
		jj = int( (y - ymin) / (ymax - ymin) * nBin )
		
		if( ii >= 0 and ii < nBin and jj >= 0 and jj < nBin ):
		        binC[jj,ii] = binC[jj,ii] + 1
			inds[jj][ii].append(vz)
		# end
	# end
	
	for i in range(nBin):
		for j in range(nBin):
			if( len(inds[j][i]) == 1 ):
				binV[j,i] = np.mean( np.array(inds[j][i]) )
			# end
			if( len(inds[j][i]) >= 2 ):
				binV[j,i] = np.mean( np.array(inds[j][i]) )
				binD[j,i] = np.std(  np.array(inds[j][i]) )
			# end
		# end
	# end
	
	return binC, binV, binD
	
# end

def BinField_Vel( nBin, RV, bound, alpha ):

	nPts = np.size(RV,0)-1
	
	binC  = np.zeros((nBin,nBin,nBin))
	binVel  = np.zeros((nBin,nBin,nBin))
	
	binC2 = np.zeros((nBin,nBin))
	binVel2 = np.zeros((nBin,nBin))

	xmin = bound[0,0]
	xmax = bound[0,1]
	ymin = bound[1,0]
	ymax = bound[1,1]
	zmin = np.min( RV[:,2] )
	zmax = np.max( RV[:,2] )
	
	dx = (xmax-xmin)/nBin
	dy = (ymax-ymin)/nBin
	dz = (zmax-zmin)/nBin
	
	for i in range(nPts):
		x = float(RV[i,0])
		y = float(RV[i,1])
		z = float(RV[i,2])
		vz = float(RV[i,5])
		
		ii = int( (x - xmin) / (xmax - xmin) * nBin )
		jj = int( (y - ymin) / (ymax - ymin) * nBin )
		kk = int( (z - zmin) / (zmax - zmin) * nBin )
		
		if( ii >= 0 and ii < nBin and jj >= 0 and jj < nBin and kk >= 0 and kk < nBin ):
		        binC[jj,ii,kk]   = binC[jj,ii,kk] + 1
		        binVel[jj,ii,kk] = binVel[jj,ii,kk] + vz
		# end
	# end
	
	for i in range(nBin):
		for j in range(nBin):
			for k in range(nBin):
				if( binC[i,j,k] > 1 ):
					binVel[i,j,k] = binVel[i,j,k]/binC[i,j,k]
				# end
			# end
		# end
	# end
	
	# get extinction weights
	w = np.zeros((nBin,nBin,nBin))
	for i in range(nBin):
		for j in range(nBin):
			for k in range(nBin):
				if( binC[i,j,k] > 0 ):
					w[i,j,k] = np.exp(-alpha * np.sum(binC[i,j,nBin-k-1:nBin])*dz/(dx*dy*dz)/(nPts) )
				# end
			# end
		# end
	# end

	for i in range(nBin):
		for j in range(nBin):
			sumV = 0
			sumW = 0
			
			for k in range(nBin):
				if( binC[i,j,k] > 0 ):
					w = np.exp(-alpha * np.sum(binC[i,j,nBin-k-1:nBin])*dz/(dx*dy*dz)/(nPts) )
					
					sumV += binC[i,j,k]*w*binVel[i,j,k]
					sumW += binC[i,j,k]*w
				# end
			# end
			if( np.sum(binC[i,j]) > 0 ):
				binVel2[i,j] = sumV/sumW
			else:
				binVel2[i,j] = 0
			# end
			binC2[i,j] = np.sum(binC[i,j,:])
		# end
	# end
	
	return binC2, binVel2
	
# end

def getInitPop( nPop, nParam, xLim ):
	
	# get initial population
	popSol = []
	for i in range(nPop):
		x = np.zeros(nParam)
		for j in range(nParam):
			x[j] = np.random.uniform( xLim[j,0], xLim[j,1] )
		# end
		popSol.append( x )
	# end
	popSol = np.array(popSol)
	
	return popSol
# end

def evalPop( nPop, popSol, nParam, nBin,  bound, T, scrTM, scrMU, a, b ):
	
	popFit = []
	for i in range(nPop):
#		print popSol[i]
		M, U = solve( popSol[i,:], nParam, nBin, bound )
#		M = np.ones((nBin,nBin))
#		U = np.ones((nBin,nBin))
		
		tmScore = MachineScore( nBin, T, M, scrTM )
		muScore = MachineScore( nBin, M, U, scrMU )
		
		muScore2 = np.exp( -(muScore - a)**2/(2*b**2))
		
		popFit.append( tmScore*muScore2 )
	# end
	popFit = np.array(popFit)
#	print " "
	
	return popFit
# end

def Selection( nPop, popSol, popFit ):
	
	selectType = 0
	
	parents = []
	
	if( selectType == 0 ):
		# get selection probabilities
		popProb = np.cumsum( popFit/np.sum(popFit) )
		
		for i in range(nPop/2):
			r1 = np.random.uniform(0,1)
			r2 = np.random.uniform(0,1)
			
			ind1 = np.argmax( r1 <= popProb )
			ind2 = np.argmax( r2 <= popProb )
			
			parents.append( [ popSol[ind1], popSol[ind2] ] )
		# end
	else:
		w = 1
	# end
	parents = np.array(parents)
	
	return parents
# end

def Crossover( nPop, nParam, parents ):
	
	crossType = 0
	
	popSol = np.zeros((nPop,nParam))
	
	if( crossType == 0 ):
		for i in range(nPop/2):
			r  = np.random.uniform(0,1)
			c1 = parents[i,0]*r     + parents[i,1]*(1-r)
			c2 = parents[i,0]*(1-r) + parents[i,1]*r
			popSol[2*i,:]   = c1
			popSol[2*i+1,:] = c2
		# end
	else:
		w = 1
	# end
	popSol = np.array(popSol)
	
	return popSol
# end

def Mutate( nPop, nParam, popSol, cov, xLim ):
	
	popSol2 = np.zeros((nPop,nParam))
	for i in range(nPop):
		popSol2[i,:] = np.random.multivariate_normal(mean=popSol[i,:],cov=cov,size=1)[0]
		
		for j in range(nParam):
			if(   xLim[j,0] > popSol2[i,j] ):
				popSol2[i,j] = xLim[j,0] + np.abs(xLim[j,0] - popSol2[i,j])
			elif( xLim[j,1] < popSol2[i,j] ):
				popSol2[i,j] = xLim[j,1] - np.abs(xLim[j,1] - popSol2[i,j])
			# end
		# end
	# end
	
	return popSol2
# end

def getCovMatrix( cov, covInit, C, mean, beta, step, burn, nPop, nParam, chainAll, popSol, r, scaleInit, toMix, mixProb, mixAmp ):
	
	chainF = np.array(chainAll)
	
	# get AM cov matrix
	if( step < burn ):
		C = deepcopy(covInit)
	elif( step == burn ):
		mean = np.mean( np.array(chainAll)[nPop*burn/2:,:], axis=0 )
		C    = np.cov( np.transpose( np.array(chainAll)[nPop*burn/2:,:] ) )
#		mean = np.mean( np.array(chainAll)[:,:], axis=0 )
#		C    = np.cov( np.transpose( np.array(chainAll)[:,:] ) )
		
#		r    = np.prod( pWidth**2 )**(0.5/nParam)
		cov  = C + beta*covInit
	elif( step > burn ):
		for i in range(nPop):
			gamma = 1.0/(nPop*step+1+i)
			dx   = popSol[i] - mean
			mean = mean + gamma*dx
			C    = C    + gamma*(np.outer(dx,dx) - C)
		# end
		
#		r    = 1.0
		cov  = C + beta*covInit
#		r    = r*np.exp( gamma*(accProb - P) )
	# end
	
	# apply mixing
	if( toMix == 1 ):
		if( step <= burn ):
			# scale
			for i in range(nParam):
				s = np.random.uniform(0,1)
				
				# thinning
				if( s <= mixProb[0] ):
					cov[i,i] = (r*mixAmp[0])**2*covInit[i,i]
				# widening
				elif( s <= mixProb[0] + mixProb[1] ):
					cov[i,i] = (r*mixAmp[1])**2*covInit[i,i]
				# fixing
				else:
					cov[i,i] = (r)**2*covInit[i,i]
				# end
			# end
		elif( step > burn ):
			# decompose, normalize
			w, v = LA.eig(cov)
			w    = scaleInit*w/np.abs(np.prod(w))**(1.0/nParam)
			
			# mix
			for i in range(nParam):
				s = np.random.uniform(0,1)
				
				# thinning
				if( s <= mixProb[0] ):
					w[i] = (r*mixAmp[0])**2*w[i]
				# widening
				elif( s <= mixProb[0] + mixProb[1] ):
					w[i] = (r*mixAmp[1])**2*w[i]
				# fixing
				else:
					w[i] = (r)**2*w[i]
				# end
			# end
			
			# recompose matrix
			W   = np.diag(w)
#			cov = np.dot( np.dot(v,W), LA.inv(v) )
			cov = np.dot( np.dot(v,W), np.transpose(v) )
			cov = cov.real
			cov = 0.5*( cov + np.transpose(cov) )
		# end
	# end
	
	return cov, mean, C
# end

def GA( nGen, nPop, nParam, start, xLim, pWidth, nBin, bound, T, toMix, burn, beta, mixAmp, mixProb, scrTM, scrMU, a, b ):
	
	covInit = np.diag(pWidth**2)
	cov     = np.diag(pWidth**2)
	C       = np.diag(pWidth**2)
	mean    = deepcopy(start)
	
	scaleInit = np.prod( pWidth**2 )**(1.0/nParam)
	r = 1.0
	
	# all generations
	chain  = []
	chainF = []
	
	# get initial population
	popSol = getInitPop( nPop, nParam, xLim )
	popFit = evalPop( nPop, popSol, nParam, nBin, bound, T, scrTM, scrMU, a, b )
	
	# add init to all
	chain = [ popSol ]
	error = [ popFit ]
	for i in range(nPop):
		chainF.append( popSol[i] )
	# end
	
#	print "min error gen " + str(0) + ": "
#	print min(popFit)
	
	# Random walk
	for step in range(nGen):
		print str(step+1) + "/" + str(nGen)
		
		for i in range(nPop):
#			print popSol[i]
			print popFit[i]
		# end
		
		# get covariance matrix
		cov, mean, C = getCovMatrix( cov, covInit, C, mean, beta, step, burn, nPop, nParam, chainF, popSol, r, scaleInit, toMix, mixProb, mixAmp )
		
		# perform selection
		parents = Selection( nPop, popSol, popFit )
		
		# perform crossover
		popSol = Crossover( nPop, nParam, parents )
		
		# perform mutation
		popSol = Mutate( nPop, nParam, popSol, cov, xLim )
		
		# calculate errors
		popFit = evalPop( nPop, popSol, nParam, nBin, bound, T, scrTM, scrMU, a, b )
		
		chain.append( popSol )
		error.append( popFit )
		for i in range(nPop):
			chainF.append( popSol[i] )
		# end
		
#		print "min error gen " + str(step+1) + ": "
#		print min(popFit)
		
		print " "
	# end
	
	chain  = np.array(chain)
	error  = np.array(error)
	
	return chain, error

# end

"""
# perform mutation
if( step <= n/3 ):
	for i in range(nPop):
		popSol[i] = Mutate( popSol[i], cov )
	# end
else:
	for i in range(nPop):
		popSol[i] = Mutate( popSol[i], cov/(1.0+np.log(step-n/3))**1.2 )
	# end
# end
"""

def cumMin( chain1, error1 ):
	
	error2 = []
	chain2 = []
	
	curMin = error1[0] + 1
	for i in range( len(error1) ):
		if( error1[i] < curMin ):
			error2.append( error1[i]   )
			chain2.append( chain1[i,:] )
			curMin = error1[i]
		else:
			error2.append( error2[-1]   )
			chain2.append( chain2[-1] )
		# end
	# end
	error2 = np.array(error2)
	chain2 = np.array(chain2)
	
	return chain2, error2
# end







def equals(a, b):
	
	n = len(a)
	
	test = 1
	for i in range(n):
		if( a[i] != b[i] ):
			test = test*0
	
	return test
	
# end

def print2D(A):
	
#	print('\n'.join([' '.join(['{:4}'.format(item) for item in row]) for row in A]))
#	print('\n'.join([' '.join(['{0:.2f}'.format(item) for item in row]) for row in A]))
	print('\n'.join([' '.join(['{0:8.4f}'.format(item) for item in row]) for row in A]))
	
# end






main()






