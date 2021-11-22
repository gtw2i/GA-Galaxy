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
import cv2
import multiprocessing as mp
import threading as thr
from threading import Thread
import os
import time
import Queue

import socket as skt


##############
#    MAIN    #
##############

def main():
	
	#####################
	###   VARIABLES   ###
	#####################
	
	zooFiles = []
	zooFiles.append( "587736941981466667" )
	zooFiles.append( "587726033843585146" )
	zooFiles.append( "587727177926508595" )
	"""
	zooFiles.append( "587739647284805725" )
	zooFiles.append( "587727178988388373" )
	zooFiles.append( "587722984435351614" )
	zooFiles.append( "587729227151704160" )
	zooFiles.append( "587735043609329845" )
	zooFiles.append( "587741391565422775" )
	zooFiles.append( "587745402001817662" )
	zooFiles.append( "588017604696408086" )
	zooFiles.append(        "hst_Arp_272" )
	zooFiles.append( "587733080814583863" )
	zooFiles.append( "587738569246376675" )
	zooFiles.append( "587738569249390718" )
	zooFiles.append( "587739153356095531" )
	zooFiles.append( "587739721900163101" )
	zooFiles.append( "587742014353702970" )
	zooFiles.append( "587746029596311590" )
	zooFiles.append( "588011124116422756" )
	zooFiles.append( "758877153600208945" )
	"""
	
#	zooBase = "587739647284805725"
#	zooBase = "587726033843585146"
#	zooBase = "587727178988388373"
	
	zooFiles.append( "587726033843585146" )
	
	ext = "_combined.txt"
	
	nProc = 2**3
	nParam = 14
	nTarg = 2**0
	nFile = len(zooFiles)
#	nFile = 1
	
	extra = 0.2
		
	hostname = skt.gethostname()
	
	# -----------
	
	nPop = nTarg
	bound = []
	T = []
	nBin = 0
	a = 0
	b = 0
	scrTM = 0
	scrMU = 0
	
	# -----------
	
	###############################
	###   DATA/INITIALIZATION   ###
	###############################
	
	# read zoo file
	data = []
	xLim = np.zeros((nFile,nParam,2))
	for i in range(nFile):
		xxx, nModel, nCol = ReadAndCleanupData( zooFiles[i] + ext )
		data.append( xxx[:nTarg,:nParam] )
		xLim[i,:,0] = np.min( data[i], axis=0 )
		xLim[i,:,1] = np.max( data[i], axis=0 )
		
#		xLim[i,10,:] = np.array([ 0.0, 360.0 ])
#		xLim[i,11,:] = np.array([ 0.0, 360.0 ])
#		xLim[i,12,:] = np.array([ 0.0, 360.0 ])
#		xLim[i,13,:] = np.array([ 0.0, 360.0 ])
	# end
	
	"""
	mmm[2,:]  = np.array([ -10.0, 10.0 ])
	mmm[3,:]  = np.array([ -5.0, 5.0 ])
	mmm[4,:]  = np.array([ -5.0, 5.0 ])
	mmm[5,:]  = np.array([ -10.0, 10.0 ])
	mmm[6,:]  = np.array([ 0.25, 0.75 ])
	mmm[7,:]  = np.array([ 20.0, 70.0 ])
	mmm[8,:]  = np.array([ 2.0, 6.0 ])
	mmm[9,:]  = np.array([ 3.0, 7.0 ])
	"""
	
	bound = np.zeros((nFile,nTarg,2,2))
	for i in range(nFile):
		print "File: " + str(i+1) + "/" + str(nFile)
		RVs, RV_us = evalPop( nProc, nPop, data[i], nParam, nBin, bound, T, scrTM, scrMU, a, b, hostname )
		
		for j in range(nTarg):
			X = RVs[j]
			bound[i,j,0,0] = np.min( X[:,0] )
			bound[i,j,0,1] = np.max( X[:,0] )
			bound[i,j,1,0] = np.min( X[:,1] )
			bound[i,j,1,1] = np.max( X[:,1] )
		# end
		print " "
	# end
	
	for i in range(nFile):
		print zooFiles[i]
		for j in range(nTarg):
			print "Target: " + str(j)
			print bound[i,j,:,:]
		# end
		print " "
	# end
	
	bound2 = np.zeros((nFile,2,2))
	print "max bounds:"
	for i in range(nFile):
		print zooFiles[i]
		bound2[i,0,0] = np.min( bound[i,:,0,0] )
		bound2[i,0,1] = np.max( bound[i,:,0,1] )
		bound2[i,1,0] = np.min( bound[i,:,1,0] )
		bound2[i,1,1] = np.max( bound[i,:,1,1] )
		print bound2[i,:,:]
		print " "
	# end
	
	bound3 = np.zeros((nFile,2,2))
	print "equal bounds:"
	for i in range(nFile):
		print zooFiles[i]
		xw = bound2[i,0,1] - bound2[i,0,0]
		yw = bound2[i,1,1] - bound2[i,1,0]
		
		p = max( xw, yw )
		q = np.argmax( np.array([xw, yw]) )
		
		if( q == 0 ):
			diff = xw - yw
			bound3[i,1,0] = bound2[i,1,0] - diff/2
			bound3[i,1,1] = bound2[i,1,1] + diff/2
			bound3[i,0,0] = bound2[i,0,0]
			bound3[i,0,1] = bound2[i,0,1]
		elif( q == 1 ):
			diff = yw - xw
			bound3[i,0,0] = bound2[i,0,0] - diff/2
			bound3[i,0,1] = bound2[i,0,1] + diff/2
			bound3[i,1,0] = bound2[i,1,0]
			bound3[i,1,1] = bound2[i,1,1]
		# end
		
		bound3[i,0,0] = bound2[i,0,0] - extra*p
		bound3[i,0,1] = bound2[i,0,1] + extra*p
		bound3[i,1,0] = bound2[i,1,0] - extra*p
		bound3[i,1,1] = bound2[i,1,1] + extra*p
		
		print bound3[i,:,:]
		print " "
	# end
	
	outBase = "Bounds_"
	X = np.zeros((nParam+2,2))
	for i in range(nFile):
		filename = outBase + zooFiles[i] + ".txt"
		
		X[0:nParam,:] = xLim[i,:,:]
		X[nParam:,:]  = bound3[i,:,:]
		
		np.savetxt( filename, X )
	# end
	
	
	
##############
#  END MAIN  #
##############

def ReadAndCleanupData( filePath ):
	
#	print "Cleaning target file..."
	
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
	
	# convert p,s mass to ratio,total mass
	t = data2[:,6] + data2[:,7]
	f = data2[:,6] / t
	data2[:,6] = f
	data2[:,7] = t
	
	data2 = np.array( data2, dtype=np.float32 )
	
	return data2, nModel, len(cols)
	
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
	elif( scr == 4 ):
		h = 0
		
		T  = T.flatten()
		M  = M.flatten()
		
		T[T>h] = 1
		M[M>h] = 1
		
		tmScore = np.corrcoef( T, M )[0,1]
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
	# end
	
	return tmScore
	
# end

def solve( param, nParam, nBin, bound, fileInd ):
	
	p = deepcopy(param)
	
	# convert mass units
	f    = p[6]
	t    = p[7]
	p[6] = f*p[7]
	p[7] = (1.0-f)*p[7]
	
#	p[2] = 1000
	
	paramStr = ','.join( map(str, p[0:nParam]) )
	
	# old idk	
#	call("./basic_run " + paramStr + "", shell=True)
	# no flag
#	call("./basic_run_unpreturbed " + paramStr + " > SolveMetro.out", shell=True)
	# with flag
	call("./basic_run_unpreturbed -o " + fileInd + " " + paramStr + " > SolveMetro.out", shell=True)
	
	# no flag
#	RV   = np.loadtxt("a.101")
#	RV_u = np.loadtxt("a.000")
	
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
	
	binC   = BinField( nBin, RV,   bound )
	binC_u = BinField( nBin, RV_u, bound )
	
	return binC, binC_u, RV, RV_u
	
# end

def solve_parallel( XXX, nParam, nBin, bound, scrTM, scrMU, a, b, fileInd, qOut ):
	
	index = XXX[0]
	param = XXX[1]
	
	p = deepcopy(param)
	
	# convert mass units
	f    = p[6]
	t    = p[7]
	p[6] = f*p[7]
	p[7] = (1.0-f)*p[7]
	
#	p[2] = 1000
	
	paramStr = ','.join( map(str, p[0:nParam]) )
	
	# old idk	
#	call("./basic_run " + paramStr + "", shell=True)
	# no flag
#	call("./basic_run_unpreturbed " + paramStr + " > SolveMetro.out", shell=True)
	# with flag
	call("./basic_run_unpreturbed -o " + fileInd + " " + paramStr + " > SolveMetro.out", shell=True)
	
	# no flag
#	RV   = np.loadtxt("a.101")
#	RV_u = np.loadtxt("a.000")
	
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
	
#	M = BinField( nBin, RV,   bound )
#	U = BinField( nBin, RV_u, bound )
	
#	tmScore  = MachineScore( nBin, T, M, scrTM )
#	muScore  = MachineScore( nBin, M, U, scrMU )
#	muScoreX = np.exp( -(muScore - a)**2/(2*b**2))
#	score    = (tmScore*muScoreX)**(1.0/2.0)
	
#	qOut.put( [index, M, U] )
	qOut.put( [index, RV, RV_u] )
	
	return RV, RV_u
#	return score
# end

def BinField( nBin, RV, bound ):
	
	nPts = np.size(RV,0)-1
#	print nPts
	
	binC  = np.zeros((nBin,nBin))
	
	xmin = bound[0,0]
	xmax = bound[0,1]
	ymin = bound[1,0]
	ymax = bound[1,1]
	
	dx = (xmax-xmin)/nBin
	dy = (ymax-ymin)/nBin
	
	for i in range(nPts):
		x = float(RV[i,0])
		y = float(RV[i,1])
		
		ii = int( (x - xmin) / (xmax - xmin) * nBin )
		jj = int( (y - ymin) / (ymax - ymin) * nBin )
		
		if( ii >= 0 and ii < nBin and jj >= 0 and jj < nBin ):
		        binC[jj,ii] = binC[jj,ii] + 1
		# end
	# end
	
	return binC
	
# end

def getInitPop( nPop, nParam, pFit, pReal, xLim ):
	
	initType = 0
	
	popSol = []
	
	if( initType == 0 ):
		for i in range(nPop):
			x = np.zeros(nParam)
			for j in range(nParam):
				if j in pFit:
					x[j] = np.random.uniform( xLim[j,0], xLim[j,1] )
				else:
					x[j] = pReal[j]
				# end
			# end
			popSol.append( x )
		# end
	else:
		R = []
		for j in range(nParam):
			R.append( np.linspace( xLim[j,0], xLim[j,1], nPop ) )
		# end
		R = np.array(R)
		
		for i in range(nPop):
			x = np.zeros(nParam)
			for j in range(nParam):
				if j in pFit:
					x[j] = R[j][i]
				else:
					x[j] = pReal[j]
				# end
			# end
			popSol.append( x )
		# end
	# end
	
	popSol = np.array(popSol)
	
	return popSol
# end

def evalPop( nProc, nPop, popSol, nParam, nBin, bound, T, scrTM, scrMU, a, b, hostname ):
	
	qJob = TaskQueue(num_workers=nProc)
	qOut = Queue.Queue()
	
	for j in range(nPop):
		if(   j < 10 ):
			fileInd = "00" + str(j)
		elif( j < 100 ):
			fileInd =  "0" + str(j)
		else:
			fileInd =        str(j)
		# end
		
		fileInd += "_" + hostname
#		print fileInd
		
		qJob.add_task( solve_parallel, [j, popSol[j,:]], nParam, nBin, bound, scrTM, scrMU, a, b, fileInd, qOut )
	# end
	
	qJob.join()
	
	out  = []
	inds = []
	for i in range(nPop):
		out.append( qOut.get() )
		inds.append( out[i][0] )
	# end
	inds = np.argsort(inds)
	
	RV = []
	RV_u = []
	for i in range(nPop):
		RV.append( out[inds[i]][1] )
		RV_u.append( out[inds[i]][2] )
	# end
	
	return RV, RV_u
# end

def Selection( nPop, popSol, popFit, nKeep ):
	
	# 0: fitness-squared proportional selection
	# 1: rank proportional selection
	selectType = 0
	
	parSol = []
	parFit = []
	
	if( selectType == 0 ):
		xxx = popFit**2
		xxx = xxx/np.sum(xxx)
		popProb = np.cumsum( xxx )
		
		for i in range(nPop-nKeep):
			r1 = np.random.uniform(0,1)
			r2 = np.random.uniform(0,1)
			
			ind1 = np.argmax( r1 <= popProb )
			ind2 = np.argmax( r2 <= popProb )
			
			parSol.append( [ popSol[ind1], popSol[ind2] ] )
			parFit.append( [ popFit[ind1], popFit[ind2] ] )
		# end
		
		srt = np.argsort( popFit )
		for i in range(1,nKeep+1):
			parSol.append( [ popSol[srt[-i]], popSol[srt[-i]] ] )
			parFit.append( [ popFit[srt[-i]], popFit[srt[-i]] ] )
		# end
	else:
		# get selection probabilities
		inds = popFit.argsort() + np.ones(nPop)
		popProb = np.cumsum( inds/np.sum(inds) )
		
		for i in range(nPop):
			r1 = np.random.uniform(0,1)
			r2 = np.random.uniform(0,1)
			
			ind1 = np.argmax( r1 <= popProb )
			ind2 = np.argmax( r2 <= popProb )
			
			parents.append( [ popSol[ind1], popSol[ind2] ] )
		# end

	# end
	parSol = np.array(parSol)
	parFit = np.array(parFit)
	
	return parSol, parFit
# end

def Crossover( nPop, nParam, parSol, parFit, cov, step, burn ):
	
	popSol = np.zeros((nPop,nParam))
	
	if( step < burn ):
		for i in range(nPop):
			for j in range(nParam):
				r0 = parFit[i,0]/(parFit[i,0]+parFit[i,1])
				r  = np.random.uniform(0,1)
				
				if( r < r0 ):
					popSol[i,j] = parSol[i,0,j]
				else:
					popSol[i,j] = parSol[i,1,j]
				# end
			# end
		# end
	else:
		# get PCA
		w, v = LA.eig(cov)
		
		# convert to PCA basis
		pcaPar = []
		for i in range(nPop):
			p1 = np.dot( v, parSol[i,0,:] )
			p2 = np.dot( v, parSol[i,0,:] )
			pcaPar.append( [p1, p2] )
		# end
		pcaPar = np.array(pcaPar)
		
		# mix in PCA basis
		for i in range(nPop):
			for j in range(nParam):
				r0 = parFit[i,0]/(parFit[i,0]+parFit[i,1])
				r  = np.random.uniform(0,1)
				
				if( r < r0 ):
					popSol[i,j] = pcaPar[i,0,j]
				else:
					popSol[i,j] = pcaPar[i,1,j]
				# end
			# end
		# end
		
		# convert back to parameter basis
		vinv = LA.inv(v)
		for i in range(nPop):
			popSol[i,:] = np.dot( vinv, popSol[i,:] )
		# end
	# end
	popSol = np.array(popSol)
	
	return popSol
# end

def Mutate( step, nGen, nPop, nParam, popSol, cov, xLim, pFit, toFlip, flips, flipProb):
	
	popSol2 = np.zeros((nPop,nParam))
	
	for i in range(nPop):
		x = np.random.multivariate_normal(mean=popSol[i,:],cov=cov,size=1)[0]
		
		if( step in flips and toFlip ):
			if( 2 in pFit and 5 in pFit ):
				r = np.random.uniform(0,1)
				if( r < 0.5 ):
					x[2] = -x[2]
					x[5] = -x[5]
				# end
			# end
			if( 10 in pFit ):
				r = np.random.uniform(0,1)
				if( r < 0.5 ):
					x[10] = 180 + x[10]
				# end
				x[10] = x[10] % 360
			# end
			if( 11 in pFit ):
				r = np.random.uniform(0,1)
				if( r < 0.5 ):
					x[11] = 180 + x[11]
				# end
				x[11] = x[11] % 360
			# end
			if( 12 in pFit ):
				r = np.random.uniform(0,1)
				if( r < 0.25 ):
					x[12] = 180 - x[12]
				elif( r < 0.50 ):
					x[12] = 180 + x[12]
				elif( r < 0.75 ):
					x[12] = 360 - x[12]
				# end
				x[12] = x[12] % 360
			# end
			if( 13 in pFit ):
				r = np.random.uniform(0,1)
				if( r < 0.25 ):
					x[13] = 180 - x[13]
				elif( r < 0.50 ):
					x[13] = 180 + x[13]
				elif( r < 0.75 ):
					x[13] = 360 - x[13]
				# end
				x[13] = x[13] % 360
			# end
		# end
		
		eps = 0.00001
		for j in range(nParam):
			if(   xLim[j,0] > x[j] ):
#				popSol2[i,j] = 0.5*( xLim[j,0] + popSol[i,j] )
				popSol2[i,j] = xLim[j,0] + eps*(xLim[j,1]-xLim[j,0])
			elif( xLim[j,1] < x[j] ):
#				popSol2[i,j] = 0.5*( xLim[j,1] + popSol[i,j] )
				popSol2[i,j] = xLim[j,1] - eps*(xLim[j,1]-xLim[j,0])
			else:
				popSol2[i,j] = x[j]
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
		if( step < burn ):
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
		elif( step >= burn ):
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

def GA( nProc, nGen, nPop, nParam, pFit, start, xLim, pWidth, nBin, bound, T, flips, flipProb, toFlip, toMix, burn, beta, mixAmp, mixProb, scrTM, scrMU, a, b, shift, nKeep, toDynStep, dynRate ):
	
	covInit = np.diag(pWidth**2)
	cov     = np.diag(pWidth**2)
	C       = np.diag(pWidth**2)
	mean    = deepcopy(start)
	
	scaleInit = np.prod( pWidth**2 )**(1.0/nParam)
	r = 1.0
	
	# all generations
	chain  = []
	chainF = []
	
	M = []
	
	# get initial population
	print "initial solutions"
	popSol    = getInitPop( nPop, nParam, pFit, start, xLim )
	popFit, X = evalPop( nProc, nPop, popSol, nParam, nBin, bound, T, scrTM, scrMU, a, b, shift )
	M.append(X)
	print np.sort(popFit)
	print " "
	
	# add init to all
	chain = [ popSol ]
	fit = [ popFit ]
	for i in range(nPop):
		chainF.append( popSol[i] )
	# end
	
	# Random walk
	for step in range(nGen):
		print "step: " + str(step+1) + "/" + str(nGen)
		
		if( toDynStep == 1 and step >= burn ):
			r = 1.0/np.log( np.e+dynRate*(step-burn) )
		# end
		print r
		
		# get covariance matrix
		cov, mean, C = getCovMatrix( cov, covInit, C, mean, beta, step, burn, nPop, nParam, chainF, popSol, r, scaleInit, toMix, mixProb, mixAmp )
		
		# perform selection
		parSol, parFit = Selection( nPop, popSol, popFit, nKeep )
		
		# perform crossover
		popSol = Crossover( nPop, nParam, parSol, parFit, cov, step, nGen )
		
		# perform mutation
		popSol = Mutate( step, nGen, nPop, nParam, popSol, cov, xLim, pFit, toFlip, flips, flipProb )
		
		# calculate fits
		popFit, X = evalPop( nProc, nPop, popSol, nParam, nBin, bound, T, scrTM, scrMU, a, b, shift )
		
		M.append(X)
		chain.append( popSol )
		fit.append( popFit )
		for i in range(nPop):
			chainF.append( popSol[i] )
		# end
		
		print np.sort(popFit)
		print " "
	# end
	chain = np.array(chain)
	fit   = np.array(fit)
	M     = np.array(M)
	
	return chain, fit, M

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

class TaskQueue(Queue.Queue):
	
	def __init__(self, num_workers=1):
		Queue.Queue.__init__(self)
		self.num_workers = num_workers
		self.start_workers()
	# end
	
	def add_task(self, task, *args, **kwargs):
		args = args or ()
		kwargs = kwargs or {}
		self.put((task, args, kwargs))
	# end
	
	def start_workers(self):
		for i in range(self.num_workers):
			t = Thread(target=self.worker)
			t.daemon = True
			t.start()
		# end
	# end
	
	def worker(self):
		while True:
			item, args, kwargs = self.get()
			item(*args, **kwargs)  
			self.task_done()
		# end
	# end
# end

def cumMin( chain1, fit1 ):
	
	fit2 = []
	chain2 = []
	
	curMax = fit1[0] - 1
	for i in range( len(fit1) ):
		if( fit1[i] > curMin ):
			fit2.append( fit1[i]   )
			chain2.append( chain1[i,:] )
			curMin = fit1[i]
		else:
			fit2.append( fit2[-1]   )
			chain2.append( chain2[-1] )
		# end
	# end
	fit2 = np.array(fit2)
	chain2 = np.array(chain2)
	
	return chain2, fit2
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
	
	print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in A]))
	
# end






main()








