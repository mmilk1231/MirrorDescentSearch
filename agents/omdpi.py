# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import math
import numpy as np
import os
import glob
import sys

import gaussianFit as gF
import RL
import Accelerated as ac
import projections as proj
import GMM_sample as GMMs
import scipy
from scipy.stats import entropy

class OMDPI:
    def __init__(self, RL):
        self.RL = RL
        timeStep = self.RL.n*self.RL.times
        dimension = self.RL.n_dims
        rfs = self.RL.n_rfs
        reps = self.RL.reps
        timeIndex = range(timeStep)        
        self.G = np.zeros((reps,dimension,timeStep,rfs))        
        self.psi = np.zeros((reps,timeStep,rfs))
        
    def init(self):
        self.f = open(self.RL.path+'/'+'Entropy-'+self.RL.MethodMode+'.csv', 'a')
        timeStep = self.RL.n*self.RL.times
        # muはrepsに依存しないため，1とした
        self.mu0=np.zeros([1, self.RL.n_dims, timeStep, self.RL.n_rfs])
        self.mu0[0,:,0,:]=self.RL.meanIni+np.zeros([self.RL.n_dims, self.RL.n_rfs])
        #self.mu0[0,:,0,:]=np.array([[.55, .25, .2]]).T
        #TODO 念のためtimestep全体にコピー
        for t in range(timeStep):
            self.mu0[0,:,t,:]=self.mu0[0,:,0,:]

        self.distroIni=[\
                [self.RL.lambdakIni, self.mu0],\
                [1.0-self.RL.lambdakIni, self.mu0]]

        # AMDクラスの初期化
        self.MDmethod=self.__amd_init(self.RL.MethodMode)

        # パラメータの初期化
        self._tmp = [np.random.rand() for _ in range(self.RL.reps)]
        self.x0 = self.ztilde0 = np.array(self._tmp)/sum(self._tmp)
        self.x=self.x0
        self.ztilde=self.ztilde0
        self.distro=self.distroIni
        self.lambdak=self.RL.lambdakIni
        
    def __amd_init(self, methodMode):
        # dimension: num of Rollouts
        d = self.RL.reps

        precision = 1e-10
        epsilon = .3
        # Simplex constrained projections
        psExpS = proj.SimplexProjectionExpSort(dimension = d, epsilon = epsilon)
        psExpS0 = proj.SimplexProjectionExpSort(dimension = d, epsilon = 0)
                
        h = self.RL.h
        r = 3
        p1 = psExpS
        p2 = psExpS0
        s1 = h*p1.epsilon/(1.0+d*p1.epsilon)
        s2 = h

        #x0 = np.random.rand(d).T
        
        x0 = np.ones(d).T
        x0 = x0/np.sum(x0)

        if(methodMode == 'acc'):
            method = ac.AcceleratedMethod(p1, p2, s1, s2, r, x0, 'accelerated descent')
            #method = ac.AcceleratedMethodWithRestartGradScheme(p1, p2, s1, s2, r, x0, 'gradient restart')
        elif(methodMode == 'norm'):
            method = ac.MDMethod(p2, s2, x0, 'mirror descent')
        else:
            error('unknown method mode in OMDPI')

        return method

    def __approximate(self, data, xtilde, ztilde, lambdak, params1, params2):
        '''
        mux, sigmax = gF.solver2(np.squeeze(data), xtilde)
        muz, sigmaz = gF.solver2(np.squeeze(data), ztilde)
        mux1 = np.array([[[[mux]]] for i in range(10)])
        muz1 = np.array([[[[muz]]] for i in range(10)])
        params=[[lambdak, mux1, sigmax], [1.0-lambdak, muz1, sigmaz]]
        return params
        '''
        
        reps = self.RL.reps
        timestep = self.RL.n*self.RL.times
        mean1 = np.squeeze(np.reshape(params1[1][0,:,0,:],[1,-1]))
        mean2 = np.squeeze(np.reshape(params2[1][0,:,0,:],[1,-1]))

        Delta1=0
        Delta2=0

        #theta1=0
        #theta2=0
        for i in range(reps):
            _data = np.squeeze(np.reshape(data[i,:,0,:],[1,-1]))
            epsilon1 = _data-mean1
            epsilon2 = _data-mean2

            # ノイズの期待値を計算
            Delta1 += epsilon1*xtilde[i]
            Delta2 += epsilon2*ztilde[i]
            #theta1 += _data*xtilde[i]
            #theta2 += _data*ztilde[i]

        # θの更新
        theta1=np.add(mean1,Delta1)
        theta2=np.add(mean2,Delta2)

        theta11=np.reshape(theta1,(1, self.RL.n_dims, 1, self.RL.n_rfs))
        theta22=np.reshape(theta2,(1, self.RL.n_dims, 1, self.RL.n_rfs))
        #TODO ごり押しでtimestep分コピー
        theta11=np.tile(theta11[:], (timestep,1))
        theta22=np.tile(theta22[:], (timestep,1))

        # weight, meanを更新
        params=[[lambdak, theta11], [1.0-lambdak, theta22]]
        return params

    def __update(self, data, distro, stdEps, mean, r):
        rfs = self.RL.n_rfs
        dims = self.RL.n_dims
        valMat = stdEps**2*np.matrix(np.identity(dims*rfs))

        # 累積報酬和を計算
        S=np.rot90(np.rot90((np.cumsum(np.rot90(np.rot90(r.T)), 0)))) 

        # weight, mean, variance
        distroX = [distro[0][0],distro[0][1], valMat]
        distroZ = [distro[1][0],distro[1][1], valMat]

        # xの離散分布を求める
        x = GMMs.data_GMMs_likelihood(data, distroX, distroZ)
        #xtilde = GMMs.data_normalized_likelihood(data, distroX)
        
        # ztildeの離散分布を求める
        ztilde = GMMs.data_normalized_likelihood(data, distroZ)

        # 正規化
        g = S[0,:]
        g = (g-min(g))/(max(g)-min(g))

        if(not self.RL.skipMode):
            self.x = x
            self.ztilde = ztilde
        # AMDに従って分布を更新
        if(self.RL.ztildeXequal):
            x, xtilde, ztilde, lambdak = self.MDmethod.step(g, np.ones_like(self.x)/sum(np.ones_like(self.x)), np.ones_like(self.ztilde)/sum(np.ones_like(self.ztilde)))
        else:
            x, xtilde, ztilde, lambdak = self.MDmethod.step(g, self.x, self.ztilde)
        if(self.RL.skipMode):
            self.x = x
            self.ztilde = ztilde
        self.f.write(str(entropy(self.x, self.ztilde, 2))+'\n')
        #self.f.write(str(entropy(x, ztilde, 2))+'\n')
#        print('x: '+str(self.x))
#        print('ztilde: '+str(self.ztilde))
#        print('entropy: '+str(entropy(self.x, self.ztilde, 2)))

        # 連続分布を更新
        return self.__approximate(data, xtilde, ztilde, lambdak, distroX, distroZ)

    def act_and_train(self, obs, reward):
        return action

    def __rollouts(self, reps, distro, stdEps):
        timeStep = self.RL.n*self.RL.times
        rfs = self.RL.n_rfs
        dims = self.RL.n_dims
        
        valMat = stdEps**2*np.matrix(np.identity(dims*rfs))
        # weight, mean, variance
        distro1 = [distro[0][0],distro[0][1],valMat]
        distro2 = [distro[1][0],distro[1][1],valMat]

        mean=np.zeros([reps, dims, timeStep, rfs])
        data=np.zeros([reps, dims, timeStep, rfs])
        for k in range(reps):
            # 2つの分布からサンプリング
            data[k,:,0,:], mean[k,:,0,:]=GMMs.GMM_sample(distro1, distro2)
        for t in range(timeStep):
            data[:,:,t,:]=data[:,:,0,:]
            mean[:,:,t,:]=mean[:,:,0,:]
        R=np.zeros((reps,timeStep))
        for k in range(reps):
            R[k] = self.RL.task.step(reps, mean, data - mean, self.G, self.psi, k)
        return np.array(R), mean, data

    def act(self, obs, gDof, _epsilon, theta, t):
        dof = self.RL.n_dims
        xi, dxi, ddxi = obs[t]
        # Todo: 10 = rfs
        action = np.zeros((dof,10))
        #報酬の計算（t=100というのは経由点を通りすぎて欲しい時間）
        for d in range(dof):
            gTg=np.sum(np.array(gDof[d])*np.array(gDof[d]))
            gTeps=np.array(gDof[d])*np.array(_epsilon[d][t])
            Meps=gDof[d]*gTeps/(gTg+1.e-10)
            action[d] = theta[d]+Meps
        return action
    
    def simulate(self, reps, step):        
        distro=self.distro

        # 探索ノイズ
        noiseMult = float(self.RL.updates-step)/float(self.RL.updates)
        noiseMult = np.max((0.1, noiseMult))
        stdEps = self.RL.std*noiseMult

        # ロールアウト
        r,mean,data = self.__rollouts(reps, distro, stdEps)

        # 更新
        if(reps>1): distro = self.__update(data, distro, stdEps, mean, r)

        # ノイズのないロールアウト
        r,_,_ = self.__rollouts(1, distro, 0)

        # コストの計算
        self.RL.cost[step] = np.sum(r)
        self.distro = distro
