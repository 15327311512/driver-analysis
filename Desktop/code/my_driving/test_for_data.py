#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:22:14 2019

@author: cyq
"""
import numpy as np
import sympy
import numpy
import scipy.io as scio

def maxent_irl(dis,vc,ac,jac,vo,ao,jao, lr):
  """
  Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

  inputs:
    feat_map    NxD matrix - the features for each state
    P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of 
                                       landing at state s1 when taking action 
                                       a at state s0
    gamma       float - RL discount factor
    trajs       a list of demonstrations
    lr          float - learning rate
    n_iters     int - number of optimization steps

  returns
    rewards     Nx1 vector - recoverred state rewards
  """
   theta = np.random.uniform(size=4)
  feat_exp = np.zeros([feat_map.shape[1]])
  for episode in trajs:
    for step in episode:
      feat_exp += feat_map[step.cur_state,:]
      #print feat_exp
  feat_exp = feat_exp/len(trajs)
  
  #print feat_map
  #print len(feat_map[0])
  #print len(feat_map)
  

  # training
  for iteration in range(n_iters):
  
    if iteration % (n_iters/20) == 0:
      print ('iteration: {}/{}'.format(iteration, n_iters))
    
    # compute reward function
    rewards = np.dot(feat_map, theta)

    # compute policy
    _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=False)
    
    # compute state visition frequences
    svf = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=False)
    
    # compute gradients
    grad = feat_exp - feat_map.T.dot(svf)

    # update params
    #print grad
    theta += lr * grad

    rewards = np.dot(feat_map, theta)
    return theta
if __name__ == '__main__':
    dataFile = '/home/cyq/Desktop/code/my_driving/data/encourter_data.mat'
    data1 = scio.loadmat(dataFile)
    dataFile2 = '/home/cyq/Desktop/code/my_driving/data/encourter_name.mat'
    data2 = scio.loadmat(dataFile2)
    print(data1['encourter_data'][2][4])
    
    dis=np.zeros(10000)
    vc=np.zeros(10000)
    ac=np.zeros(10000)
    jac=np.zeros(10000)
    vo=np.zeros(10000)
    ao=np.zeros(10000)
    jao=np.zeros(10000)
    for iteration in range(8000,9000):
        dis[iteration]=numpy.sqrt(numpy.square(data1['encourter_data'][iteration][4]-data1['encourter_data'][iteration][7])+numpy.square(data1['encourter_data'][iteration][5]-data1['encourter_data'][iteration][8]))
        vc[iteration+1]=1000000*numpy.sqrt(numpy.square(data1['encourter_data'][iteration+1][4]-data1['encourter_data'][iteration][4])+numpy.square(data1['encourter_data'][iteration+1][5]-data1['encourter_data'][iteration][5]))
        if (iteration>=8001):
            ac[iteration+1]=1000000*(vc[iteration]-vc[iteration-1])
        if (iteration>=8002):
            jac[iteration+1]=1000000*(ac[iteration]-ac[iteration-1])
        vo[iteration+1]=1000000*numpy.sqrt(numpy.square(data1['encourter_data'][iteration+1][7]-data1['encourter_data'][iteration][7])+numpy.square(data1['encourter_data'][iteration+1][8]-data1['encourter_data'][iteration][8]))
        if (iteration>=8001):
            ao[iteration+1]=1000000*(vo[iteration]-vo[iteration-1])
        if (iteration>=8002):
            jao[iteration+1]=1000000*(ao[iteration]-ao[iteration-1])
    maxent_irl(dis,vc,ac,jac,vo,ao,jao, 0.2)    
    print(theta)
    r=theta[0]+theta[1]+theta[2]
            
    
            


