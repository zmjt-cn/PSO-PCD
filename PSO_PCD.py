#!/usr/bin/python==========v2.7.5==========
# PSO_PCD.py===============v1.1.2==========
# ===='多晶超硬材料超精密抛光路径规划软件'====
# ==========='基于粒子群算法编写 ===========
# ======== 为转向抛光聚晶金刚石开发'========
# (依赖numpy库)
# Authors: Ren Yabin, Yang Bo
# Emails: <s1345358@126.com>, <boyang@hebut.edu.cn>
# Licence: Apache License 2.0

import math
import os
import sys
import numpy as np
import copy as cp
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# =================== 粒子群算法 ====================

class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim):
        self.__pos = np.random.uniform(0, x_max, (1,dim))  # 粒子的位置
        self.__vel = np.random.uniform(-max_vel, max_vel, (1,dim))  # 粒子的速度
        self.__bestPos = np.zeros((1, dim))  # 粒子最好的位置
        self.__fitnessValue = fit_fun(self.__pos)  # 适应度函数值

    def set_pos(self, value):
        self.__pos = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, value):
        self.__bestPos = value

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, value):
        self.__vel = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class PSO:
    def __init__(self, dim, size, iter_num, x_max, max_vel, tol, \
                 best_fitness_value=float('Inf'), C1=2, C2=2, W=1):
        self.C1 = C1
        self.C2 = C2
        self.W = W
        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max
        self.max_vel = max_vel  # 粒子最大速度
        self.tol = tol  # 截至条件
        self.best_fitness_value = best_fitness_value
        self.best_position = np.zeros((1, dim))  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值

        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim)\
                               for i in range(self.size)]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, value):
        self.best_position = value

    def get_bestPosition(self):
        return self.best_position

    # 更新速度
    def update_vel(self, part):
        vel_value = self.W * part.get_vel() + self.C1 * np.random.rand()\
              * (part.get_best_pos() - part.get_pos()) + self.C2 * np.random.rand() \
                * (self.get_bestPosition() - part.get_pos())
        vel_value[vel_value > self.max_vel] = self.max_vel
        vel_value[vel_value < -self.max_vel] = -self.max_vel
        part.set_vel(vel_value)

    # 更新位置
    def update_pos(self, part):
        pos_value = part.get_pos() + part.get_vel()
        vel_value = part.get_vel()
        for i in range(num_j):
            if pos_value[0,i]<0:
                pos_value[0,i]=0
                vel_value[0,i]=-vel_value[0,i]
            # elif pos_value[0,i]>=ΔD:
            #     pos_value[0,i]=ΔD
                # vel_value[0,i]=-vel_value[0,i]/2
        part.set_vel(vel_value)
        part.set_pos(pos_value)
        value = fit_fun(part.get_pos())
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            part.set_best_pos(pos_value)
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            self.set_bestPosition(pos_value)

    def update_ndim(self):
        global High
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue())  \
                # 每次迭代完把当前的最优适应度存到列表
            if i % 100 == 0:
                # High = High1
                print('第{}次最佳适应值为{}'.format(i, self.get_bestFitnessValue()))
            if self.get_bestFitnessValue() < self.tol:
                break

        return self.fitness_val_list, self.get_bestPosition()

# 总去除率公式
# dHij = eta(ij) * t(j) / S(i);
# Cv = sigma/mu; 目标函数 = 标准差/均值
# sigma = std(H_); mu = mean(H_);

# ================== 参数设置 ==================
# def get_gamma(High):
#     gamma = ((High - np.min(High))/np.max(High - np.min(High)))*(1-0.2)+0.2 # 高度差参数
#     return gamma

Dataname = 'DataOutput.txt'
fon = open(Dataname, 'w')
numtime = 20 # 生成PCD样品数
num_crystal = 30 # 晶粒数量
num_j = 13 # 总执行晶向 j条
dJ = math.pi/12 # 晶向间隔 pi/12
ii = []
SaSa = []
Sa11 = []

for n in range(numtime):
    J = np.arange(num_j)
    t = np.zeros(num_j)
    r = 100 # 平均晶粒尺寸 nm
    h = 5000 # 平均晶粒高度 nm
    sigma = 50
    ΔD_max = 2
    ΔD_min = 0.5 # 深度间隔 nm
    Sa = 0.5 # 程序终止条件（预期粗糙度）
    S = np.absolute(np.random.normal(math.pi*(r/2)**2, math.pi*10**2, (num_crystal,1))) \
        # 晶粒面积 正态分布 nm^2
    High00 = np.random.normal(0, sigma, (num_crystal,1))+h # 晶粒初始高度 nm Cv=0.025
    for _ in range(len(High00)):
        if High00[_,0] < 5000-40:
            High00[_,0] = 5000-40*(np.random.rand()*0.3+0.7)
        # elif High00[_,0] > 5000+40:
        #     High00[_,0] = 5000+40*(np.random.rand()*0.4+0.6)
    High0 = cp.copy(High00)
    High_ = cp.copy(High0)
    # High = High0
    def gamma1(x,ra):
        # gamma = ((x - np.min(x))/np.max(x - np.min(x)))*(1-0.5)+0.5 # 高度差参数
        ra1 = ra/10
        if ra1 >= 1:
            ra1 = 1
        gamma = ((x - np.min(x))/np.max(x - np.min(x)))*ra1+1-ra1
        return gamma
    dHighi = np.zeros((num_crystal, 1))
    dHighij = np.zeros((num_crystal, num_j)) # 初始化去除高度矩阵 nm
    # Cv0 = np.std(High0)#/np.mean(High0)
    numRa = int(np.rint(np.sqrt(num_crystal)))
    Sa00 = np.sum(np.dot(S.T,np.abs(High0-np.mean(High0)))/np.sum(S)) # 表面粗糙度
    # Ra00 = np.sum(np.dot(np.sqrt(S[:numRa]).T,\
    #                      np.abs(High0[:numRa]-np.mean(High0[:numRa])))/np.sum(np.sqrt(S[:numRa])))
    Sa0 = cp.copy(Sa00)
    Sa1 = cp.copy(Sa0)
    angle = np.random.uniform(0, 1, (num_crystal,1))*math.pi # 晶粒取向(最高去除率)
    phi_angle = (angle//dJ).astype(int) + ((angle%dJ - dJ/2)>0).astype(int) # 量化晶粒取向
    plane = np.random.randint(0,3,(num_crystal,1)) # 晶粒晶面指数分布-->0(001),1(110),2(111)
    # plane[0:math.floor(num_crystal/3),0] = 2
    # plane = np
    # t0[np.where(plane == 0)[0]]=1
    print(plane.T)
    eta = np.zeros((num_crystal, num_j)) # 初始化去除率矩阵 nm^3/min
    print('晶粒面积 S = '+str([format(S.T[0,x],'.2f') for x in range(num_crystal)])+'nm2\n')
    # ================== 覆写去除率矩阵 ==================
    for i in range(num_crystal):
        if plane[i,0] == 0: # (001)
            eta_dat = 'Rate_data/eta001.txt'
            star0 = 12
        elif plane[i,0] == 1: # (110)
            eta_dat = 'Rate_data/eta110.txt'
            star0 = 12
        elif plane[i,0] == 2: # (111)
            eta_dat = 'Rate_data/eta111.txt'
            star0 = 16
        fo = open(eta_dat, 'r')
        eta_i = []
        for line in fo:
            eta_i.append(line.split())
        fo.close()
        for k in range(25,48):
            eta_i.append([str(k),eta_i[k-24][1]])
        eta[i,:] = np.array(eta_i).astype(float)[(star0-phi_angle[i,0]):(star0-phi_angle[i,0]+13),1]

    # ================== Main program ==================
    def fit_fun(t):
        global High1, High1_, dHighij, dHighi #, Cv1, Ra1
        # gamma = get_gamma(High)
        for i in range(num_crystal):
            for j in range(num_j):
                dHighij[i,j] = eta[i,j] * t[0,j]
        for ij in range(num_crystal):
            dHighi[ij,0] = np.sum(dHighij[ij,:])
            if np.sum(High0[ij,0] - np.max(High0)) <= -ΔD:
                dHighi[ij,0] = 0
            else:
                dHighi[ij,0] = dHighi[ij,0]*np.sum(High0[ij,0] - np.max(High0)+ΔD)/ΔD
        num1 = np.where(High0 <= (np.sum(np.max(High0))-ΔD)) # 索引不去除晶粒编号
        num2 = np.where(High0 > (np.sum(np.max(High0))-ΔD))
        High_[num1] = (np.sum(np.max(High0))-ΔD*(np.random.rand()*0.2+1))
        High1_ = High_ - dHighi
        High1 = High0 - dHighi
        # Cv1 = np.std(High1)#/np.mean(High1)
        Sa1_ = getSa(High1_)
        # np.sum(np.dot(S.T,np.abs(High1-np.mean(High1)))/np.sum(S))
        # Ra1 = np.sum(np.dot(np.sqrt(S[:numRa]).T,\
        #                     np.abs(High1[:numRa]-np.mean(High1[:numRa])))/np.sum(np.sqrt(S[:numRa])))
        return1 = Sa1_
        return return1

    def getSa(HH):
        Sag = np.sum(np.dot(S.T,np.abs(HH-np.mean(HH)))/np.sum(S))
        return Sag    

    if __name__ == '__main__':
        # test
        Sa2 = Sa0
        Sa3 = Sa0
        ΔD = cp.copy(ΔD_max)
        for i in range(200):
            print('i =',i,'ΔD=',ΔD)
            # gamma = gamma1(High0,Sa0)
            pso = PSO(13, 20, 401, 10, 1, 1e-4, C1=2, C2=2, W=1.5)
            fit_var_list, best_pos = pso.update_ndim()
            t = t + best_pos
            # print(np.around(t, 4))
            fit_fun(best_pos)
            Sa1_ = getSa(High1_)
            Sa1 = getSa(High1)
            print("第"+str(i)+"次抛光时间:t=" + str(best_pos.astype(str))+'\n')
            print("总抛光时间:t=" + str(np.around(t,4))+'\n')
            print('Sa0='+str(np.around(Sa0,2))+'nm','Sa1_='+str(np.around(Sa1_,2))+'nm',\
                  "Sa1=" + str(np.around(Sa1,2))+'nm\n')
            #plt.plot(range(len(fit_var_list)), fit_var_list, alpha=0.5)
            print('High =',str(np.around(High1.T, 2))+'\n')
            nn = i
            if Sa1 <= ΔD*1.2 or np.sum(best_pos) == 0:
                ΔD = ΔD*3/4
                if ΔD <= ΔD_min:
                    ΔD = ΔD_min
            if Sa1 <= 10 and np.sum(best_pos) == 0:
                t0 = np.zeros((1,13)); t0[0,0:2] = 0.2
                ΔD = ΔD*2
                if ΔD >= 5:
                    ΔD = 5
                fit_fun(t0)
                t = t + t0
            if Sa1 <= Sa:
                break
            Sa3 = Sa2
            Sa2 = Sa0
            Sa0 = Sa1
            High0 = High1
    
    # [Cv, std, mean] = fit_fun(t)
    fon.write('n = '+str(n+1)+'  i = '+str(nn+1)+'  深度间隔 '+str(ΔD)+'nm\n')
    fon.write('抛光时间 t = '+str([format(t[0,x],'.2f') for x in range(num_j)])+'min\n')
    fon.write('晶面分布 = '+str([int(plane.T[0,x]) for x in range(num_crystal)])+'\n')
    fon.write('晶向分布 = '+str([int(phi_angle.T[0,x]) for x in range(num_crystal)])+'\n')
    fon.write('初始Sa0 = '+str(np.around(Sa00, 2))+'nm\t')
    fon.write('最终Sa1 = '+str(np.around(Sa1, 2))+'nm\n')
    # fon.write('\nCv0 = '+str(np.around(Cv0, 2)))
    # fon.write('\tCv1 = '+str(np.around(Cv1, 2)))
    fon.write('初始高度High0 = '+str([format(High00.T[0,x],'.2f') for x in range(num_crystal)])+'nm\n')
    fon.write('最终高度High1 = '+str([format(High1.T[0,x],'.2f') for x in range(num_crystal)])+'nm\n')
    fon.write('晶粒面积 S = '+str([format(S.T[0,x],'.2f') for x in range(num_crystal)])+'nm2\n\n')
    # print('初始Ra0='+str(Ra0),'最终Ra1='+str(fit_var_list[-1])+'\n')
    # print('初始变异系数='+str(Cv0),'最终系数='+str(Cv1))
    ii.append(nn+1)
    SaSa.append(np.around(Sa00, 2))
    Sa11.append(np.around(Sa1, 2))
fon.write('分段数 = '+str(ii)+'\t'+'平均分段数 = '+str(np.mean(ii))+'\n')
fon.write('初始Sa = '+str(SaSa)+'\t'+'平均初始Sa = '+str(np.mean(SaSa))+'\n')
fon.write('最终Sa = '+str(Sa11)+'\t'+'平均最终Sa = '+str(np.mean(Sa11))+\
          '\t'+'最大最终Sa = '+str(np.max(Sa11))+\
            '\t'+'最小最终Sa = '+str(np.min(Sa11))+'\n')
fon.close()
1 # 断言用，无具体含义

# Changelog #
# -------- (SC_NPD.m) v0.0 --------
# v0.0.0, 1st Written May 27, 2023. MATLAB file. Original version. Modified for the first time.
# v0.0.1, 2nd Written Jun 12, 2023. Program optimization.
# ------- (test001.py) v0.1 -------
# v0.1.0, 1st Overhaul Jun 7, 2023. Program renaming & optimization.
# ------- (PSO_PCD.py) v1.0 -------
# v1.0.0, 2nd Overhaul Sep 22, 2023. Program renaming & optimization. PCD粒子群算法(去除率固定)
# v1.0.1, 2nd written Oct 7, 2023. Program rewrite. 梯度PCD粒子群算法(可变去除率,引入高度差参数)
# v1.0.2, 3rd written Oct 19, 2023. Program testing & optimization.
# v1.0.3, 4th written Nov 2, 2023. Program testing & optimization.
# v1.0.4, 5th written Nov 14, 2023. Program testing & optimization.
# v1.0.5, 6th written Dec 7, 2023. Program testing & optimization.
# ------- (PSO_PCD.py) v1.1 -------
# v1.1.0, 3rd Overhaul Dec 7, 2023. Program structure adjustment and optimization. 分段PCD粒子群算法(按ΔD分段)
# v1.1.1, 2nd written Jan 8, 2024. Bug fixes & program optimization.
# v1.1.2, 3rd written Jan 23, 2024. Program optimization.
# ...
