#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import interpolate
from multiprocessing import Pool
from numba import jit
import warnings#不显示warnings
import os


warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)#不显示warning


def getQinByRnn(Position,Q_Data):
    '''

    :param Position:电站位置 1：二级 2 三级
    :param Q_Data: 上游下泄流量过程 Td-8~Td+1-8
    :return:
    '''
    data_path="D:/desktop/pywork/LsTM/Dataset2"+str(Position)+".xlsx"
    data=pd.read_excel(data_path,index_col="Time")
    normolize_data = (data - np.mean(data)) / np.std(data)
    #X_data,Y_data=normolize_data.iloc[500:,:-1],normolize_data.iloc[500:,-1]
    xmean=np.mean(data)[-2]
    xstd=np.std(data)[-2]
    ymean = np.mean(data)[-1]
    ystd = np.std(data)[-1]
    Q_Data_normolize=(Q_Data-xmean)/xstd
    #加载已训练的模型
    model_path="D:/desktop/pywork/lstm/model"+str(Position)+"/"
    with tf.Session() as sess:
        #载入图
        saver = tf.train.import_meta_graph(model_path + 'lstm_model0.ckpt.meta')
        #载入参数
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        #载入变量
        x=tf.get_default_graph().get_tensor_by_name('x:0')
        pred=tf.get_default_graph().get_tensor_by_name('pred:0')
        #print('Successfully load the pre-trained model!')
        predict=[]
        for i in range(np.shape(Q_Data)[0]-x.shape[1]+1):
            X_data=Q_Data_normolize[i:i+x.shape[1]]
            next_seq=sess.run(pred,feed_dict={x:X_data.reshape(-1,x.shape[1],x.shape[2])})
            predict.append(next_seq)

        Q_in=np.array([value for value in np.reshape(predict, [-1])]) * ystd + ymean
        #Q_in Td~Td+1
        return Q_in

def read_data():
    '''
    读取数据
    '''
    data_place="D:/desktop/pywork/LsTM/changhuang.xlsx"


    data = pd.read_excel(data_place,sheet_name="基础参数",header=1)
    z_vinc=data.loc[data.loc[:,'库水位']>=0,['库水位', '库容']].values
    z_vinh=data.loc[data.loc[:,'库水位1']>=0,['库水位1', '库容1']].values
    w_qinc=data.loc[data.loc[:,'尾水位']>=0,['尾水位', '流量']].values
    w_qindh=data.loc[data.loc[:,'尾水位1']>=0,['尾水位1', '流量1']].values
    w_qinxh=data.loc[data.loc[:,'尾水位2']>=0,['尾水位2', '流量2']].values

    data2=pd.read_excel(data_place,sheet_name="实际运行资料", header=0)
    Q_c=data2.loc[data2.loc[:,"Q_c"]>=0,'Q_c'].values
    Q_qu=data2.loc[data2.loc[:,"Q_qu"]>=0,'Q_qu'].values
    Q_h1=data2.loc[data2.loc[:,"Q_h"]>=0,'Q_h'].values
    N_all=data2.loc[data2.loc[:,"N_allkw"]>=0,'N_allkw'].values
    N_ct=data2.loc[data2.loc[:,"N_ckw"]>=0,'N_ckw'].values

    return z_vinc,z_vinh,w_qinc,w_qindh,w_qinxh,Q_c,Q_qu,Q_h1,N_all,N_ct



class powerstation():
    '''
    电站类
    '''
    def __init__(self,name,N_min, N_max, s_min,s_max,q_min, q_max, A, Z_min, Z_max, z_v, w_q, h_loss,gnum=1,vib=np.zeros((2,)),unit=1):
        self.name=name#电站名称
        self.unit=unit#机组台数
        self.getenum=gnum#闸门数目
        self.N_min=N_min#最小出力
        self.N_max=N_max#最大出力
        self.s_min=s_min#最小下泄流量
        self.s_max=s_max#最大下泄流量
        self.q_min=q_min#最小发电流量
        self.q_max=q_max#最大发电流量
        self.A=A#出力系数
        self.Z_max=Z_max#最高水位
        self.Z_min=Z_min#最低水位
        self.z_v=z_v#水位库容曲线
        self.w_q=w_q#尾水位流量曲线
        self.h_loss=h_loss#水头损失
        self.vibrange=vib#机组振动区
        self.gnum=gnum#闸门个数
        # self.Z_b=0#初水位
        # self.Z_e=0#末水位
        # self.Q_in=0#入库流量
        # self.q_fd=0#发电流量
        # self.q_loss=0#弃水流量
        # self.q_down=0#下泄流量
        # self.N_each=np.zeros(unit)#机组出力
        # self.gate=np.zeros(gnum)#各个闸门
    def N_get_z(self,N,t,z0,Q,Z_max,Z_min):
        '''由出力获取水位'''
        z1_max=Z_max
        z1_min=Z_min
        N_temp=0
        while True:
            z1=(z1_min+z1_max)/2.0
            if abs(z1-Z_max)<=0.0001:#来水过大 水库需要产生非满发弃水
                z1=z1_max
                r=self.z_get_N2(z1,t,z0,Q,N)
                if r=='cant':
                    return 'cant'
                else:
                    q,qloss=r
                    return N,z1,q,qloss
            elif abs(z1-Z_min)<=0.0001: #来水过小 为满足下泄流量而弃水
                r=self.z_get_N3(t,z0,Q,N)
                if r=='cant':
                    return 'cant'
                else:
                    z1,q,qloss=r
                    return N,z1,q,qloss
            elif abs(z1_max-z1_min)<=0.001:#计算精度问题 以能算出结果为目标
                k=0
                while True:
                    k+=1
                    if k>2000:
                       return 'cant'
                    if (z1==Z_max)|(z1==Z_min):
                        return 'cant'
                    r = self.z_get_N5(z1, t, z0, Q, N)
                    if r == 'cant0':#下泄小
                        z1 = max(z1 - 0.0005, Z_min)
                    elif r=='cant1':#下泄大
                        z1 = min(z1 + 0.0005, Z_max)
                    elif r=='cant2':#下泄不够完成出力
                        z1 = max(z1 - 0.0005, Z_min)
                    elif r == 'cant3':  # 水头不够
                        z1=min(z1+0.0001,Z_max)
                    else:
                        q, qloss = r
                        return N, z1, q, qloss
            r=self.z_get_N(z1,t,z0,Q)
            if r=='cant0':
                z1_max=z1
                continue
            elif r=='cant1':
                z1_min=z1
                continue
            else:
                N_temp,q,q_loss=r
                if abs(N_temp-N)<5000:
                    return N_temp,z1,q,q_loss
                elif N_temp>N:
                    z1_min=(z1+z1_min)/2
                elif N_temp<N:
                    z1_max=(z1+z1_max)/2

    def z_get_N5(self, z1, t, z0, Q, N):  # 以满足出力为原则 用于调节末水位
        if z1 == z0:
            q = Q
        else:
            v0 = interpolate.interp1d(self.z_v[:, 0], self.z_v[:, 1], kind='linear')(z0)
            v1 = interpolate.interp1d(self.z_v[:, 0], self.z_v[:, 1], kind='linear')(z1)
            q = Q - ((v1 - v0) * (10 ** 8)) / t
        if q <= self.s_min:
            return 'cant0'
        elif q > self.s_max:
            return 'cant1'
        w = interpolate.interp1d(self.w_q[:, 1], self.w_q[:, 0], kind='linear')(q)
        qf = N / (self.A * ((z1 + z0) / 2 - w - self.h_loss))
        q_loss = q - qf
        if q_loss < 0:
            return 'cant2'
        elif qf > self.q_max:
            return 'cant3'
        return qf, q_loss

    def z_get_N2(self,z1,t,z0,Q,N):#以满足出力为原则 到达最高水位产生非满发弃水
        if z1==z0:
            q=Q
        else:
            v0=interpolate.interp1d(self.z_v[:,0],self.z_v[:,1],kind='linear')(z0)
            v1=interpolate.interp1d(self.z_v[:,0],self.z_v[:,1],kind='linear')(z1)
            q=Q-((v1-v0)*(10**8))/t
        if q<=self.s_min:
            return 'cant'
        if q>self.s_max:
            return 'cant'
        w=interpolate.interp1d(self.w_q[:,1],self.w_q[:,0],kind='linear')(q)
        qf=N/(self.A*((z1+z0)/2-w-self.h_loss))
        q_loss=q-qf
        if qf>self.q_max:
            return 'cant'
        if q_loss<0:
            return 'cant'
        return qf,q_loss



    def z_get_N3(self,t,z0,Q,N):#以满足出力为原则 保证下泄生态流量而弃水
        v0=interpolate.interp1d(self.z_v[:,0],self.z_v[:,1],kind='linear')(z0)
        v1=(Q-self.s_min)*t/(10**8)+v0
        z1=interpolate.interp1d(self.z_v[:,1],self.z_v[:,0],kind='linear')(v1)
        if (z1>self.Z_max)|(z1<self.Z_min):
            return 'cant'
        w=interpolate.interp1d(self.w_q[:,1],self.w_q[:,0],kind='linear')(self.s_min)
        q_loss=0
        qf=N/(self.A*((z1+z0)/2-w-self.h_loss))
        if qf>self.s_min:
            return 'cant'
        q_loss=self.s_min-qf
        return z1,qf,q_loss

    def z_get_N4(self,z1,t,z0,Q,N):#满足发电为原则 产生非常规弃水 用于特殊调度
        v0=interpolate.interp1d(self.z_v[:,0],self.z_v[:,1],kind='linear')(z0)
        v1=interpolate.interp1d(self.z_v[:,0],self.z_v[:,1],kind='linear')(z1)
        q=Q-((v1-v0)*(10**8))/t
        if q<=self.s_min:
            return 'cant0'
        if q>self.s_max:
            return 'cant1'
        w=interpolate.interp1d(self.w_q[:,1],self.w_q[:,0],kind='linear')(q)
        qf=N/(self.A*((z1+z0)/2-w-self.h_loss))
        q_loss=q-qf
        if q_loss<0:
            return 'cant0'
        return qf,q_loss


    def z_get_N(self,z1,t,z0,Q):#以弃水最小发电最大为原则 在满发弃水
        v0=interpolate.interp1d(self.z_v[:,0],self.z_v[:,1],kind='linear')(z0)
        v1=interpolate.interp1d(self.z_v[:,0],self.z_v[:,1],kind='linear')(z1)
        q=Q-((v1-v0)*(10**8))/t
        if q<=self.s_min:
            return 'cant0'
        if q>self.s_max:
            return 'cant1'
        w=interpolate.interp1d(self.w_q[:,1],self.w_q[:,0],kind='linear')(q)
        q_loss=0
        if q>self.q_max:
            q_loss=q-self.q_max
            q=self.q_max
        N=self.A*((z1+z0)/2-w-self.h_loss)*q
        return N,q,q_loss





def GetStationProccess(N_goal,Z_start, Q_in,Station):
    '''
    由出力过程计算水位过程
    考虑站内机组分配直接按避开振动区分配
    '''
    timestep=len(N_goal)
    q_fd = np.zeros(timestep)
    q_loss = np.zeros(timestep)
    Z_up = np.zeros(timestep+1)
    N_unit = np.zeros([Station.unit, timestep])  # 站内机组分配使用
    N_temp = np.zeros(timestep)  # 站内机组分配使用
    Z_up[0]=Z_start#初始水位
    for t in range(timestep):
        # 保证水位变幅不超过1m?
        Z_max = min(Z_up[t] + 1, Station.Z_max)
        Z_min = max(Z_up[t] - 1, Station.Z_min)
        r= Station.N_get_z(N_goal[t], 15 * 60, Z_up[t], Q_in[t], Z_max, Z_min)
        if r == 'cant':
            return "cant"
        else:
            N_temp[t], Z_up[t + 1], q_fd[t], q_loss[t] = r
    return q_fd,q_loss,Z_up,N_temp


def GetCascadeProcess(N_goal_list,Z_start_list,Q_in_head,StationList):
    '''
    梯级衔接过程 考虑水力联系
    :param N_goal_list:各电站负荷过程
    :param Z_start_list:各电站初始水位
    :param Q_in_head:
    :param StationList:
    :return:
    '''
    numStation=len(StationList)
    timeStep=len(Q_in_head)
    Q_in_list=np.zeros([numStation,timeStep])
    q_fd_list=np.zeros([numStation,timeStep])
    q_loss_list=np.zeros([numStation,timeStep])
    N_unit_list=[None]*numStation
    Z_up_list=np.zeros([numStation,timeStep+1])
    N_temp_list=np.zeros([numStation,timeStep])
    for i in range(len(StationList)):
        if(i==0):
            Q_in=Q_in_head
        else:
            Q_data=np.hstack([np.array([ q_fd_list[i-1,0]+q_loss_list[i-1,0] for j in range(5)]),q_fd_list[i-1,:-2]+q_loss_list[i-1,:-2]])  #上游电站下泄流量 前面直接填充
            Q_in=getQinByRnn('',Q_data)#用训练好的模型计算入库流量
        r=GetStationProccess(N_goal_list[i],Z_start_list[i],Q_in,StationList[i])
        if r=='cant':
            return False
        else:
            q_fd_list[i,:],q_loss_list[i,:],Z_up_list[i,:],N_temp_list[i,:]=r
            Q_in_list[i,:]=Q_in[:]
    return Q_in_list,q_fd_list,q_loss_list,Z_up_list,N_temp_list


def chulixianzhi(stationlist,N_goal):
    '''
    缩小可行出力范围 减小计算时间
    '''
    numStation=len(stationlist)
    timestep=len(N_goal)
    N_max=np.zeros([numStation,timestep])
    N_min = np.zeros([numStation, timestep])
    for i in range(len(stationlist)):
        N_max[i,:]=min(N_goal[i]-np.sum([station.N_min for station in stationlist])+stationlist[i].N_min,stationlist[i].N_max)
        N_min[i,:]=max(N_goal[i]-np.sum([station.N_max for station in stationlist])+stationlist[i].N_max,0)
    return N_max,N_min

def initpopvfit(sizepop,Z_start,N_goal,Q_in_head,stationlist):
    '''
    初始化种群
    #以各个电站的出力过程为粒子
    '''
    numStation=len(stationlist)
    timestep=len(N_goal)
    N_temp_list=np.zeros([numStation,timestep])
    pop=np.zeros((sizepop,numStation,timestep))
    v=np.zeros((sizepop,numStation,timestep))
    i=0
    N_max,N_min=chulixianzhi(stationlist,N_goal)
    while i<sizepop:

        for p in range(numStation):
            w=stationlist[p].N_max/sum([ s.N_max for s in StationList])
            for t in range(timestep):
                rd = np.random.rand()
                p_temp=(w+(rd*0.3-0.15))*N_all[t]
                if p_temp>N_max[p,t]:
                    pop[i, p, t]=N_max[p,t]
                if p_temp<N_min[p,t]:
                    pop[i, p, t] = N_max[p, t]
                else:
                    pop[i,p,t]=p_temp.copy()
            pop[i,-1]=N_goal-np.sum(pop[i,:numStation-1],axis=0)
        if (np.sum(pop[i,-1]>N_max[-1])+np.sum(pop[i,-1]<N_min[-1]))>1:
            continue
        r=GetCascadeProcess(pop[i],Z_start,Q_in_head,stationlist)
        if r==False:
            continue
        else:
            print("get pop num :%d" %(i+1))
            i+=1
    print('初始种群获取完成')
            
    return pop,v

def get_fitness(particle,Z_start_list,Q_in_head,StationList,N_max,N_min):
    '''
    计算种群的适应度值
    '''
    if (np.sum(particle[-1] > N_max[-1]) + np.sum(particle[-1]< N_min[-1])) > 1:
        return 0
    r=GetCascadeProcess(particle,Z_start_list,Q_in_head,StationList)
    if r==False:#仅设置了硬性约束，不满足则将适应度归零
        return 0

    else:
        Q_in_list,q_fd_list,q_loss_list,Z_up_list,N_temp_list=r
        #目标函数设置为耗水率最小 m^3/s / kw·h   加入弃水平稳惩罚项
        adj_gate=np.sum(np.abs(q_loss_list[:,1:]-q_loss_list[:,:-1])>100)

        fitness=(np.sum(N_temp_list)/4)/np.sum(q_fd_list)-adj_gate

    return fitness

#---------------------------------------------------------------------------------------------------------------------------
#粒子群算法
#定义参数
if __name__ == '__main__':
    wmax = 1#惯性权重
    wmin = 0.4
    lr = (1.49618,1.49618)#加速常数/学习因子
    maxgen = 100#最大迭代数
    sizepop =20#种群数
    z_vinc,z_vinh,w_qinc,w_qindh,w_qinxh,Q_c,Q_qu,Q_h1,N_all,N_ct=read_data()
    StationList=[]
    StationList.append(powerstation("changheba",0, 2600000, 166.5,5077, 0, 1451.72, 8.5, 1650, 1690, z_vinc, w_qinc, 0 ))
    StationList.append(powerstation("huangjinping",0,800000,168,5336,0,1414,8.5,1472,1476,z_vinh,w_qindh,0))
    N_max,N_min=chulixianzhi(StationList,N_all)
    numStation=len(StationList)
    timestep=len(Q_c)
    #print(N_c_max,N_c_min)
    speedrange ={'max':(N_max-N_min)/10 ,'min':-(N_max-N_min)/10}  #飞行速度范围最大值取出力范围的1/10
    Z_start_list=np.array([1678.78,1472.90])
    pop,v=initpopvfit(sizepop,Z_start_list,N_all,Q_c,StationList)
    #print(pop)
    fitness=np.zeros(sizepop)
    for i in range(sizepop):
        fitness[i]=get_fitness(pop[i],Z_start_list,Q_c,StationList,N_max,N_min)
    #print(fitness)
    gbestpop,gbestfitness = pop[fitness.argmax()].copy(),fitness.max()#群体极值获取
    pbestpop,pbestfitness = pop.copy(),fitness.copy()#初始种群的个体极值为其本身
    result=np.zeros(maxgen)
    result[0]=gbestfitness
    gen=1
    while gen<maxgen:#迭代上限
        print("第%d代更新" %gen)
        #速度更新
        #w=wmax-gen*(wmax-wmin)/maxgen
        for i in range(sizepop):
            for p in range(numStation-1):
                for d in range(96):
                    v[i,p,d]=v[i,p,d]*wmax+lr[0]*np.random.rand()*(pbestpop[i,p,d]-pop[i,p,d])+lr[1]*np.random.rand()*(gbestpop[p,d]-pop[i,p,d])
                    if v[i][p][d]>speedrange["max"][p][d]:
                        v[i][p][d]=speedrange["max"][p][d]
                    if v[i][p][d]<speedrange["min"][p][d]:
                        v[i][p][d]=speedrange["min"][p][d]
        pop+=v
        for i in range(sizepop):
            for p in range(numStation-1):
                for t in range(96):
                    if pop[i,p,t]>N_max[p,t]:
                        pop[i,p,t]=N_max[p,t]
                    elif pop[i,p,t]<N_min[p,t]:
                        pop[i,p,t]=N_min[p,t]
            pop[i, -1] = N_all - np.sum(pop[i, :numStation - 1], axis=0)

        #适应度更新
        p=Pool()
        fitness_temp=[None]*sizepop
        for i in range(sizepop):#多进程使用加快计算速度
            #print("适应度计算%d" %i)
            fitness_temp[i]=p.apply_async(get_fitness,args=(pop[i],Z_start_list,Q_c,StationList,N_max,N_min))
        p.close()
        p.join()
        fitness=np.array([fitness_temp[i].get() for i in range(sizepop)])
        for j in range(sizepop):
            if fitness[j]>pbestfitness[j]:
                pbestfitness[j]=fitness[j]
                pbestpop[j]=pop[j].copy()
        if pbestfitness.max()>gbestfitness:
            gbestfitness=pbestfitness.max()
            gbestpop=pbestpop[pbestfitness.argmax()].copy()
        print("全局极值：",gbestfitness)
        print("种群适应度：\n",fitness)
        result[gen]=gbestfitness
        gen+=1

    plt.figure()
    plt.plot(result,color='r')
    plt.show()


    Q_in_list,q_fd_list,q_loss_list,Z_up_list,N_temp_list=GetCascadeProcess(gbestpop,Z_start_list,Q_c,StationList)
    fig, ax = plt.subplots()
    labels = ['chb', 'hjp']
    ax.stackplot(range(len(N_all)), N_temp_list[0], N_temp_list[1], labels=labels)
    ax.legend()
    plt.show()

    result_data = pd.DataFrame()
    for p in range(len(StationList)):
        result_data["Z"+StationList[p].name]=pd.Series(Z_up_list[p])
        result_data["N"+StationList[p].name]=pd.Series(N_temp_list[p])
        result_data["qfd"+StationList[p].name]=pd.Series(q_fd_list[p])
        result_data["qloss"+StationList[p].name]=pd.Series(q_loss_list[p])
        result_data["Qin"+StationList[p].name]=pd.Series(Q_in_list[p])
        result_data["Qout"+StationList[p].name]=pd.Series(q_fd_list[p]+q_loss_list[p])
    result_data.to_excel("res4.xlsx")

