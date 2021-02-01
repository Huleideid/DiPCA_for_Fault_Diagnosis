# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.stats
from sklearn.model_selection import KFold

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 归一化数据
def autos(X):
    m = X.shape[0]
    n = X.shape[1]
    X_m = np.zeros((m, n))
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    for i in range(n):
        a = np.ones(m) * mu[i]
        X_m[:, i] = (X[:, i]-a) / sigma[i]
    return X_m, mu, sigma

def autos_test(data,m_train,v_train):
    m = data.shape[0]
    n = data.shape[1]
    data_new = np.zeros((m, n))
    for i in range(n):
        a = np.ones(m) * m_train[i]
        data_new[:, i] = (data[:, i] - a) / v_train[i]
    return data_new

def pc_number(X):
    U, S, V = np.linalg.svd(X)
    if S.shape[0] == 1:
        i = 1
    else:
        i = 0
        var = 0
        while var < 0.85*sum(S*S):
            var = var+S[i]*S[i]
            i = i + 1
    return i

def DiPCA(X, s, a):
    n = X.shape[0]
    m = X.shape[1]
    N = n - s
    Xe = X[s:N+s, :]
    alpha = 0.01
    level = 1-alpha
    P = np.zeros((m, a))
    W = np.zeros((m, a))
    T = np.zeros((n, a))
    w = np.ones(m)
    w = w / np.linalg.norm(w, ord=2)

    if s > 0:
        l = 0
        while l < a:
            iterr = 1000
            temp = np.dot(X, w)
            while iterr > 0.00001:
                t = np.dot(X, w)
                beta = np.ones((s))
                for i in range(s):
                    beta[i] = np.dot(t[i:N+i-1].T, t[s:N+s-1])
                beta = beta / np.linalg.norm(beta, ord=2)
                w = np.zeros(m)

                for i in range(s):
                    w = w + beta[i]*(np.dot(X[s:N+s-1, :].T, t[i:N+i-1]) +
                                     np.dot(X[i:N+i-1].T, t[s:N+s-1]))
                w = w / np.linalg.norm(w, ord=2)
                t = np.dot(X, w)
                iterr = np.linalg.norm((t-temp), ord=2)

                temp = t
            p = np.dot(X.T, t)/np.dot(t.T, t)
            p = X.T@ t/(t.T@t)

            t = np.array([t]).T

            p = np.array([p]).T
            X = X - np.dot(t, p.T)
            P[:, l] = p[:, 0]
            W[:, l] = w
            T[:, l] = t[:, 0]
            l = l+1

        # Dynamic Inner Modeling
        TT = T[0:N, :]
        j = 1
        while j < s:
            TT = np.c_[TT, T[j:(N+j), :]]
            j = j+1
        Theta = np.dot(np.dot(np.linalg.inv(np.dot(TT.T, TT)), TT.T), T[s:N+s, :])

        V = T[s:N+s, :] - np.dot(TT, Theta)
        a_v = pc_number(V)
        _, Sv, Pv = np.linalg.svd(V)
        Pv = Pv.T
        Pv = Pv[:, 0:a_v]
        lambda_v = 1/(N-1)*np.diag(Sv[0:a_v]**2)
        if a_v!=a: # 注意是否T^2和Q都存在
            gv = 1/(N-1)*sum(Sv[a_v:a]**4)/sum(Sv[a_v:a]**2)
            hv = (sum(Sv[a_v:a]**2)**2)/sum(Sv[a_v:a]**4)
            Tv2_lim = a_v * (N ** 2 - 1) / (N * (N - a_v))* scipy.stats.f.ppf(level, a_v, N-a_v)
            Qv_lim = gv*scipy.stats.chi2.ppf(level, hv)
            PHI_v = np.dot(np.dot(Pv, np.linalg.inv(lambda_v)), Pv.T)/Tv2_lim + (np.identity(len(Pv@Pv.T))-Pv@Pv.T)/Qv_lim;
            SS_v=1/(N-1)*V.T@V
            g_phi_v=np.trace((SS_v@PHI_v)@(SS_v@PHI_v))/(np.trace(SS_v@PHI_v))
            h_phi_v=(np.trace(SS_v@SS_v)**2)/np.trace((SS_v@PHI_v)@(SS_v@PHI_v))
            phi_v_lim = g_phi_v*scipy.stats.chi2.ppf(level, h_phi_v)
        else:
            Tv2_lim = a_v * (N ** 2 - 1) / (N * (N - a_v))* scipy.stats.f.ppf(level, a_v, N-a_v)
            PHI_v = np.dot(np.dot(Pv, np.linalg.inv(lambda_v)), Pv.T)
            phi_v_lim=Tv2_lim
        Xe = Xe-np.dot(np.dot(TT, Theta), P.T)
    a_s = pc_number(Xe)
    _, Ss, Ps = np.linalg.svd(Xe)
    Ps = Ps.T
    Ps = Ps[:,0:a_s]
    Ts = np.dot(Xe, Ps)
    lambda_s = 1 / (N - 1) * np.diag(Ss[0:a_s] ** 2)
    m = Ss.shape[0]
    gs = 1 / (N - 1) * sum(Ss[a_s:m] ** 4) / sum(Ss[a_s:m] ** 2)
    hs = (sum(Ss[a_s:m] ** 2) ** 2) / sum(Ss[a_s:m] ** 4)

    Ts2_lim = scipy.stats.chi2.ppf(level,a_s)
    Qs_lim = gs*scipy.stats.chi2.ppf(level,hs)
    return P,W,Theta,Ps,lambda_s,PHI_v,phi_v_lim,Ts2_lim ,Qs_lim

def DiPCA_test(X,P,W,Theta,Ps,s,lambda_s,PHI_v):
    """
    DiPCA测试 for 监控
    """
    n = X.shape[0]
    N = n - s
    a = P.shape[1]
    Mst = np.dot(np.dot(Ps, np.linalg.inv(lambda_s)), Ps.T)
    Msq = np.eye((Mst.shape[0])) - np.dot(Ps, Ps.T)
    R = np.dot(W, np.linalg.inv(np.dot(P.T, W)))
    if s > 0:
        T = np.dot(X, R)
        TTs = T[s:N+s, :]
        Ts = np.zeros((N, s))
        TT = T[0:N, :]
        i = 1
        while i < s:
            Ts = T[i:N+i, :]
            TT = np.c_[TT, Ts]
            i = i + 1
        TTshat = np.dot(TT, Theta)

    phi_v_index = np.zeros(N)
    Ts_index = np.zeros(N)
    Qs_index = np.zeros(N)
    k = s
    while k < s+N:
        if s > 0:
            temp = TTs[k-s, :] - TTshat[k-s, :]
            temp = np.array([temp])
            v = temp.T
            phi_v_index[k-s] = np.dot(np.dot(v.T, PHI_v), v)
            e = X[k-s, :].T - np.dot(P, TTshat[k-s, :].T)
        else:
            e = X[k-s, :].T
        Ts_index[k-s] = np.dot(np.dot(e.T, Mst), e)
        Qs_index[k-s] = np.dot(np.dot(e.T, Msq), e)
        k = k+1

    return phi_v_index,Ts_index,Qs_index


def DiPCA_predict(X,P,W,Theta,s):
    """
    DiPCA预测
    """
    n = X.shape[0]
    N = n - s
    a = P.shape[1]
    x_predict_d=np.zeros(X.shape, dtype=float)

    R = np.dot(W, np.linalg.inv(np.dot(P.T, W)))
    if s > 0:
        T = np.dot(X, R)
        TT = T[0:N, :]
        i = 1
        while i < s:
            Ts = T[i:N+i, :]
            TT = np.c_[TT, Ts]
            i = i + 1
        TTshat = np.dot(TT, Theta)
        x_predict_d[s:,:] =TTshat@P.T;
    return x_predict_d



def DiPCA_cv(X,s_range,a_range,fold):
    """
    DiPCA交叉验证选取主元数
    输入：X 训练数据,s_range 选择滞后阶数最大值,a_range 选择潜变量最大值, fold 交叉验证的折数,
    """
    kf = KFold(n_splits=fold,random_state=1,shuffle=False)
    press=np.zeros((s_range,a_range,fold), float)
    for i in range(s_range):
        for j in range(a_range):
            count=0
            for train_index, valid_index in kf.split(X):
                count+=1
                X_train, X_valid = X[train_index], X[valid_index]
                P,W,Theta,Ps,lambda_s,PHI_v,phi_v_lim,Ts2_lim ,Qs_lim = DiPCA(X_train, i+1, j+1);#建模
                X_predict=DiPCA_predict(X_valid,P,W,Theta, i+1)#预测
                press[i][j][count-1]=np.linalg.norm(X_valid-X_predict,ord=2)**2/X_valid.shape[0]
    press=np.sum(press, axis=2)
    (s,a)=np.where(press==np.min(press))#选择press最小作为s,a
    s+=1
    a+=1
    return int(s),int(a)

"""
    DiPCA可视化
    目前主要是三个监控指标，包括动态综合指标,静态T2指标和静态指标
    参数
    ----------
"""
def DiPCA_visualization(phi_v_index,Ts_index,Qs_index,phi_v_lim,Ts2_lim,Qs_lim):
    plt.figure(figsize=(9.6,6.4),dpi=600)
    ax1 = plt.subplot(3,1,1)
    ax1.plot(phi_v_index)
    ax1.plot(phi_v_lim*np.ones(len(phi_v_index)),'r--')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('$\phi_v$')
    ax1.set_title('monitor')
    ax2 = plt.subplot(3,1,2)
    ax2.plot(Ts_index)
    ax2.plot(Ts2_lim*np.ones(len(phi_v_index)),'r--')
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('$T^2_s$')
    ax3 = plt.subplot(3,1,3)
    ax3.plot(Qs_index)
    ax3.plot(Qs_lim*np.ones(len(phi_v_index)),'r--')
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('$Q_s$')
    plt.show()


x_train= loadmat("./data/d00.mat")['d00']
x_test = loadmat("./data/d05te.mat")['d05te']
x_train=np.array(x_train)
x_test=np.array(x_test)
X_train, X_mean, X_s = autos(x_train)
X_test = autos_test(x_test,X_mean,X_s)
s_range=2
a_range=5
fold=5
[s,a]=DiPCA_cv(X_train,s_range,a_range,fold)#交叉验证选取主元数
P,W,Theta,Ps,lambda_s,PHI_v,phi_v_lim,Ts2_lim ,Qs_lim = DiPCA(X_train, s, a);#建模
phi_v_index,Ts_index,Qs_index = DiPCA_test(X_test, P, W, Theta, Ps, s, lambda_s, PHI_v);# 测试
DiPCA_visualization(phi_v_index,Ts_index,Qs_index,phi_v_lim,Ts2_lim,Qs_lim);# 监测结果可视化
