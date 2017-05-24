# -*- coding:utf-8 -*-
import numpy as np
import math

def get_normal_data(N, b_mu, b_sigma, g_mu, g_sigma, b_prob, g_prob):
    H = np.zeros(N)
    b_num = int(N * b_prob)
    g_num = int(N * g_prob)

    for i in range(b_num):
        H[i] = np.random.normal(b_mu, b_sigma)
    for i in range(b_num, N):
        H[i] = np.random.normal(g_mu, g_sigma)

    np.random.shuffle(H)
    return H

def get_fai(x, mu, sigma):
    exponent = math.exp(-(math.pow(x - mu, 2)) / (2 * math.pow(sigma, 2)))
    prob = (1. / (math.sqrt(2 * math.pi) * sigma)) * exponent
    return prob

def cal_mu_avg(N, omega, H):
    result = 0.0
    for i in range(N):
        result += omega[i] * H[i]
    return (result)

def cal_sigma_avg(N, omega, H, mu):
    result = 0.
    for i in range(N):
        result += omega[i] * math.pow(H[i] - mu, 2)
    return (result)

def arrive_to_opt(old, new):
    old = np.matrix(old)
    new = np.matrix(new)
    temp = old - new
    if temp * temp.T < 1e-8:
        return True
    return False

def gmm(H):
    N = len(H)
    b_mu = 178.0
    b_sigma2 = 1.0
    g_mu = 150.0
    g_sigma2 = 1.0
    b_prob = 0.5
    g_prob = 0.5

    b_omega = range(N)
    g_omega = range(N)

    old = [b_mu, b_sigma2, b_prob, g_mu, g_sigma2, g_prob]
    new = []

    iters = 0

    while iters < 10000:
        #E-step
        for i in range(N):
            b_omega[i] = b_prob * get_fai(H[i], b_mu, b_sigma2)
            g_omega[i] = g_prob * get_fai(H[i], g_mu, g_sigma2)
            # print b_omega[i], g_omega[i]
            sum_omega = b_omega[i] + g_omega[i]
            b_omega[i] /= sum_omega
            g_omega[i] /= sum_omega

        #M-step
        b_num = sum(b_omega)
        g_num = sum(g_omega)

        b_prob = (b_num) / (N)
        g_prob = (g_num) / (N)

        b_mu_old = b_mu
        g_mu_old = g_mu
        b_mu = cal_mu_avg(N, b_omega, H) / b_num
        g_mu = cal_mu_avg(N, g_omega, H) / g_num
        b_sigma2 = math.sqrt((cal_sigma_avg(N, b_omega, H, b_mu_old) / b_num))
        g_sigma2 = math.sqrt((cal_sigma_avg(N, g_omega, H, g_mu_old) / g_num))

        new = [b_mu, b_sigma2, b_prob, g_mu, g_sigma2, g_prob]
        if arrive_to_opt(old, new):
            break
        old = new
        iters += 1
        print iters, old
    return new

if __name__ == '__main__':
    # 产生数据
    N = 1000
    b_mu = 170.0
    b_sigma2 = 5
    g_mu = 160.0
    g_sigma2 = 6
    b_prob = 0.8
    g_prob = 0.2
    # 进行分类
    Heights = get_normal_data(N, b_mu, b_sigma2, g_mu, g_sigma2, b_prob, g_prob)
    print Heights
    parmas = gmm(Heights)