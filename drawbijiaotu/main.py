import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import pandas as pd
import matplotlib


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams['axes.unicode_minus'] = False

    # 读取CSV文件
    df1 = pd.read_csv('1.csv', header=None)
    df2 = pd.read_csv('2.csv', header=None)
    df3 = pd.read_csv('3.csv', header=None)
    df4 = pd.read_csv('4.csv', header=None)
    df5 = pd.read_csv('5.csv', header=None)


    # 将DataFrame转换成NumPy数组
    data1 = df1.to_numpy()
    data2 = df2.to_numpy()
    data3 = df3.to_numpy()
    data4 = df4.to_numpy()
    data5 = df5.to_numpy()

    # 获取数据的行数和列数
    num_rows, num_cols = data1.shape
    # plt.xscale('log')

    # plt.yscale('log')



    plt.plot(range(1, num_cols + 1), data1[0, :], label='未进行元学习')
    plt.plot(range(1, num_cols + 1), data1[1, :], label='元学习率为5×10$^{-5}$')
    plt.plot(range(1, num_cols + 1), data2[1, :], label='元学习率为1×10$^{-4}$')
    plt.plot(range(1, num_cols + 1), data3[1, :], label='元学习率为5×10$^{-4}$')
    plt.plot(range(1, num_cols + 1), data4[1, :], label='元学习率为1×10$^{-3}$')
    plt.plot(range(1, num_cols + 1), data5[2, :], label='元学习率为2×10$^{-3}$')



    # 添加图例、标题和坐标轴标签
    plt.legend(fontsize=8)

    plt.xlabel('基学习迭代次数', fontsize=14)
    plt.ylabel('MSE损失值', fontsize=14)

    plt.savefig('csbijiaotu.png', dpi=300)

    # 显示图形
    plt.show()
