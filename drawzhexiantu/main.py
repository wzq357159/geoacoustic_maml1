import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import pandas as pd
import matplotlib


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 读取CSV文件
    df = pd.read_csv('testdata.csv', header=None)

    # 将DataFrame转换成NumPy数组
    data = df.to_numpy()
    # 获取数据的行数和列数
    num_rows, num_cols = data.shape
    # plt.xscale('log')

    # plt.yscale('log')


    # 循环遍历每一行，画出折线图
    # for i in range(num_rows):
    #     if i in [0,1,5,10,20,40]:
    #         plt.plot(range(1, num_cols + 1), data[i, :], label='Line {}'.format(i + 1))
    plt.plot(range(1, num_cols + 1), data[0, :], label='未进行元学习')
    plt.plot(range(1, num_cols + 1), data[1, :], label='元学习迭代1次')
    plt.plot(range(1, num_cols + 1), data[2, :], label='元学习迭代39次')
    # plt.plot(range(1, num_cols + 1), data[6, :], label='元学习迭代234次',color = 'red',linewidth=2)
    # plt.plot(range(1, num_cols + 1), data[11, :], label='元学习迭代390次',color = 'cyan',linewidth=1)

    plt.plot(range(1, num_cols + 1), data[6, :], label='元学习迭代234次', color='red')
    plt.plot(range(1, num_cols + 1), data[11, :], label='元学习迭代390次', color='cyan')


    # 添加图例、标题和坐标轴标签
    plt.legend(fontsize=8)

    plt.xlabel('基学习迭代次数', fontsize=14)
    plt.ylabel('MSE损失值', fontsize=14)

    plt.savefig('testdata.png', dpi=300)

    # 显示图形
    plt.show()












