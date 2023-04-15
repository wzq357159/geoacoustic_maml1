import numpy as np
import matplotlib
import matplotlib.pyplot as plt


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    testy = np.load('testy.npy')
    testy_act = np.load('testy_act.npy')
    y_p = testy[:, :, 500, :]

    y_p1 = y_p[11].flatten()
    y_a1 = testy_act[11].flatten()
    sorted_idx1 = np.argsort(y_a1)
    y_a1 = y_a1[sorted_idx1]
    y_p1 = y_p1[sorted_idx1]

    y_p2 = y_p[0].flatten()
    y_a2 = testy_act[0].flatten()
    sorted_idx2 = np.argsort(y_a2)
    y_a2 = y_a2[sorted_idx2]
    y_p2 = y_p2[sorted_idx2]

    x = np.arange(len(y_a1))
    fig, ax = plt.subplots()

    ax.scatter(x, y_p2, color='green', label='未进行元学习的结果', s=25)
    ax.scatter(x, y_p1, color='blue', label='元学习后的结果', s=25)
    ax.scatter(x, y_a1, color='red', label='真实值', s=10)

    ax.set_xlabel('样本索引',fontsize=12)
    ax.set_ylabel(r'$\alpha_{i}$'+'反演结果',fontsize=12)

    ax.legend()
    plt.savefig('cs_scatter_plot.png')

    plt.show()


