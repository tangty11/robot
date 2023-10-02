import numpy as np


def G(x, y):
    sigma = 1.5
    res = 1/(2*np.pi*sigma**2) * np.exp(-(x**2+y**2)/(2*sigma**2))
    return res

def gaosihe(s):
    ori = [[s[0], s[1], s[2]], [s[3], s[4], s[5]], [s[6], s[7], s[8]]]
    fin = []
    for i in range(3):
        for j in range(3):
            fin.append(G(i-1, j-1))

    sum_fin = sum(fin)
    for k in range(9):
        fin[k] = fin[k] / sum_fin

    ii = 0
    result = 0
    for i in range(3):
        for j in range(3):
            result += ori[i][j] * fin[ii]
            ii += 1

    return result

if __name__ == '__main__':
    s = []
    for k in range(9):
            s.append(int(input('')))

    print(gaosihe(s))
