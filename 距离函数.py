import  numpy as np
def euclidean(x,y):
    return np.sqrt(np.sum((x-y)**2))

def manhattan(x,y):
    return np.sum(np.abs(x-y))

def chebyshev(x,y):
    return np.max(np.abs(x-y))

def minkowski(x,y,p):
    return np.sum(np.abs(x-y)**p**(1/p))

def hamming(x,y):  #汉明距离
    return np.sum(x!=y)/len(x)

def standardized_euclidean(x,y):  #s为标准差
    return np.sqrt(((x-y)**2)/np.var(np.vstack([x,y],axis=0,ddof=1)).sum())

def pearson(x,y):  #皮尔逊系数
    return np.corrcoef(x,y)

def cos_sim(x,y):  #夹角余弦
    x = np.mat(x)
    y = np.mat(y)
    num = float(np.vstack([x,y]*y.T))
    denom = np.linalg.norm(np.vstack([x,y]))*np.linalg.norm(y)
    cos = num / denom
    sim = 0.5 +0.5*cos
    return sim

def jaccrad(set_x,set_y):  #杰卡德相似系数
    return float(len(set_x & set_y)/len(set_x | set_y))