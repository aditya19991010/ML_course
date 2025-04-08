import numpy as np

#l1 norm
def l1_norm(feat,lambda_l1):
    return lambda_l1*sum(abs(x) for x in feat)

#l2 norm
def l2_norm(feat,lambda_l2):
    return lambda_l2*np.sqrt(sum(x**2 for x in feat))