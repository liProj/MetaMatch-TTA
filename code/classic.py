"""Classic detectors: CEM, AMF (adaptive matched filter), ACE, MF, SAM. Input X: N x L, prior d: L."""
import numpy as np

def _cov(X, mu=None):
    if mu is None: mu = X.mean(0)
    Xc = X - mu
    return Xc.T @ Xc / len(X), mu

def cem(X, d, eps=1e-6):
    R = X.T @ X / len(X)
    R += eps * np.trace(R) / R.shape[0] * np.eye(R.shape[0])
    Ri = np.linalg.inv(R)
    w = Ri @ d / (d @ Ri @ d)
    return X @ w

def amf(X, d, eps=1e-6):
    K, mu = _cov(X)
    K += eps * np.trace(K) / K.shape[0] * np.eye(K.shape[0])
    Ki = np.linalg.inv(K)
    dc = d - mu; Xc = X - mu
    num = Xc @ (Ki @ dc)
    return num ** 2 / (dc @ Ki @ dc)

def ace(X, d, eps=1e-6):
    K, mu = _cov(X)
    K += eps * np.trace(K) / K.shape[0] * np.eye(K.shape[0])
    Ki = np.linalg.inv(K)
    dc = d - mu; Xc = X - mu
    num = Xc @ (Ki @ dc)
    den = np.einsum('ij,jk,ik->i', Xc, Ki, Xc)
    return num ** 2 / (dc @ Ki @ dc) / (den + 1e-12)

def mf(X, d, eps=1e-6):
    K, mu = _cov(X)
    K += eps * np.trace(K) / K.shape[0] * np.eye(K.shape[0])
    Ki = np.linalg.inv(K)
    dc = d - mu; Xc = X - mu
    return Xc @ (Ki @ dc) / (dc @ Ki @ dc)

def sam(X, d):
    cos = X @ d / (np.linalg.norm(X, axis=1) * np.linalg.norm(d) + 1e-12)
    return -np.arccos(np.clip(cos, -1, 1))

DETECTORS = dict(CEM=cem, AMF=amf, ACE=ace, MF=mf, SAM=sam)
