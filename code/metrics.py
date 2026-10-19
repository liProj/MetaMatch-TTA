"""AUC metrics for hyperspectral target detection, following the reference article's Table II:
AUC1 = AUC(FPR,PD)  (ROC area, higher better)
AUC2 = AUC(tau,FPR) (area under FPR vs threshold, lower better)
AUC3 = AUC(tau,PD)  (area under PD vs threshold, higher better)
AUC4 = AUC_OAD = AUC1 + AUC3 - AUC2
AUC5 = AUC_SNPR = AUC3 / AUC2
Detection maps are min-max normalised to [0,1] before the threshold-based AUCs (standard practice, e.g. Chang 2021 [54]).
"""
import numpy as np

def _norm01(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)

def roc_3d(score, gt, n_thr=None):
    """Return thresholds tau (descending from 1 to 0), PD(tau), FPR(tau).
    If n_thr is None every unique score is used (exact). Otherwise a uniform grid."""
    s = _norm01(score); g = np.asarray(gt).ravel().astype(bool)
    if n_thr is None:
        tau = np.unique(np.concatenate([[0.0, 1.0], s]))[::-1]
    else:
        tau = np.linspace(1.0, 0.0, n_thr)
    # PD(tau) = P(s >= tau | target); FPR(tau) = P(s >= tau | background)
    st = np.sort(s[g]); sb = np.sort(s[~g])
    pd = 1.0 - np.searchsorted(st, tau, side='left') / max(len(st), 1)
    fpr = 1.0 - np.searchsorted(sb, tau, side='left') / max(len(sb), 1)
    return tau, pd, fpr

def _trapz(y, x):
    return float(np.abs(np.trapezoid(y, x)))

def auc_metrics(score, gt, n_thr=None):
    tau, pd, fpr = roc_3d(score, gt, n_thr)
    auc1 = _trapz(pd, fpr)            # (FPR, PD) ROC area
    auc2 = _trapz(fpr, tau)           # (tau, FPR)
    auc3 = _trapz(pd, tau)            # (tau, PD)
    auc4 = auc1 + auc3 - auc2
    auc5 = auc3 / max(auc2, 1e-12)
    return dict(AUC1=auc1, AUC2=auc2, AUC3=auc3, AUC4=auc4, AUC5=auc5)

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    gt = np.zeros(10000, bool); gt[:50] = True
    s = rng.normal(0, 1, 10000); s[:50] += 4
    print(auc_metrics(s, gt))
