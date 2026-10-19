"""Prior-construction protocols for HTD evaluation.
P0 'mean'    : mean of ALL ground-truth target pixels (reference article; label-leaking, saturated)
P1 'pix1'    : one ground-truth pixel drawn at random (realistic 'one reference spectrum'); repeated draws
P2 'pixk'    : mean of k random ground-truth pixels (k=3,5)
P3 'cross'   : prior = mean target spectrum of ANOTHER scene of the same sensor/band layout (library-style prior)
P4 'perturb' : P0 prior under illumination gain g~U(0.7,1.3), smooth multiplicative distortion and noise (spectral variability stress)
All priors are returned in raw scene units."""
import numpy as np
from data import load_scene, target_prior

CROSS = {  # scene -> donor scene (same band count / sensor family)
    'SanDiego': 'SanDiego2', 'AVIRIS-2': 'SanDiego2', 'SanDiego2': 'AVIRIS-2',
    'ABU-A3': 'ABU-A2', 'ABU-A2': 'ABU-A3', 'ABU-U4': 'ABU-A3',
}

def smooth_noise(L, rng, scale=0.1, width=15):
    e = rng.standard_normal(L + 2 * width)
    k = np.exp(-0.5 * (np.arange(-width, width + 1) / (width / 2.5)) ** 2); k /= k.sum()
    e = np.convolve(e, k, mode='same')[width:-width]
    return 1 + scale * e / (e.std() + 1e-12)

def make_prior(scene, protocol, draw=0, k=3):
    rng = np.random.default_rng(1000 + draw)
    X = scene['data']; g = scene['gt']; T = X[g]
    if protocol == 'mean':
        return T.mean(0)
    if protocol == 'pix1':
        return T[rng.integers(len(T))]
    if protocol.startswith('pixk'):
        kk = int(protocol[4:]) if len(protocol) > 4 else k
        return T[rng.choice(len(T), min(kk, len(T)), replace=False)].mean(0)
    if protocol == 'cross':
        donor = load_scene(CROSS[scene['name']])
        d = target_prior(donor)
        # match overall brightness to this scene's max (sensor units differ by crop)
        return d / donor['data'].max() * X.max()
    if protocol == 'perturb':
        p = T.mean(0)
        gain = rng.uniform(0.7, 1.3)
        p = p * gain * smooth_noise(len(p), rng, 0.08)
        p = p + rng.standard_normal(len(p)) * 0.02 * p.std()
        return p
    raise ValueError(protocol)

PROTOCOLS = ['mean', 'pix1', 'pixk3', 'pixk5', 'perturb', 'cross']
