"""Scene loader. Each scene -> dict(data=HxWxL float32 (raw units), gt=HxW bool, name, src)."""
import os, numpy as np, scipy.io as sio
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'dl')

SCENES = {
    'ABU-A3':   ('abu/abu-airport-3.mat', None),
    'ABU-B3':   ('abu/abu-beach-3.mat', None),
    'ABU-U4':   ('abu/abu-urban-4.mat', None),
    'SanDiego': ('htdmamba/Sandiego.mat', None),
    'HYDICE':   ('hydice/HYDICE-urban.mat', None),
    'WHR':      ('whr/extracted', 'envi'),
}
DEV_SCENES = {
    'Segundo':  ('dev/Segundo.mat', None),
    'AVIRIS-1': ('dev/aviris_1.mat', None),
    'AVIRIS-2': ('dev/aviris_2.mat', None),
    'Cri':      ('dev/Cri.mat', None),
    'SanDiego2':('htdmamba/Sandiego2.mat', None),
    'ABU-A2':   ('htdmamba/abu-airport-2.mat', None),
    'ABU-B4':   ('htdmamba/abu-beach-4.mat', None),
}

def _read_envi(path_noext):
    hdr = {}
    with open(path_noext + '.hdr', 'r', errors='ignore') as f:
        txt = f.read()
    import re
    for key in ['samples', 'lines', 'bands', 'data type', 'interleave', 'byte order', 'header offset']:
        m = re.search(r'%s\s*=\s*([^\n{]+)' % key, txt)
        if m: hdr[key] = m.group(1).strip()
    dt = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64, 12: np.uint16}[int(hdr['data type'])]
    ns, nl, nb = int(hdr['samples']), int(hdr['lines']), int(hdr['bands'])
    bo = '>' if hdr.get('byte order', '0').strip() == '1' else '<'
    off = int(hdr.get('header offset', '0'))
    arr = np.fromfile(path_noext + '.img', dtype=np.dtype(dt).newbyteorder(bo), offset=off)
    il = hdr.get('interleave', 'bsq').lower()
    if il == 'bsq':  arr = arr.reshape(nb, nl, ns).transpose(1, 2, 0)
    elif il == 'bil': arr = arr.reshape(nl, nb, ns).transpose(0, 2, 1)
    else:             arr = arr.reshape(nl, ns, nb)
    return arr

def load_scene(name):
    tbl = {**SCENES, **DEV_SCENES}
    rel, kind = tbl[name]
    p = os.path.join(ROOT, rel)
    if kind == 'envi':
        data = _read_envi(os.path.join(p, 'WHU-Hi-River'))
        gt = _read_envi(os.path.join(p, 'Anomaly detection', 'Target_mask'))  # 36 px = Table I (plastic plates + gray panels)
        gt = gt[..., 0] if gt.ndim == 3 else gt
    else:
        d = sio.loadmat(p)
        data = d['data']; gt = d['map']
    data = np.asarray(data, dtype=np.float32)
    gt = np.asarray(gt).squeeze() > 0
    return dict(data=data, gt=gt, name=name, src=rel)

def target_prior(scene):
    """Reference article: 'a consistent prior derived from the mean spectrum of the target region'."""
    X = scene['data']; g = scene['gt']
    return X[g].mean(0)

if __name__ == '__main__':
    import sys
    for n in (sys.argv[1:] or list(SCENES) + list(DEV_SCENES)):
        try:
            s = load_scene(n)
            print(n, s['data'].shape, 'targets', int(s['gt'].sum()), 'range', float(s['data'].min()), float(s['data'].max()))
        except Exception as e:
            print(n, 'ERR', e)
