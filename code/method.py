"""New method: a scene-agnostic, meta-trained pair matcher with physics-informed whitening and
test-time self-distillation (TTSD). Trained ONCE episodically on the development scenes with
self-supervised material episodes; applied to any scene / band count / prior protocol.
Components (each ablatable):
  W  : background whitening of pixel and prior (scene covariance, shrinkage)      -> physics-informed input
  M  : band-agnostic 1-D conv pair encoder + attention pooling (the meta-matcher)
  A  : physics-based spectral variability augmentation in the episodes
  R  : test-time prior refinement (transductive)
  D  : test-time self-distillation into a scene-specific student
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, math, time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------- scene preprocessing ----------------
def whitener(X, shrink=0.05):
    mu = X.mean(0)
    Xc = X - mu
    K = Xc.T @ Xc / len(X)
    K = (1 - shrink) * K + shrink * np.trace(K) / K.shape[0] * np.eye(K.shape[0])
    w, V = np.linalg.eigh(K)
    Wm = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-10))) @ V.T
    return mu.astype(np.float32), Wm.astype(np.float32)

def prep_scene(scene, n_clusters=40, seed=0):
    X = scene['data'].reshape(-1, scene['data'].shape[-1]).astype(np.float32)
    X = X / X.max()
    mu, Wm = whitener(X)
    Xw = (X - mu) @ Wm
    Z = PCA(min(30, X.shape[1]), random_state=seed).fit_transform(X)
    lab = KMeans(n_clusters, n_init=3, random_state=seed).fit_predict(Z)
    return dict(name=scene['name'], X=X, Xw=Xw.astype(np.float32), mu=mu, Wm=Wm, lab=lab, gt=scene['gt'].ravel(),
                H=scene['data'].shape[0], W=scene['data'].shape[1], sd=X.std(0).mean())

# ---------------- physics-based variability augmentation ----------------
def smooth_mult(L, rng, scale, width=15, n=1):
    from scipy.ndimage import gaussian_filter1d
    e = gaussian_filter1d(rng.standard_normal((n, L + 2 * width)), width / 2.5, axis=1, mode='reflect')[:, width:-width]
    return 1 + scale * e / (e.std(1, keepdims=True) + 1e-12)

def augment(x, bg_pool, rng, sd, strength=1.0):
    """x: n x L spectra (normalised). Illumination gain, smooth multiplicative distortion,
    sub-pixel linear mixing with background, additive noise."""
    n, L = x.shape
    out = x.copy()
    gain = rng.uniform(1 - 0.3 * strength, 1 + 0.3 * strength, (n, 1))
    out = out * gain
    sm = smooth_mult(L, rng, 0.08 * strength, n=n)
    out = out * sm
    mix = rng.random(n) < 0.5
    if mix.any():
        a = rng.uniform(1 - 0.5 * strength, 1.0, (mix.sum(), 1))
        b = bg_pool[rng.integers(0, len(bg_pool), mix.sum())]
        out[mix] = a * out[mix] + (1 - a) * b
    out = out + rng.standard_normal(out.shape) * 0.02 * sd * strength
    return out.astype(np.float32)

# ---------------- matcher network ----------------
class ResBlock(nn.Module):
    def __init__(self, c, k, d):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, k, padding=d * (k // 2), dilation=d); self.c2 = nn.Conv1d(c, c, k, padding=d * (k // 2), dilation=d)
        self.n1 = nn.GroupNorm(8, c); self.n2 = nn.GroupNorm(8, c)
    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x))); h = self.n2(self.c2(h))
        return F.gelu(x + h)

class PairMatcher(nn.Module):
    """Input: pixel x, prior d (raw normalised) and their whitened versions. Any band count L."""
    def __init__(self, c=64, use_white=True, use_raw=True, n_scalar=4):
        super().__init__()
        self.use_white = use_white; self.use_raw = use_raw
        nin = 1 + (4 if use_raw else 0) + (4 if use_white else 0)
        self.stem = nn.Conv1d(nin, c, 7, padding=3)
        self.blocks = nn.Sequential(ResBlock(c, 5, 1), ResBlock(c, 5, 2), ResBlock(c, 3, 4), ResBlock(c, 3, 8))
        self.q = nn.Parameter(torch.randn(1, c, 1) * 0.02)
        self.scal = nn.Sequential(nn.Linear(n_scalar, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(3 * c + 32, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def pair_feats(self, x, d, xw, dw):
        L = x.shape[-1]
        pos = torch.linspace(0, 1, L, device=x.device)[None, None].expand(len(x), 1, L)
        ch = [pos]
        if self.use_raw: ch += [x[:, None], d[:, None], (x * d)[:, None], (x - d)[:, None]]
        if self.use_white: ch += [xw[:, None], dw[:, None], (xw * dw)[:, None], (xw - dw)[:, None]]
        return torch.cat(ch, 1)
    def forward(self, x, d, xw, dw, scal):
        f = self.pair_feats(x, d, xw, dw)
        h = self.blocks(F.gelu(self.stem(f)))                    # B x c x L
        att = torch.softmax((h * self.q).sum(1, keepdim=True) / math.sqrt(h.shape[1]), -1)
        pooled = torch.cat([(h * att).sum(-1), h.mean(-1), h.amax(-1), self.scal(scal)], 1)
        return self.head(pooled).squeeze(1)

def scalar_stats(x, d, xw, dw):
    """Classical statistics as scalar side inputs: whitened cosine (ACE), AMF-like projection, raw cosine, whitened norm ratio."""
    eps = 1e-8
    nxw = xw.norm(dim=-1) + eps; ndw = dw.norm(dim=-1) + eps
    ace = (xw * dw).sum(-1) / (nxw * ndw)
    amf = (xw * dw).sum(-1) / (ndw ** 2)
    cos = (x * d).sum(-1) / ((x.norm(dim=-1) + eps) * (d.norm(dim=-1) + eps))
    rat = torch.log(nxw / ndw)
    return torch.stack([ace, torch.clamp(amf, -3, 3), cos, rat], 1)

# ---------------- episodic meta-training ----------------
class EpisodeSampler:
    def __init__(self, scenes, rng, aug=True, p_label=0.2, strength=1.0):
        self.S = scenes; self.rng = rng; self.aug = aug; self.p_label = p_label; self.strength = strength
        for s in self.S:
            cnt = np.bincount(s['lab']); s['clusters'] = [np.where(s['lab'] == c)[0] for c in range(len(cnt)) if cnt[c] >= 30]
            cen = np.stack([s['Xw'][idx].mean(0) for idx in s['clusters']])
            dist = ((cen[:, None] - cen[None]) ** 2).sum(-1); np.fill_diagonal(dist, np.inf)
            s['near'] = np.argsort(dist, 1)[:, :3]
            s['tgt'] = np.where(s['gt'])[0]
    def sample(self, n_pos=64, n_neg=192):
        rng = self.rng
        s = self.S[rng.integers(len(self.S))]
        use_label = (rng.random() < self.p_label) and len(s['tgt']) >= 10
        if use_label:
            members = s['tgt']; others = np.where(~s['gt'])[0]; hard = None
        else:
            ci = rng.integers(len(s['clusters'])); members = s['clusters'][ci]
            mask = s['lab'] != s['lab'][members[0]]; others = np.where(mask)[0]
            hard = np.concatenate([s['clusters'][j] for j in s['near'][ci]])
        X = s['X']
        # prior construction
        r = rng.random()
        if r < 0.5: pr = X[members[rng.integers(len(members))]]
        elif r < 0.8: pr = X[rng.choice(members, min(len(members), rng.integers(2, 6)), replace=False)].mean(0)
        else: pr = X[members].mean(0)
        pos = X[rng.choice(members, n_pos, replace=len(members) < n_pos)]
        n_hard = n_neg // 4 if hard is not None else 0
        neg = X[rng.choice(others, n_neg - n_hard, replace=False)]
        if n_hard: neg = np.concatenate([neg, X[rng.choice(hard, n_hard, replace=len(hard) < n_hard)]])
        if self.aug:
            bgpool = X[rng.choice(others, 256, replace=False)]
            if rng.random() < 0.5: pr = augment(pr[None], bgpool, rng, s['sd'], self.strength)[0]
            sel = rng.random(n_pos) < 0.6
            if sel.any(): pos[sel] = augment(pos[sel], bgpool, rng, s['sd'], self.strength)
            seln = rng.random(len(neg)) < 0.3
            if seln.any(): neg[seln] = augment(neg[seln], bgpool, rng, s['sd'], self.strength)
        xs = np.concatenate([pos, neg]); y = np.concatenate([np.ones(n_pos), np.zeros(len(neg))]).astype(np.float32)
        xw = (xs - s['mu']) @ s['Wm']; dw = (pr - s['mu']) @ s['Wm']
        return xs.astype(np.float32), np.repeat(pr[None], len(xs), 0).astype(np.float32), xw.astype(np.float32), np.repeat(dw[None], len(xs), 0).astype(np.float32), y

def meta_train(scenes, steps=2000, seed=0, device='cuda', log=print, aug=True, p_label=0.2, strength=1.0, lr=1e-3, episodes=4, **netkw):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = PairMatcher(**netkw).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    samp = EpisodeSampler(scenes, rng, aug, p_label, strength)
    t0 = time.time(); run = 0
    for it in range(steps):
        # a batch = 2 episodes (possibly different L) -> process separately, sum losses
        loss = 0
        for _ in range(episodes):
            xs, pr, xw, dw, y = samp.sample()
            xs, pr, xw, dw, y = [torch.tensor(a, device=device) for a in (xs, pr, xw, dw, y)]
            logit = net(xs, pr, xw, dw, scalar_stats(xs, pr, xw, dw))
            loss = loss + F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(2.0, device=device))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        if it % 500 == 499: log(f'  step {it+1} loss {run:.4f} {time.time()-t0:.0f}s')
    return net

# ---------------- inference + test-time self-distillation ----------------
@torch.no_grad()
def score_scene(net, P, prior_raw, device='cuda', bs=8192):
    net.eval()
    X = P['X']; pr = (prior_raw / 1.0).astype(np.float32)
    dw = (pr - P['mu']) @ P['Wm']
    out = []
    for i in range(0, len(X), bs):
        xs = torch.tensor(X[i:i + bs], device=device); xw = torch.tensor(P['Xw'][i:i + bs], device=device)
        prt = torch.tensor(pr, device=device)[None].expand(len(xs), -1); dwt = torch.tensor(dw, device=device)[None].expand(len(xs), -1)
        out.append(torch.sigmoid(net(xs, prt, xw, dwt, scalar_stats(xs, prt, xw, dwt))).cpu())
    return torch.cat(out).numpy()

def refine_prior(P, score, prior, frac=0.002, min_n=5, max_n=200):
    n = int(np.clip(frac * len(score), min_n, max_n))
    top = np.argsort(-score)[:n]
    return P['X'][top].mean(0), top

class Student(nn.Module):
    """Scene-specific student on whitened spectra (band count fixed per scene)."""
    def __init__(self, L, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * L, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, xw, dw): return self.net(torch.cat([xw, xw * dw], -1)).squeeze(1)

def distill(P, teacher_score, prior, device='cuda', epochs=40, seed=0, log=None):
    torch.manual_seed(seed)
    L = P['X'].shape[1]
    st = Student(L).to(device); opt = torch.optim.Adam(st.parameters(), 1e-3)
    xw = torch.tensor(P['Xw'], device=device); dw = torch.tensor(((prior - P['mu']) @ P['Wm']).astype(np.float32), device=device)[None].expand(len(xw), -1)
    t = torch.tensor(teacher_score, device=device, dtype=torch.float32)
    # weight confident pixels more; soft targets (knowledge distillation)
    w = 0.2 + 0.8 * torch.abs(t - 0.5) * 2
    n = len(xw); bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = (w[idx] * F.binary_cross_entropy_with_logits(st(xw[idx], dw[idx]), t[idx], reduction='none')).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    st.eval()
    with torch.no_grad():
        return torch.sigmoid(st(xw, dw)).cpu().numpy()

def detect_full(net, P, prior_raw, use_refine=True, use_distill=True, rounds=2, device='cuda', seed=0):
    """Returns dict of scores for each stage."""
    out = {}
    s = score_scene(net, P, prior_raw, device); out['zero'] = s
    pr = prior_raw
    if use_refine:
        for r in range(rounds):
            pr2, top = refine_prior(P, s, pr)
            s2 = score_scene(net, P, pr2, device)
            s = np.maximum(s, s2); pr = pr2
        out['refined'] = s
    if use_distill:
        st = distill(P, s, pr, device, seed=seed)
        out['student'] = st
        out['final'] = 0.5 * (s + st)
    else:
        out['final'] = s
    # fixed monotone sharpening (AUC1-invariant; analogous to the reference article's eq. 37 contrast transform)
    out['sharp'] = out['final'] ** 4
    # classical detectors driven by the refined prior, and a rank ensemble with the refined matcher score
    from classic import amf, ace
    from scipy.stats import rankdata
    X64 = P['X'].astype(np.float64); pr64 = np.asarray(pr, np.float64)
    out['amf_ref'] = amf(X64, pr64); out['ace_ref'] = ace(X64, pr64)
    base = out['refined'] if use_refine else out['zero']
    out['ens'] = 0.5 * (rankdata(base) + rankdata(out['ace_ref'])) / len(base)
    out['ens_amf'] = 0.5 * (rankdata(base) + rankdata(out['amf_ref'])) / len(base)
    return out
