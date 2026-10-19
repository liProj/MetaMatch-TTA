"""CPU release checks; no dataset download or full experiment training."""
from pathlib import Path
import ast
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from method import PairMatcher, scalar_stats, whitener, detect_full
from metrics import auc_metrics

torch.set_num_threads(2)
torch.manual_seed(0)
for name in ['data', 'classic', 'method', 'metrics', 'protocols', 'run_meta']:
    ast.parse((ROOT / 'code' / (name + '.py')).read_text(encoding='utf-8'))
net = PairMatcher().eval()
assert sum(p.numel() for p in net.parameters()) == 165857
with torch.no_grad():
    for bands in [46, 102, 175, 188, 189, 205, 224]:
        x, d, xw, dw = [torch.randn(3, bands) for _ in range(4)]
        output = net(x, d, xw, dw, scalar_stats(x, d, xw, dw))
        assert output.shape == (3,) and torch.isfinite(output).all()
for folder in ['final', 'meta']:
    for checkpoint in sorted((ROOT / 'results' / folder).glob('*.pt')):
        model = PairMatcher(use_white='nowhite' not in checkpoint.name)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        assert all(torch.isfinite(value).all() for value in state.values())

rng = np.random.default_rng(0)
X = rng.random((48, 12), dtype=np.float32)
mu, Wm = whitener(X)
prepared = {'X': X, 'Xw': (X - mu) @ Wm, 'mu': mu, 'Wm': Wm}
before = {name: value.clone() for name, value in net.state_dict().items()}
outputs = detect_full(net, prepared, X[0], use_refine=True, use_distill=False, device='cpu')
assert {'zero', 'refined', 'final'} <= outputs.keys()
for value in outputs.values():
    assert value.shape == (48,) and np.isfinite(value).all()
assert all(torch.equal(value, before[name]) for name, value in net.state_dict().items())
assert np.array_equal(outputs['final'], outputs['refined'])
assert np.all(outputs['refined'] >= outputs['zero'])
perfect = auc_metrics(np.array([0., .1, .9, 1.]), np.array([0, 0, 1, 1]))
assert abs(perfect['AUC1'] - 1.0) < 1e-12
print('PASS: syntax, 165857 parameters, seven band counts, all included checkpoints,')
print('      finite CPU inference, fixed weights during reference updates, and ROC metric.')
