"""Meta-train the pair matcher and evaluate under the prior protocols.
--mode lodo : leave-one-dev-scene-out on the development pool (method design only)
--mode final: train on ALL dev scenes, test the six target scenes (run once with frozen config)
--mode self : train on dev, test on dev in-sample (sanity)"""
import argparse, json, os, time, numpy as np, torch, pickle
from data import load_scene, SCENES, DEV_SCENES
from metrics import auc_metrics
from protocols import make_prior, CROSS
from method import prep_scene, meta_train, detect_full

DEV = ['Segundo', 'AVIRIS-2', 'Cri', 'SanDiego2', 'ABU-A2', 'ABU-B4']
FOLDS = {'A': ['Segundo', 'Cri', 'ABU-B4'], 'B': ['AVIRIS-2', 'SanDiego2', 'ABU-A2']}  # 2-fold cross-scene dev protocol
ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='lodo'); ap.add_argument('--tag', default='meta')
ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--seeds', default='0')
ap.add_argument('--protocols', default='mean,pix1,pixk3,pixk5,perturb,cross')
ap.add_argument('--draws', type=int, default=3)
ap.add_argument('--load', default=None, help='state_dict to evaluate instead of training (final/self modes)')
ap.add_argument('--no_aug', action='store_true'); ap.add_argument('--no_white', action='store_true'); ap.add_argument('--no_raw', action='store_true')
ap.add_argument('--no_refine', action='store_true'); ap.add_argument('--no_distill', action='store_true')
ap.add_argument('--p_label', type=float, default=0.2); ap.add_argument('--strength', type=float, default=1.0)
ap.add_argument('--out', default='../results/meta'); ap.add_argument('--save_maps', action='store_true')
ap.add_argument('--test_scenes', default=None); ap.add_argument('--held', default=None)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); os.makedirs('../results/prep', exist_ok=True)
def get_prep(n):
    f = f'../results/prep/{n}.pkl'
    if os.path.exists(f): return pickle.load(open(f, 'rb'))
    P = prep_scene(load_scene(n)); pickle.dump(P, open(f, 'wb')); return P
res = {}
def evaluate(net, test_names, seed):
    for n in test_names:
        P = get_prep(n); sc = load_scene(n)
        for p in args.protocols.split(','):
            if p == 'cross' and n not in CROSS: continue
            draws = [0] if p in ('mean', 'cross') else range(args.draws)
            for d in draws:
                pr = make_prior(sc, p, d) / sc['data'].max()
                t0 = time.time()
                outs = detect_full(net, P, pr.astype(np.float32), not args.no_refine, not args.no_distill, seed=seed)
                for stage, s in outs.items():
                    m = auc_metrics(s, P['gt']); m['time_s'] = time.time() - t0
                    res[f'{n}|{p}|{d}|{seed}|{stage}'] = m
                if args.save_maps: np.save(f'{args.out}/{args.tag}_{n}_{p}_{d}_s{seed}.npy', outs['final'].reshape(P['H'], P['W']).astype(np.float32))
                print(f'  {n:9s} {p:7s} d{d} ' + ' '.join(f"{st}:{res[f'{n}|{p}|{d}|{seed}|{st}']['AUC1']:.4f}/{res[f'{n}|{p}|{d}|{seed}|{st}']['AUC2']:.4f}" for st in outs), flush=True)
        json.dump(res, open(f'{args.out}/{args.tag}.json', 'w'), indent=1)
netkw = dict(use_white=not args.no_white, use_raw=not args.no_raw)
for seed in [int(s) for s in args.seeds.split(',')]:
    if args.mode == 'lodo':
        for fold in (args.held.split(',') if args.held else ['A', 'B']):
            held = FOLDS[fold]; tr = [get_prep(n) for n in DEV if n not in held]
            print(f'[2-fold dev: train {[s["name"] for s in tr]} -> test {held} seed {seed}]', flush=True)
            t0 = time.time(); net = meta_train(tr, args.steps, seed, aug=not args.no_aug, p_label=args.p_label, strength=args.strength, **netkw)
            print(f'  meta-train {time.time()-t0:.0f}s', flush=True)
            torch.save(net.state_dict(), f'{args.out}/{args.tag}_fold{fold}_s{seed}.pt')
            evaluate(net, held, seed)
    else:
        tr = [get_prep(n) for n in DEV]
        print(f'[train on all dev, seed {seed}]', flush=True)
        if args.load:
            from method import PairMatcher
            net = PairMatcher(**netkw).cuda(); net.load_state_dict(torch.load(args.load))
        else:
            t0 = time.time(); net = meta_train(tr, args.steps, seed, aug=not args.no_aug, p_label=args.p_label, strength=args.strength, **netkw)
            print(f'  meta-train {time.time()-t0:.0f}s', flush=True)
            torch.save(net.state_dict(), f'{args.out}/{args.tag}_s{seed}.pt')
        tests = args.test_scenes.split(',') if args.test_scenes else (list(SCENES) if args.mode == 'final' else DEV)
        evaluate(net, tests, seed)
