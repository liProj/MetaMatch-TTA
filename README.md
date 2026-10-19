# MetaMatch-TTA

**Background-Aware Hyperspectral Target Detection with Mismatched Reference Spectra**

MetaMatch-TTA learns a shared spectral pair matcher from material tasks across development scenes. It combines raw spectra, background-whitened spectra, and analytical matching statistics. At inference, the matcher keeps its weights fixed and performs two gradient-free reference updates using candidates from the current image.

- **MetaMatch**: direct matching with the pretrained pair network.
- **MetaMatch-TTA**: direct matching followed by two reference-refinement rounds.
- **MetaMatch-TTA +S**: an optional teacher--student fusion experiment. The student is trained for each test scene and is separate from the fixed-weight main method.

The matcher has **165,857 parameters**. Sequence pooling supports different input band counts with the same weights.

## Paper

- [English main paper](paper/research_manuscript_en.pdf) · [LaTeX](paper/research_manuscript_en.tex)
- [English supplementary material](paper/research_supplement_en.pdf) · [LaTeX](paper/research_supplement_en.tex)

The main paper and supplementary material contain 20 tables, 16 figures, and 28 main-paper references in total. These are manuscript drafts. Keep both PDFs in the same directory to use the main paper's links to supplementary material.

## Repository contents

```text
code/                   Core model, scene loader, protocols, metrics, and experiment runner
data/README.md          Dataset sources and the expected local directory layout
results/final/          Target-scene run records and checkpoints for seeds 0, 1, and 2
results/meta/           Development split and component-ablation records/checkpoints
tools/                  Numerical result aggregation
tests/                  CPU smoke checks for inference and saved checkpoints
paper/                  English manuscripts and required figure/table sources
```

Raw imagery, preprocessing caches, detection-map arrays, temporary files, and historical manuscript backups are excluded from this repository.

## Environment

Use Python 3.10 or newer and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The experiment runner uses CUDA. Install a PyTorch build compatible with your NVIDIA GPU before running training or the full evaluation. The release smoke checks can run on CPU. NumPy 2.0 or newer is required by the metric implementation.

Release checks were performed with Python 3.13, NumPy 2.4.1, SciPy 1.17.0, scikit-learn 1.7.2, PyTorch 2.9.1, and Matplotlib 3.10.8. This records the upload validation environment, rather than a claim about the original training environment.

## Data

Obtain the scenes from the sources listed in [data/README.md](data/README.md), and place them under `data/dl/` using the specified filenames. MATLAB scenes must contain `data` (`H × W × L`) and `map` (`H × W`) arrays. WHR uses ENVI files.

The six target scenes are ABU-A3, ABU-B3, ABU-U4, SanDiego, HYDICE, and WHR. The six development scenes are Segundo, AVIRIS-2, Cri, SanDiego2, ABU-A2, and ABU-B4. Training uses the development scenes, including labeled target tasks with probability 0.2. Target-scene labels are used to construct the defined benchmark references and evaluate predictions.

## Train and evaluate

Run experiment commands **from `code/`**, because result and preprocessing-cache paths are relative to this directory.

```bash
cd code
python data.py ABU-A3 SanDiego HYDICE WHR
```

Train on all six development scenes and evaluate the fixed-weight main method on the target scenes:

```bash
python run_meta.py --mode final --tag metamatch_tta --steps 2000 --seeds 0,1,2 --draws 3 --protocols mean,pix1,pixk3,pixk5,perturb,cross --no_distill --out ../results/new_runs
```

`--no_distill` selects the main method without the optional student. To evaluate the +S extension, omit this flag. Add `--save_maps` to save the runner's `final` output maps. With distillation disabled, `final` is the refined matcher output. With distillation enabled, it is the teacher--student mean.

Evaluate a provided checkpoint:

```bash
python run_meta.py --mode final --load ../results/final/final_s0.pt --seeds 0 --tag checkpoint_s0 --draws 3 --no_distill --out ../results/checkpoint_eval
```

The current runner prepares development scenes before checkpoint evaluation as well, so its dataset layout should be complete. Preprocessing caches are stored under `results/prep/`; regenerate them if the underlying imagery or masks change.

Two complementary development splits and component ablations:

```bash
python run_meta.py --mode lodo --tag dev_split --held A,B --seeds 0 --steps 2000 --draws 3 --no_distill --out ../results/dev_runs
python run_meta.py --mode lodo --tag dev_noaug --held A --seeds 0 --steps 2000 --draws 3 --no_aug --no_distill --out ../results/dev_runs
python run_meta.py --mode lodo --tag dev_nowhite --held A --seeds 0 --steps 2000 --draws 3 --no_white --no_distill --out ../results/dev_runs
```

Despite the historical mode name `lodo`, the runner implements **two complementary three-scene training / three-scene evaluation splits**. Removing whitened sequence channels retains scalar side information.

## Reference protocols and output keys

| Paper protocol | Runner name | Reference |
| --- | --- | --- |
| P0 | `mean` | Mean of all ground-truth target pixels |
| P1 | `pix1` | Random single target pixel |
| P2 | `pixk3`, `pixk5` | Mean of three or five randomly sampled target pixels |
| P3 | `cross` | Mean target spectrum from a compatible donor image |
| P4 | `perturb` | Target mean with prescribed gain, smooth spectral deformation, and noise |

Reference draws use seeds `1000 + draw`. P3 evaluates only defined donor--target pairs and does not establish arbitrary cross-sensor calibration or physical material identity.

| Output key | Meaning |
| --- | --- |
| `zero` | MetaMatch |
| `refined` | MetaMatch-TTA |
| `student` | Optional scene-specific student |
| `final` | Teacher--student average when enabled; otherwise the current matcher output |
| `sharp` | Fourth power of `final` |
| `amf_ref`, `ace_ref`, `ens`, `ens_amf` | Additional analytical/reference and rank-fusion diagnostics |

The diagnostic analytical outputs are part of the saved experimental records. They are distinct from the recent published-method comparisons in the paper. Published comparison values are attributed in the manuscript and are not presented as results produced by this repository.

## Results and validation

The manuscript reports scene-macro-average ROC-AUC values of **0.9902** for MetaMatch-TTA with random single-pixel references and **0.9903** with synthetically perturbed references. Direct MetaMatch achieves **0.9895** and **0.9914**, respectively. These summaries use three training seeds and shared reference draws 0, 1, and 2.

From the repository root, summarize the included run records or run the CPU smoke checks:

```bash
python tools/summarize_results.py
python tests/smoke_test.py
```

The smoke checks verify parameter count, variable-band inference, checkpoint compatibility, finite scores, fixed matcher weights during gradient-free reference refinement, and metric behavior on a synthetic ranking example. They do not rerun the full training experiment.

## Compile the English paper

Use XeLaTeX with Times New Roman and Arial installed. The required figures and tables are included. From `paper/`, run:

```bash
python build_english.py
```

The two documents are compiled alternately to resolve references to supplementary tables and sections. Figure 1 includes editable TikZ/SVG sources in `paper/method_vector/`. Two translated statistical plots can be regenerated using `paper/english_figures/build.py` and their bundled numerical input.
