# Dataset layout

Raw datasets are obtained separately from their original releases or the mirrors used in the manuscript. The paper's scene table records image dimensions, band counts, and target counts. Dataset versions and masks should be checked against that table before comparing results.

The local loader expects this layout relative to the repository root:

```text
data/dl/
  abu/
    abu-airport-3.mat
    abu-beach-3.mat
    abu-urban-4.mat
  hydice/
    HYDICE-urban.mat
  htdmamba/
    Sandiego.mat
    Sandiego2.mat
    abu-airport-2.mat
    abu-beach-4.mat
  dev/
    Segundo.mat
    aviris_2.mat
    Cri.mat
  whr/extracted/
    WHU-Hi-River.hdr
    WHU-Hi-River.img
    Anomaly detection/
      Target_mask.hdr
      Target_mask.img
```

The MATLAB loader reads arrays named `data` and `map`. Positive mask values are converted to target pixels. The optional AVIRIS-1 entry in `code/data.py` is not one of the six development scenes used by `run_meta.py`.

Sources recorded in the manuscript:

- ABU target scenes: `sxt1996/Airport-Beach-Urban-ABU` on GitHub.
- HYDICE: `sxt1996/HYDICE` on GitHub.
- Segundo, Cri, and AVIRIS-2: the `sxt1996/Segundo`, `sxt1996/Cri`, and `sxt1996/AVIRIS-2` GitHub mirrors.
- SanDiego, SanDiego2, ABU-A2, and ABU-B4: `shendb2022/HTD-Mamba` on GitHub.
- WHR: Wuhan University WHU-Hi-River release at `rsidea.whu.edu.cn`.

The WHR loader uses the anomaly-detection `Target_mask`, containing 36 targets in the manuscript's version. The HYDICE version in this project contains 21 target pixels. These details differ from some published experimental versions.

Data are excluded from Git. Refer to each original source for dataset access and usage terms.
