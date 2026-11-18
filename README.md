# Retinal Vascular Graph Stability Analysis

An intermediate-level research project on the propagation of segmentation errors into retinal vascular topology and geometry. The workflow skeletonizes expert vessel masks, constructs branch graphs, measures geometric properties and compares robustness across two real public datasets.

## Research question

How do small segmentation changes affect retinal vascular graph structure, branch geometry, tortuosity and diameter estimates, and how much do simple short-branch filtering and centerline smoothing change the measured stability?

## Datasets

### HRF

The official FAU HRF dataset supplies 45 colour fundus images, expert vessel segmentations and field-of-view masks.

- https://www5.cs.fau.de/research/data/fundus-images/
- https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip
- Licence: CC BY 4.0

Citation: Budai, A., Bock, R., Maier, A., Hornegger, J., & Michelson, G. (2013). Robust Vessel Segmentation in Fundus Images. *International Journal of Biomedical Imaging*, 2013.

### CHASE_DB1

Kingston University's dataset includes 28 retinal fundus images and two independent human vessel annotations per image. The analysis uses the first human observer (`1stHO`) annotations. The field of view is estimated from the non-black foreground of the original image because a matching FOV ground-truth mask is not supplied in the downloaded dataset.

- Official dataset record: https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/
- Official archive: https://researchinnovation.kingston.ac.uk/files/40659508/CHASEDB1.zip
- Public archive mirror: https://github.com/playyalp1/CHASE_DB1
- Licence: CC BY 4.0

The Kingston archive and its former staffnet host can return HTTP 403 on some networks. The downloader therefore tries the official archive first and then the public mirror. It validates the ZIP and checks for 28 original images plus 28 first-observer annotations before extraction. The mirror is a convenience copy; Kingston remains the dataset owner and primary citation source.
- Citation: Barman, S., Hoppe, A., Remagnino, P., Fraz, M. M., Uyyanonvara, B., Rudnicka, A. R., & Owen, C. G. (2012). CHASE_DB1 retinal vessel reference dataset. Kingston University. DOI: 10.1109/TBME.2012.2205687.

The original archive is downloaded by `scripts/download_data.py`. It is not committed to Git. The script validates the ZIP and the number of images and annotation pairs before analysis. If CHASE_DB1 is unavailable, the script reports a warning and leaves the already downloaded HRF data ready to use. The notebook defaults to HRF-only mode; set `INCLUDE_CHASE=1` only after CHASE_DB1 has been installed.

## Experimental design

For every image, the original expert mask is compared with controlled erosion and dilation perturbations. These transformations are not added patient observations; they are repeatable experimental conditions generated from real expert annotations. Absolute pixel lengths are not compared directly between datasets because their image resolutions differ. Relative feature changes are summarized within each dataset.

The graph representation reports connected components, graph nodes, branches, endpoints, junctions, total branch length, tortuosity and vessel diameter estimates. A short-branch filtering analysis is included as a simple robustness check. Centerline smoothing is used only for an additional tortuosity estimate; the original skeleton and graph metrics remain available for comparison.

## HTTPS downloads on Windows

The downloader uses the operating system certificate store through `truststore` and keeps TLS certificate verification enabled. This supports Windows-managed root certificates and intermediate certificates. If your network requires an additional trusted CA, set `SSL_CERT_FILE` to a PEM-encoded CA bundle before running the downloader. Do not disable certificate verification.

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_data.py
jupyter lab
```

Open `notebooks/02_vascular_graph_stability.ipynb` and run all cells. The default is HRF-only, so CHASE_DB1 download restrictions will not block the project. If CHASE_DB1 downloads successfully, enable it in PowerShell before launching Jupyter:

```powershell
$env:INCLUDE_CHASE="1"
jupyter lab
```

Use `python scripts/download_data.py --hrf-only` to skip CHASE_DB1 explicitly, or `python scripts/download_data.py --require-chase` when both datasets are mandatory. Set `RETINA_SAMPLE_LIMIT=3` for a short smoke test; without it, all available images are processed (45 HRF plus 28 CHASE_DB1 when enabled).

## Outputs

- `results/per_image_graph_features.csv`
- `results/branch_features.csv`
- `results/dataset_stability_summary.csv`
- `results/graphs/*.graphml` (original-mask graphs)
- `results/figures/*_graph_stability.png`

## Tests

```bash
pytest -q
```

The tests use small artificial masks as unit-test fixtures only. No generated image is included or treated as retinal research data.

## Limitations

The graph is derived from an 8-connected pixel skeleton. Junction clusters and branch paths depend on skeleton connectivity, so parameters and implementation choices can influence branch counts. CHASE_DB1 has no provider-supplied FOV mask in this package; the ROI is estimated from the image foreground. Pixel measurements are not physical measurements. Controlled perturbations do not fully represent model-specific segmentation error. This is a reproducible methodological analysis, not a clinical diagnostic system.
