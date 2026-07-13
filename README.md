
![Sparty banner](https://github.com/MorganeFi/sparty/blob/main/docs/_static/images/logo.png?raw=true)

<div align="center">

# **sparty**

**Spatial analysis for in-situ RNA transcripts in python**

---

<p align="center">
  <a href="https://sparty.readthedocs.io/en/latest/" target="_blank">Documentation</a> •
  <a href="https://sparty.readthedocs.io/en/latest/tutorials.html" target="_blank">Examples</a> •
  <a href="https://www.biorxiv.org/" target="_blank">Preprint</a>
</p>

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/cobioda/sparty/test.yaml?branch=main
[link-tests]: https://github.com/cobioda/sparty/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/sparty

</div>

## Background

<p>
  A pthon package build on top of spatialdata for Single-Cell In-Situ Spatial-Omics data analysis, developped to handle Vizgen (merscope), Nanostring (cosmx) and 10xGenomics (Xenium) experiments.
</p>

<p align="center">
  <img src="https://github.com/MorganeFi/sparty/blob/main/docs/_static/images/logo.png" width="300px">
</p>

## Features

-   **Read in-situ spatial-omics assays experiments**: build on top of spatialdata package
-   **Automatic cell type annotation**: scanvi implementation
-   **Import anatomical .csv shape file from xenium explorer**: as anndata observations
-   **Automatic run pseudobulk data analysis**: using decoupler and pydeseq2 packages
-   **Compute cell type proportion in region**: integrating statistical test in case of replicates
-   **Produce high quality spatial figures**: build on top of spatialdata_plot package

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].
-   [Tutorials][link-tutorial]

## Installation

1. Create a conda environment (Python >= 3.11)
2. Install sparty using pip:

```bash
conda create -n sparty python==3.11
conda activate sparty
pip install git+https://github.com/cobioda/sparty.git@main
```

## Contribution

If you found a bug or you want to propose a new feature, please use the [issue tracker][issue-tracker].

[issue-tracker]: https://github.com/cobioda/sparty/issues
[changelog]: https://sparty.readthedocs.io/en/latest/changelog.html
[link-docs]: https://sparty.readthedocs.io
[link-api]: https://sparty.readthedocs.io/en/latest/api.html
[link-tutorial]: https://sparty.readthedocs.io/en/latest/tutorials.html
