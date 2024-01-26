# Single-Cell In-Situ python package

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/bfxomics/scispy/test.yaml?branch=main
[link-tests]: https://github.com/bfxomics/scispy/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/scispy

scispy is a pyhton package for in-situ spatial-omics datasets analysis, mainly developped for vizgen merscope,
scispy is build on top of spatialdata and spatialdata-io and spatialdata-plot librairies which can handle for
Nanostring (cosmx) and 10xGenomics (Xenium) experiments.

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.9 or newer installed on your system.

<!--
1) Install the latest release of `scispy` from `PyPI <https://pypi.org/project/scispy/>`_:

```bash
pip install scispy
```
-->

Install the latest development version:

```bash
pip install git+https://github.com/cobioda/scispy.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out the main developer of this package: in the [kevin lebrigand](mailto:lebrigand@ipmc.cnrs.fr).
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> preprint available soon

[issue-tracker]: https://github.com/cobioda/scispy/issues
[changelog]: https://scispy.readthedocs.io/en/latest/changelog.html
[link-docs]: https://scispy.readthedocs.io
[link-api]: https://scispy.readthedocs.io/en/latest/api.html
