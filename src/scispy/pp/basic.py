import anndata as an
import pandas as pd
import scanpy as sc
import scvi
import squidpy as sq
from matplotlib import pyplot as plt


def filter_and_run_scanpy(
    adata: an.AnnData, min_counts: int = 10, resolution: float = 0.5, key: str = "clusters"
) -> an.AnnData:
    """Filter and run scanpy analysis.

    Parameters
    ----------
    adata
        Anndata object.
    min_counts
        minimum transcript count to keep cell.

    Returns
    -------
    Anndata analyzed object.
    """
    sc.pp.filter_cells(adata, min_counts=min_counts, inplace=True)

    print("total cells=", adata.shape[0])
    print("mean transcripts per cell=", adata.obs["barcodeCount"].mean())
    print("median transcripts per cell=", adata.obs["barcodeCount"].median())

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, key_added=key)

    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    sc.pl.embedding(adata, "umap", color=key, ax=axs[0], show=False)
    sq.pl.spatial_scatter(adata, color=key, shape=None, size=1, ax=axs[1])
    plt.tight_layout()

    return adata


def annotate(
    ad_spatial: an.AnnData,
    ad_ref: an.AnnData,
    label_ref: str = "celltype",
    label_key: str = "celltype",
    metaref2add: str = None,
) -> an.AnnData:
    """Transfert cell type label from single cell to spatial annData object using SCVI (scArches developpers credits for final knn neighbor transfert)

    Parameters
    ----------
    ad_spatial
        Anndata spatial object.
    ad_ref
        Anndata single-cell reference object.
    label_ref
        .obs key in single-cell reference object.
    label_key
        .obs key in spatial object.
    metaref2add
        .obs key in single-cell reference object to transfert to spatial.

    Returns
    -------
    Anndata labeled object.
    """
    ad_spatial.var.index = ad_spatial.var.index.str.upper()  # lower
    ad_ref.var.index = ad_ref.var.index.str.upper()

    print("ref. ", ad_ref.shape)
    print("viz. ", ad_spatial.shape)

    # Select shared gene panel genes only
    genes_Vizgen = ad_spatial.var.index
    genes_10x = ad_ref.var.index
    genes_shared = genes_Vizgen.intersection(genes_10x)  # List of shared genes
    ad_emb = ad_spatial[:, genes_Vizgen.isin(genes_shared)].copy()
    ad_ref = ad_ref[:, genes_10x.isin(genes_shared)]

    print(len(genes_shared), "common genes")

    # missed = list(set(genes_Vizgen) - set(genes_shared))
    # print("gene missed = ", missed)

    # Concatenate the datasets
    concat = ad_ref.concatenate(ad_emb, batch_key="tech", batch_categories=["10x", "MERFISH"]).copy()
    # concat.layers["counts"] = concat.X.copy()
    # sc.pp.normalize_total(concat, target_sum=1e4)
    # sc.pp.log1p(concat)
    # concat.raw = concat  # keep full dimension safe

    # Use the annotations from the 10x, and treat the MERFISH as unlabeled
    concat.obs[f"{label_key}"] = "nan"
    mask = concat.obs["tech"] == "10x"
    concat.obs[f"{label_key}"][mask] = ad_ref.obs[label_ref].values

    # Create the scVI latent space
    scvi.model.SCVI.setup_anndata(concat, layer="counts", batch_key="tech")
    vae = scvi.model.SCVI(concat)
    # Train the model
    vae.train()

    # Register the object and run scANVI
    scvi.model.SCANVI.setup_anndata(
        concat,
        layer="counts",
        batch_key="tech",
        labels_key=label_key,
        unlabeled_category="nan",
    )

    lvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=label_key, unlabeled_category="nan", adata=concat)
    lvae.train(max_epochs=20, n_samples_per_label=100)

    concat.obs["C_scANVI"] = lvae.predict(concat)
    concat.obsm["X_scANVI"] = lvae.get_latent_representation(concat)

    merfish_mask = concat.obs["tech"] == "MERFISH"
    ad_spatial.obs[f"{label_key}"] = concat.obs["C_scANVI"][merfish_mask].values

    if metaref2add:
        d = pd.Series(ad_ref.obs[f"{metaref2add}"].values, index=ad_ref.obs[f"{label_ref}"]).to_dict()
        ad_spatial.obs[f"{metaref2add}"] = ad_spatial.obs[f"{label_key}"].map(d)

    return ad_spatial
