import anndata as an
import pandas as pd
import scanpy as sc
import scvi
import spatialdata as sd
import squidpy as sq
from matplotlib import pyplot as plt


def run_scanpy(sdata: sd.SpatialData, min_counts: int = 20, resolution: float = 0.5, key: str = "leiden"):
    """Filter and run scanpy analysis.

    Parameters
    ----------
    sdata
        SpatialData object.
    min_counts
        minimum transcript count to keep cell.
    resolution
        resolution for clustering.
    key
        key to add for clusters.

    """
    print("total cells=", sdata.table.shape[0])

    # filter also cells with negative coordinate center_x and center_y
    # sdata.table.obs["n_counts"] = sdata.table.layers['counts'].sum(axis=1)
    # sdata.table.obs.loc[sdata.table.obs['center_x'] < 0, 'n_counts'] = 0
    # sdata.table.obs.loc[sdata.table.obs['center_y'] < 0, 'n_counts'] = 0

    sc.pp.filter_cells(sdata.table, min_counts=min_counts)

    adata = sdata.table[sdata.table.obs.center_x > 0]
    adata2 = adata[adata.obs.center_y > 0]
    del sdata.table
    sdata.table = adata2

    print("remaining cells=", sdata.table.shape[0])

    sc.pp.normalize_total(sdata.table)
    sc.pp.log1p(sdata.table)
    sdata.table.raw = sdata.table

    sc.pp.scale(sdata.table, max_value=10)
    sc.tl.pca(sdata.table, svd_solver="arpack")
    sc.pp.neighbors(sdata.table, n_neighbors=10)
    sc.tl.umap(sdata.table)
    sc.tl.leiden(sdata.table, resolution=resolution, key_added=key)

    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    sc.pl.embedding(sdata.table, "umap", color=key, ax=axs[0], show=False)
    sq.pl.spatial_scatter(sdata.table, color=key, shape=None, size=1, ax=axs[1])
    plt.tight_layout()

    # synchronize current shapes with filtered table
    sync_shape(sdata)

    # for vizgen previous analysis
    # sdata.shapes['cell_boundaries'] = sdata.shapes['cell_boundaries'].loc[sdata.table.obs.index.tolist()]


def scvi_annotate(
    ad_spatial: an.AnnData,
    ad_ref: an.AnnData,
    label_ref: str = "celltype",
    label_key: str = "celltype",
    layer: str = "counts",
    metaref2add: tuple = [],
    filter_under_score: float = 0.5,
):
    """Annotate anndata spatial cells using anndata cells reference using SCVI.

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
    layer
        layer in which we can find the raw count values.
    metaref2add
        .obs key in single-cell reference object to transfert to spatial.
    filter_under_score
        remove cells having a scvi assignment score under this cutoff

    """
    ad_spatial.var.index = ad_spatial.var.index.str.upper()
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
    # both needs to have the count values in the layer "counts" or layer received
    concat = ad_ref.concatenate(ad_emb, batch_key="tech", batch_categories=["10x", "MERFISH"]).copy()
    # concat.layers[layer] = concat.raw.X.copy()

    # Use the annotations from the 10x, and treat the MERFISH as unlabeled
    concat.obs[f"{label_key}"] = "nan"
    mask = concat.obs["tech"] == "10x"
    concat.obs[f"{label_key}"][mask] = ad_ref.obs[label_ref].values

    # Create the scVI latent space
    scvi.model.SCVI.setup_anndata(concat, layer=layer, batch_key="tech")
    vae = scvi.model.SCVI(concat)
    # Train the model
    vae.train()

    # Register the object and run scANVI
    scvi.model.SCANVI.setup_anndata(
        concat,
        layer=layer,
        batch_key="tech",
        labels_key=label_key,
        unlabeled_category="nan",
    )

    lvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=label_key, unlabeled_category="nan", adata=concat)
    lvae.train(max_epochs=20, n_samples_per_label=100)

    concat.obs["C_scANVI"] = lvae.predict(concat)
    concat.obsm["X_scANVI"] = lvae.get_latent_representation(concat)

    # add score
    df_soft = lvae.predict(concat, soft=True)
    concat.obs["score"] = df_soft.max(axis=1)

    merfish_mask = concat.obs["tech"] == "MERFISH"
    ad_spatial.obs[f"{label_key}"] = concat.obs["C_scANVI"][merfish_mask].values
    ad_spatial.obs[f"{label_key}_score"] = concat.obs["score"][merfish_mask].values

    ad_spatial.obs[f"{label_key}"] = ad_spatial.obs[f"{label_key}"].astype("category")

    for i in range(0, len(metaref2add)):
        d = pd.Series(ad_ref.obs[f"{metaref2add[i]}"].values, index=ad_ref.obs[f"{label_ref}"]).to_dict()
        ad_spatial.obs[f"{metaref2add[i]}"] = ad_spatial.obs[f"{label_key}"].map(d)
        ad_spatial.obs[f"{metaref2add[i]}"] = ad_spatial.obs[f"{metaref2add[i]}"].astype("category")

    # remove cells having a bad score
    nb_cells = ad_spatial.shape[0]
    ad_spatial = ad_spatial[ad_spatial.obs[f"{label_key}_score"] >= filter_under_score]
    filtered_cells = nb_cells - ad_spatial.shape[0]
    print("low assignment score filtering ", filtered_cells)


def sync_shape(
    sdata: sd.SpatialData,
    shape_key: str = None,
):
    """Synchronize shapes with table

    Parameters
    ----------
    sdata
        SpatialData object.
    shape_key
        key of shapes to synchronize

    """
    if shape_key is None:
        shape_key = sdata.table.uns["spatialdata_attrs"]["region"]

    instance_key = sdata.table.uns["spatialdata_attrs"]["instance_key"]
    sdata.shapes[shape_key] = sdata.shapes[shape_key].loc[sdata.table.obs[instance_key].tolist()]


# def switch_region(
#    sdata: sd.SpatialData,
#    region: str = "cell_boundaries",  # could be cell_boundaries, nucleus_boundaries or cell_circles for xenium
# ):
#    """Swith to region of SpatialData object
#
#    Parameters
#    ----------
#    sdata
#        SpatialData object.
#    region
#        region (need to be a valid shape element).
#    """
#    # if i switch region
#    sdata.table.obs.region = region
#    sdata.table.uns["spatialdata_attrs"]["region"] = [region]
#    # i need to sync it to table
#    instance_key = sdata.table.uns["spatialdata_attrs"]["instance_key"]
#    sdata.shapes[region] = sdata.shapes[region].loc[sdata.table.obs[instance_key].tolist()]
