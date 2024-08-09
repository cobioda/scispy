import anndata as an
import pandas as pd
import scanpy as sc
import scvi
import spatialdata as sd
import squidpy as sq
from matplotlib import pyplot as plt
import numpy as np
import yaml
import warnings

def run_scanpy(
    sdata: 'sd.SpatialData',
    min_counts: 'int | None' = 20,
    max_value: 'float | None' = 10,
    n_neighbors: 'int' = 10,
    resolution: 'float' = 0.5,
    key: 'str' = 'leiden',
    positive_coord_only: 'bool' = True,
    *args,
    **kwargs,
):
    """Filter and run scanpy analysis.

    Parameters
    ----------
    sdata
        SpatialData object.
    min_counts
        minimum transcript count to keep cell.
    max_value
        clip to this value after scaling. If `None`, do not clip.
    n_neighbors
        number of local neighborhood.
    resolution
        resolution for clustering.
    key
        key to add for clusters.
    positive_coord_only
        keep only cells with positive coordinates
    """
    adata = sdata['table']
    print("Total cells: ", adata.shape[0])
    sc.pp.filter_cells(adata, min_counts=min_counts, *args, **kwargs)

    # filter also cells with negative coordinate center_x and center_y
    # sdata.table.obs["n_counts"] = sdata.table.layers['counts'].sum(axis=1)
    # sdata.table.obs.loc[sdata.table.obs['center_x'] < 0, 'n_counts'] = 0
    # sdata.table.obs.loc[sdata.table.obs['center_y'] < 0, 'n_counts'] = 0

    if positive_coord_only & check_if_negative_coords(sdata):
        print("Remove cells with negative coordinates")
        # inutile si pas de coords neg
        adata = adata[adata.obs['center_x'] > 0]
        adata2 = adata[adata.obs['center_y'] > 0]
        del sdata.tables["table"]
        sdata['table'] = adata2
    print("Remaining cells:", adata.shape[0])

    sc.pp.normalize_total(adata, *args, **kwargs)
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.scale(adata, max_value=max_value, *args, **kwargs)
    sc.tl.pca(adata, svd_solver="arpack", *args, **kwargs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, *args, **kwargs)
    sc.tl.umap(adata, *args, **kwargs)
    sc.tl.leiden(adata, resolution=resolution, key_added=key, *args, **kwargs)
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    sc.pl.embedding(adata, "umap", color=key, ax=axs[0], show=False)
    sq.pl.spatial_scatter(adata, color=key, shape=None, size=1, ax=axs[1])
    plt.tight_layout()

    # synchronize current shapes with filtered table
    sync_shape(sdata)

    # for vizgen previous analysis
    # sdata.shapes['cell_boundaries'] = sdata.shapes['cell_boundaries'].loc[sdata.table.obs.index.tolist()]
    return

def check_if_negative_coords(
    sdata: 'sd.SpatialData'
): 
    """Check if some cells have negative coordinates.

    Parameters
    ----------
    sdata
        SpatialData object.
    
    Return
    """
    return np.any((sdata['table'].obs['center_x'] <= 0) | (sdata['table'].obs['center_y'] <= 0))


def sync_shape(
    sdata: 'sd.SpatialData',
    shape_key: 'str | None' = None,
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
        shape_key = sdata['table'].uns["spatialdata_attrs"]["region"]

    instance_key = sdata['table'].uns["spatialdata_attrs"]["instance_key"]
    sdata[shape_key] = sdata[shape_key].loc[sdata['table'].obs[instance_key].tolist()]


def replaceGeneNames(
    adata: 'an.AnnData',
    path: 'str',
    genes: 'set',
):
    """Replace old gene name by the new gene name.

    Parameters
    ----------
    adata
        AnnData object.
    path
        path to YAML file with old gene name and new gene name.
    genes
        gene names to modifiy.
    """
    with open(path, 'r') as file:
        genes_dict = yaml.safe_load(file)

    for gene in genes:
        if gene in genes_dict.keys():
            adata.var = adata.var.rename(index={genes_dict[gene]: gene}) 
        else:
            warnings.warn(f' The new name of {gene} is not found. Please check if you find it on : "https://www.genecards.org/" and if so add it.')
    return
    
    
def prepareToScvi(
    ad_spatial: 'an.AnnData',
    ad_ref: 'an.AnnData',
    label_key: 'str' = "celltype",
    batch_categories: 'list' = ["reference", "query"],
    unlabeled_category: 'str' = "unknown",
    batch_key: 'str' = "tech",
    path_to_renamed_genes = '/data/analysis/data_fierville/SPATIAL/BPCO/Human_GRCh38_renamed_genes.yaml'
) -> 'an.AnnData':
    """Prepare the two datasets to integrate by combining common genes and concatenating them.

    Parameters
    ----------
    ad_spatial
        Anndata spatial object.
    ad_ref
        Anndata single-cell reference object.
    label_key
        .obs key in single-cell reference object and in spatial object.
    metaref2add
        .obs key in single-cell reference object to transfert to spatial.
    path_to_renamed_genes
        path to YAML file with old gene name and new gene name.

    Returns
    ----------
    Anndata object with the two datasets concatenate.
    """
    # ad_spatial.var.index = ad_spatial.var.index.str.upper()
    # ad_ref.var.index = ad_ref.var.index.str.upper()
    
    print("Reference:", ad_ref.n_vars, "genes")
    print("Qurey:", ad_spatial.n_vars, "genes")
    
    # genes in spatial only
    not_in_ref = set(ad_spatial.var_names) - set(ad_ref.var_names)
    
    if not_in_ref:
        replaceGeneNames(ad_ref, path_to_renamed_genes, genes=not_in_ref)
    else:
        print('All spatial genes are in the single-cell dataset')
    # print(set(ad_spatial.var_names) - set(ad_ref.var_names))
    genes_shared = ad_spatial.var_names.intersection(ad_ref.var_names)
    print(len(genes_shared), "common genes")

    print("Concatenate both datasets...")
    # both needs to have the count values in the layer "counts" or layer received
    concat = ad_ref[:, genes_shared].concatenate(ad_spatial[:, genes_shared], 
                                                 batch_key = batch_key, 
                                                 batch_categories=batch_categories).copy()
    concat.obs[label_key] = concat.obs[label_key].cat.add_categories([unlabeled_category])
    concat.obs.loc[concat.obs[batch_key] == batch_categories[1], label_key] = unlabeled_category
    
    return concat


def run_scANVI(
    adata_train: 'an.AnnData',
    layer: 'str' = "counts",
    batch_key: 'str' = "tech",
    label_key: 'str' = 'celltype',
    n_layers: 'int' = 2,
    n_latent: 'int' = 50,
    unlabeled_category: 'str' = "unknown",
) -> 'an.AnnData':
    """Run scvi and scanvi and predict the annotation of unlabeled cells.

    Parameters
    ----------
    adata_train
        Anndata object.
    layer
        layer in which we can find the raw count values.
    batch_key
        .obs key in anndata object that separate the reference from the query.
    label_key
        .obs key in anndata object to make the annotation.
    n_layers
        number of hidden layers used for encoder and decoder NNs.
    n_latent
        dimensionality of the latent space.
    unlabeled_category
        name of the unlabeled cells in the label_key.

    Returns
    ----------
    Anndata object integrated with predicted annotation.
    """
    scvi.model.SCVI.setup_anndata(adata_train, 
                                  layer=layer, 
                                  labels_key=label_key,
                                  batch_key=batch_key)
    vae = scvi.model.SCVI(adata_train, 
                          n_layers=n_layers, 
                          n_latent=n_latent)
    vae.train()
    print('Done scVI')
    
    # scvi.model.SCANVI.setup_anndata(concat, layer=layer, batch_key="tech",
    #                                 labels_key=label_key, unlabeled_category="nan")
    # pas trop utile on l'a deja fixe pour scvi -> garde meme chose
    
    lvae = scvi.model.SCANVI.from_scvi_model(vae, 
                                             unlabeled_category=unlabeled_category,
                                             labels_key=label_key,
                                             adata=adata_train)
    lvae.train(max_epochs=20, n_samples_per_label=100)
    print('Done scANVI')
    
    adata_train.obs["C_scANVI"] = lvae.predict(adata_train)
    adata_train.obsm["X_scANVI"] = lvae.get_latent_representation(adata_train)
    
    # add score
    df_soft = lvae.predict(adata_train, soft=True) 
    adata_train.obs["score"] = df_soft.max(axis=1)

    return adata_train



def scanvi_annotate(
    sdata: 'sd.SpatialData',
    ad_ref: 'an.AnnData',
    table: 'str' = 'table',
    label_key: 'str' = "celltype",
    batch_categories: 'list' = ["reference", "query"], # batch_categories = ["single_cell", "spatial"]
    unlabeled_category: 'str' = "unknown",
    batch_key: 'str' = "tech",
    layer: 'str' = "counts",
    path_to_renamed_genes: 'str' = '/data/analysis/data_fierville/SPATIAL/BPCO/Human_GRCh38_renamed_genes.yaml',
    metaref2add: 'list' = [],
    filter_under_score: 'float | None' = 0.5,
    n_layers: 'int' = 2,
    n_latent: 'int' = 50,
) -> 'an.AnnData':
    """Make the transfert label from the single-cell dataset to the spatial dataset.

    Parameters
    ----------
    sdata
        SpatialData object. 
    ad_ref
        Anndata single-cell reference object.
    table
        table to use in the spatial object to make the integration.
    label_key
        .obs key in single-cell reference object and in spatial object.
    batch_categories
        name of the two datasets : first name have to be the annotated 
        dataset and the second the unlabeled.
    unlabeled_category
        name of the unlabeled cells in the label_key.
    batch_key
        .obs key in anndata object that separate the reference from the query.
    layer
        layer in which we can find the raw count values.
    path_to_renamed_genes
        path to YAML file with old gene name and new gene name.
    metaref2add
        .obs key in single-cell reference object to transfert to spatial.
    filter_under_score
        remove cells having a scvi assignment score under this cutoff
    n_layers
        number of hidden layers used for encoder and decoder NNs.
    n_latent
        dimensionality of the latent space.

    Returns
    ----------
    Anndata object with the two datasets concatenate.
    """
    ad_spatial = sdata[table]
    
    if layer not in ad_ref.layers.keys():
        ad_ref.layers[layer] = ad_ref.raw.X.copy()

    concat = prepareToScvi(
        ad_spatial=ad_spatial,
        ad_ref=ad_ref,
        label_key = label_key,
        batch_categories = batch_categories,
        unlabeled_category = unlabeled_category,
        batch_key = batch_key,
        layer = layer,
        path_to_renamed_genes = path_to_renamed_genes
    )
   
    concat = run_scANVI(
        adata_train=concat,
        layer = layer,
        batch_key = batch_key,
        label_key = label_key,
        n_layers = n_layers,
        n_latent = n_latent,
        unlabeled_category = unlabeled_category
    )

    mask_to_predict = concat.obs[batch_key] == batch_categories[1] # 'query'
    ad_spatial.obs[[f"{label_key}", f"{label_key}_score"]] = concat.obs[["C_scANVI", "score"]][mask_to_predict].values
    ad_spatial.obs[f"{label_key}"] = ad_spatial.obs[f"{label_key}"].astype("category")

    for i in range(0, len(metaref2add)):
        d = pd.Series(ad_ref.obs[f"{metaref2add[i]}"].values, 
                      index=ad_ref.obs[f"{label_key}"]).to_dict()
        ad_spatial.obs[f"{metaref2add[i]}"] = ad_spatial.obs[f"{label_key}"].map(d)
        ad_spatial.obs[f"{metaref2add[i]}"] = ad_spatial.obs[f"{metaref2add[i]}"].astype("category")
    
    if filter_under_score:
        print(f'Remove cells with a score < {filter_under_score}...')
        nb_cells = ad_spatial.n_obs
        ad_spatial = ad_spatial[ad_spatial.obs[f"{label_key}_score"] >= filter_under_score]
        print(f'Low assignment score filtering {nb_cells - ad_spatial.n_obs} cells.')

    return 

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
