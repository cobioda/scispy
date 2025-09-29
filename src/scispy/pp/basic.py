import anndata as an
import pandas as pd
import scanpy as sc
import scvi
import numpy as np
import spatialdata as sd
from matplotlib import pyplot as plt

def metrics_summary(
    sdata: 'sd.SpatialData',
    scale_factor: 'float' = 0.2125
) -> 'dict':
    adata = sdata['table'].copy()
    transcripts = sdata['transcripts']
    img_dims = sdata['morphology_focus']['scale0'].dims

    number_of_cells_detected = adata.n_obs
    median_transcripts_per_cell = adata.obs['transcript_counts'][adata.obs['transcript_counts'] != 0].median()
    adata.obs['genes_per_cell'] = np.sum(adata.X > 0, axis=1)
    median_genes_per_cell = adata.obs['genes_per_cell'][adata.obs['genes_per_cell'] != 0].median()

    total_cell_area = round(adata.obs['cell_area'].sum(), ndigits = 1)
    segmentation_method = adata.obs['segmentation_method'].value_counts()
    # percent_segmentation_method = dict(round(segmentation_method / number_of_cells_detected * 100, ndigits=1))
    percent_segmentation_method = round(segmentation_method / number_of_cells_detected * 100, ndigits=1)
    
    total_high_quality_decoded_transcripts = len(transcripts[
                                                 (transcripts['qv'] >= 20) &
                                                 (transcripts['is_gene'])
                                                 ])
    # is_gene = TRUE -> custom_gene + predesigned_gene
    number_of_fov_selected = len(transcripts['fov_name'].unique())
    transcripts_within_cells = len(transcripts[
                                   (transcripts['qv'] >= 20) &
                                   (transcripts['is_gene']) &
                                   (transcripts['cell_id'] != "UNASSIGNED")
                                   ])
    
    percent_of_transcripts_within_cells = round(transcripts_within_cells / total_high_quality_decoded_transcripts * 100, 
                                                ndigits=1)
    
    nuclear_transcript_count = len(transcripts[
                                   (transcripts['qv'] >= 20) &
                                   (transcripts['is_gene']) &
                                   (transcripts['cell_id'] != "UNASSIGNED") &
                                   (transcripts['overlaps_nucleus'] == 1)
                                   ])
    # The density of cells per 100 microns squared.
    nuclear_transcripts_per_100µm = round(nuclear_transcript_count / 
                                          adata.obs['nucleus_area'].sum() *100,
                                          ndigits=1)
    
    # TO improve
    region_area = round((img_dims['x'] * scale_factor) * (img_dims['y'] * scale_factor), 
                        ndigits=1) 
    # 3591422.2 ERROR trouver la bonne technique
    # 3,131,787.3
    cells_per_100µm = round(number_of_cells_detected / region_area * 100,
                            ndigits=2)
    # The density of cells per 100 microns squared.
    # 0,22 ERROR -> 0,25

    dict_columns = {'Number of cells detected': number_of_cells_detected,
                    'Median transcripts per cell': median_transcripts_per_cell,
                    'Nuclear transcripts per 100 µm²': nuclear_transcripts_per_100µm,
                    'Total high quality decoded transcripts': total_high_quality_decoded_transcripts,
                    'Number of FOV selected': number_of_fov_selected,
                    'Region area (µm²)': region_area,
                    'Total cell area (µm²)': total_cell_area,
                    'Percent of transcripts within cells': percent_of_transcripts_within_cells,	
                    'Cells per 100 µm²': cells_per_100µm,
                    'Median genes per cell': median_genes_per_cell,
                   } | dict(segmentation_method)
    return dict_columns

    
def run_scanpy(
    sdata: sd.SpatialData,
    min_counts: int = 20,
    resolution: float = 0.5,
    positive_coord_only: bool = True,
    key: str = "leiden",
    scale: bool = True,
):
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

    if positive_coord_only is True:
        adata = sdata.table[sdata.table.obs.center_x > 0]
        adata2 = adata[adata.obs.center_y > 0]
        del sdata.tables["table"]
        sdata.table = adata2

    print("remaining cells=", sdata.table.shape[0])

    sc.pp.normalize_total(sdata.table)
    sc.pp.log1p(sdata.table)
    sdata.table.raw = sdata.table

    # sc.pp.scale(sdata.table, max_value=10)
    sc.tl.pca(sdata.table, svd_solver="arpack")
    sc.pp.neighbors(sdata.table, n_neighbors=10)
    sc.tl.umap(sdata.table)
    sc.tl.leiden(sdata.table, resolution=resolution, key_added=key)

    if scale:
        sc.pp.scale(sdata.table, max_value=10)
        
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    sc.pl.embedding(sdata.table, "umap", color=key, ax=axs[0], show=False)
    sc.pl.embedding(sdata.table, "spatial", color=key, ax=axs[1])
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
    batch_size: int = 128,
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
    vae.train(batch_size=batch_size)

    # Register the object and run scANVI
    scvi.model.SCANVI.setup_anndata(
        concat,
        layer=layer,
        batch_key="tech",
        labels_key=label_key,
        unlabeled_category="nan",
    )

    lvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=label_key, unlabeled_category="nan", adata=concat)
    lvae.train(max_epochs=20, n_samples_per_label=100, batch_size=batch_size)

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
    # nb_cells = ad_spatial.shape[0]
    # ad_spatial = ad_spatial[ad_spatial.obs[f"{label_key}_score"] >= filter_under_score]
    # filtered_cells = nb_cells - ad_spatial.shape[0]
    # print("low assignment score filtering ", filtered_cells)


def sync_shape(
    sdata: sd.SpatialData,
    shape_key: str = None,
    table_key: str = 'table',

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

    _cells = sdata[table_key].obs[sdata[table_key].uns["spatialdata_attrs"]["instance_key"]].to_list()
    sdata[shape_key] = sdata[shape_key].loc[sdata[shape_key].index.isin(_cells)]

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
