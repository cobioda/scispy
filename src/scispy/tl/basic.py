import math
import anndata as ad
import dask.dataframe as dd
import decoupler as dc
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spatialdata as sd
from matplotlib import pyplot as plt
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from shapely.geometry import Polygon
from shapely import LineString, Point, get_coordinates, affinity
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import Affine, Identity, Translation, set_transformation
from statannotations.Annotator import Annotator
import shapely
from ..tl.unfolding import extendLine
import itertools


def add_shapes_from_hdf5(
    sdata: sd.SpatialData = None,
    path: str = None,
    target_coordinates: str = "microns",
):
    """Add shapes from cell boundaries hdf5 files to spatialdata object.

    Parameters
    ----------
    sdata
        SpatialData object.
    path
        path to vizgen results files containing a cell_boundaries folder.
    target_coordinates
        target coordinates system.

    """
    sdata.table.obs["geometry"] = np.array(sdata.table.obs.shape[0], dtype=object)
    z_indexes = ["0", "1", "2", "3", "4", "5", "6"]

    for fov in pd.unique(sdata.table.obs["fov"]):
        # try:
        with h5py.File(f"{path}/cell_boundaries/feature_data_{fov}.hdf5", "r") as f:
            print(fov)
            for cell_id in sdata.table.obs.index[sdata.table.obs["fov"] == fov]:
                # print(cell_id)
                for z in z_indexes:
                    node = f"featuredata/{cell_id}/zIndex_{z}/p_0/coordinates"

                    if node in f.keys():
                        temp = f[node][0]
                        polygon = Polygon(temp)
                        sdata.table.obs["geometry"][cell_id] = polygon

    geo_df = gpd.GeoDataFrame(sdata.table.obs.geometry)
    sdata.table.obs = sdata.table.obs.drop("geometry", axis=1)
    key = sdata.table.uns["spatialdata_attrs"]["region"]
    sdata.add_shapes(key, ShapesModel.parse(geo_df, transformations={target_coordinates: Identity()}))


def add_to_shapes(
    sdata: sd.SpatialData,
    shape_file: str,
    shape_key: str = "myshapes",
    scale_factor: float = 0.50825,  # if shapes comes from xenium explorer
    target_coordinates: str = "microns",
):
    """Add shape element to SpatialData.

    Parameters
    ----------
    sdata
        SpatialData object.
    shape_file
        coordinates.csv file from xenium explorer (region = "normal_1")
        # vi coordinates.csv -> remove 2 first # lines
        # dos2unix coordinates.csv
    shape_key
        key of element shape
    scale_factor
        scale factor conversion applied to x and y coordinates for real micron coordinates
    target_coordinates
        target_coordinates system

    """
    names = []
    polygons = []
    df = pd.read_csv(shape_file)
    for name, group in df.groupby("Selection"):
        if len(group) >= 3:
            poly = Polygon(zip(group.X, group.Y))
            polygons.append(poly)
            names.append(name)

    d = {"name": names, "geometry": polygons}
    gdf = gpd.GeoDataFrame(d)
    # gdf[["mytype", "myreplicate"]] = gdf["name"].str.split("_", expand=True)
    # gdf = gdf.rename(columns={"name": "myname"})

    # scale because it comes from the xenium explorer !!!
    gdf.geometry = gdf.geometry.scale(xfact=scale_factor, yfact=scale_factor, origin=(0, 0))

    # substract the initial image offset (x,y)
    image_object_key = list(sdata.images.keys())[0]
    matrix = sd.transformations.get_transformation(sdata[image_object_key], target_coordinates).to_affine_matrix(
        input_axes=["x", "y"], output_axes=["x", "y"]
    )
    x_translation = matrix[0][2]
    y_translation = matrix[1][2]
    gdf.geometry = gdf.geometry.apply(affinity.translate, xoff=x_translation, yoff=y_translation)

    # gdf[["mytype", "myreplicate"]] = gdf["name"].str.split("_", expand=True)
    # gdf = gdf.rename(columns={"name": "myname"})
    # gdf.regionType = gdf.regionType.astype("category")

    sdata.shapes[shape_key] = ShapesModel.parse(gdf, transformations={target_coordinates: Identity()})


def add_to_points(
    sdata: sd.SpatialData,
    shape_key: str="cell_boundaries",
    point_key: str = "celltype",
    label_key: str = "scmusk",
    x_key: str = "x",
    y_key: str = "y",
    target_coordinates: str = "global",
):
    """Add anatomical shapes to sdata.

    Parameters
    ----------
    sdata
        SpatialData object.
    shape_key
        shape_key in sdata from which get the centroid
    label_key
        label_key in sdata.table.obs to add as shape element
    x_key
        x coordinate in sdata.table.obs to add as shape element x coordinate
    y_key
        y coordinate in sdata.table.obs to add as shape element y coordinate
    target_coordinates
        target_coordinates system of sdata object

    """
    # can't do that
    # sdata['PGW9-2-2A_region_0_polygons']['cell_type'] = sdata.table.obs.cell_type

    # could also be done using centroid on polygons but ['x','y'] columns is great for counting along x axis in scis.pl.plot_shape_along_axis()
    # gdf = sdata['PGW9-2-2A_region_0_polygons'].centroid
    
    gdf = sdata[shape_key].centroid
    df = pd.DataFrame(get_coordinates(gdf))
    df['ct'] = list(sdata['table'].obs[label_key])
    df = df.rename(columns={0: x_key, 1: y_key})
    ddf = dd.from_pandas(df, npartitions=1)
    sdata.points[point_key] = PointsModel.parse(
            ddf, coordinates={"x": x_key, "y": y_key}, transformations={target_coordinates: Identity()}
    )
    
    #df = pd.DataFrame(sdata.table.obs[[label_key, x_key, y_key]])
    #df = df.rename(columns={label_key: "ct"})
    #ddf = dd.from_pandas(df, npartitions=1)
    #sdata.points[point_key] = PointsModel.parse(
    #    ddf, coordinates={"x": x_key, "y": y_key}, transformations={target_coordinates: Identity()}
    #)


def get_sdata_polygon(
    sdata: sd.SpatialData,
    shape_key: str = "myshapes",
    polygon_name_key: str = "name",
    polygon_name: str = None,
    color_key: str = "celltype",
    target_coordinates: str = "microns",
    figsize: tuple = (8, 2),
) -> sd.SpatialData:
    """SpatialData polygon object using sd.polygon_query()

    Parameters
    ----------
    sdata
        SpatialData object.
    shape_key
        sdata shape element key where to locate the polygon object
    polygon_name_key
        polygon name key in the sdata shape element
    polygon_name
        polygon name
    color_key
        color key for UMAP plot, needed to sync sdata.table.uns[color_key + "_colors"]
    target_coordinates
        target_coordinates system of sdata object
    figsize
        figure size
    Returns
    -------
    sdata polygon object.
    """
    poly = sdata[shape_key][sdata[shape_key][polygon_name_key] == polygon_name].geometry.item()
    sdata_poly = sd.polygon_query(
        sdata,
        poly,
        target_coordinate_system=target_coordinates,
        filter_table=True,
        points=True,
        shapes=True,
        images=True,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    sdata.pl.render_images().pl.show(ax=ax1)
    sdata_poly.pl.render_shapes(elements=shape_key, outline=True, fill_alpha=0.25, outline_color="red").pl.show(ax=ax1)
    sc.pl.embedding(sdata_poly.table, "umap", color=color_key, ax=ax2)
    plt.tight_layout()

    return sdata_poly


def prep_pseudobulk(
    sdata: sd.SpatialData,
    shape_key: str = "myshapes",
    myname_key: str = "pseudoname",
    mytype_key: str = "pseudotype",
    target_coordinates: str = "microns",
) -> sd.SpatialData:
    """Prepare sdata.table.obs for tl.run_pseudobulk()

    Parameters
    ----------
    sdata
        SpatialData object.
    shape_key
        sdata shape element key where to find the polygon defining the zones with "name" = "type_replicate"
    myname_key
        key of group
    mytype_key
        key of type
    myreplicate_key
        key of replicate
    target_coordinates
        target_coordinates system
    Returns
    -------
    sdata polygon object.
    """
    sdata[shape_key][["type", "replicate"]] = sdata[shape_key]["name"].str.split("_", expand=True)

    sdata.table.obs[myname_key] = "#NA"
    sdata.table.obs[mytype_key] = "#NA"
    # sdata.table.obs[myreplicate_key] = "#NA"

    region_key = sdata.table.uns["spatialdata_attrs"]["region"]
    my_shapes = {region_key: sdata[region_key], shape_key: sdata[shape_key]}
    my_tables = {"table": sdata["table"]}
    sdata2 = SpatialData(shapes=my_shapes, tables=my_tables)

    for i in range(0, len(sdata2[shape_key])):
        print(sdata2[shape_key].name[i])

        poly = sdata2[shape_key].geometry[i]
        myname = sdata2[shape_key]["name"][i]
        mytype = sdata2[shape_key]["type"][i]
        # myreplicate = sdata[shape_key]["replicate"][i]

        sdata3 = sd.polygon_query(
            sdata2,
            poly,
            target_coordinate_system=target_coordinates,
            filter_table=True,
        )
        sdata2.table.obs.loc[sdata3.table.obs.index.to_list(), myname_key] = myname
        sdata2.table.obs.loc[sdata3.table.obs.index.to_list(), mytype_key] = mytype
        # sdata.table.obs.loc[sdata2.table.obs.index.to_list(), myreplicate_key] = myreplicate

    return sdata2

 
def pseudobulk(
    adata: ad.AnnData,
    replicate: str,
    condition: str, # obv. need to be unique per replicate
    groups_key: str,
    conds: list | None = None,
    pairwise: list | None = None,
    groups: list | None = None,
    key_added: str = 'results',
    layer: str = "counts",
    min_cells: int = 5,
    top_volcano: int = 20,
    min_counts: int = 100,
    sign_thr: float = 0.05,
    lFCs_thr: int = 0.5,
    save: bool = False,
    save_prefix: str = "decoupler",
    figsize: tuple = (8,3),
    digits: int = 3,
    shrink_LFC: bool = False,
    plots: bool = False,
    quiet: bool = True,
):
    """Decoupler pydeseq2 pseudobulk handler.

    Parameters
    ----------
    adata
        AnnData object.
    replicate
        replicate key
    condition
        condition key
    conds
        list of the 2 conditions to compare
    groups_key
        sdata.table.obs key, i.e. cell types
    groups
        specify the cell types to work with
    key_added
        The key added to `adata.uns['scispy']` result is saved to.
    layer
        sdata.table count values layer
    min_cells
        minimum cell number to keep replicate
    min_counts
        minimum total count to keep replicate
    sign_thr
        significant value threshold
    lFCs_thr
        log-foldchange value threshold
    save
        wether or not to save plots and tables
    save_prefix
        prefix for saved plot and tables
    figsize
        figure size

    Returns
    -------
    Return a global pd.DataFrame containing the pseudobulk analysis for plotting.
    """
    # https://decoupler-py.readthedocs.io/en/latest/notebooks/pseudobulk.html
    # sns.set(font_scale=0.5)  
    adata.uns["scispy"] = {}
    
    if groups is None:
        groups = adata.obs[groups_key].cat.categories.tolist()
    # print(groups)

    if (conds is None) & (pairwise is None):
        conds = list(adata.obs[condition].cat.categories)
        pairwise = list(itertools.combinations(conds, 2))
    elif (conds != None) & (pairwise is None):
        pairwise = list(itertools.combinations(conds, 2))
    elif (conds is None) & (pairwise != None):
        conds = list(set(itertools.chain(*pairwise)))
    
    print(pairwise)
    print(conds)
    
    # if (conds is None):
    #     conds = list(adata.obs[condition].cat.categories)
    #     print(conds)

    # if pairwise is None:
    #     pairwise = list(itertools.combinations(conds, 2))
    # else:
    #     conds = list(set(itertools.chain(*pairwise)))
    # print(pairwise)
    # print(conds)

    # conds.sort()
    # conds = [reference_level, tested_level]
 
    adconds = adata[(adata.obs[condition].isin(conds)) & (adata.obs[groups_key].isin(groups))].copy()

    pdata = dc.get_pseudobulk(
        adconds,
        sample_col=replicate,  
        # groups_col=groups_key,  
        groups_col=[groups_key, condition],  
        layer=layer,
        mode="sum",
        min_cells=min_cells,
        min_counts=min_counts,
    )
    # print(pdata)
    adata.uns["scispy"]["matrice"] = pd.DataFrame(pdata.X.T, index=pdata.var_names, columns=pdata.obs_names) 
    # dc.plot_psbulk_samples(pdata, groupby=[replicate, groups_key], figsize=figsize)
    
    df_total = pd.DataFrame()
    
    for test, ref in pairwise:
        print(f'Start pseudobulk by comparing {test} versus {ref} in the condition {condition}.')
        for ct in groups:
            print(ct)
            sub = pdata[(pdata.obs[groups_key] == ct) & (pdata.obs[condition].isin([ref, test]))].copy()
            print(sub.obs[condition].unique())
            
            if sub.n_obs > 1: 
                genes = dc.filter_by_expr(sub, group=condition, min_count=5, min_total_count=5)
                sub = sub[:, genes].copy()
                
                # if (sub.n_vars > 0) & (len(sub.obs[condition].unique().tolist()) > 1):
                if (sub.n_vars > 0) & (len(sub.obs[condition].unique()) > 1):
                # if len(sub.obs[condition].unique().tolist()) > 1:
                    dds = DeseqDataSet(
                        adata=sub,
                        design=f"~{condition}",
                        # design_factors=condition,
                        # ref_level=[condition, conds[1]],
                        refit_cooks=True,
                        quiet=quiet,
                    )
                    
                    if len(sub.obs[replicate].unique()) > 2:
                    # if len(sub.obs[replicate].unique().tolist()) > 2:
                        dds.deseq2()
                        stat_res = DeseqStats(dds, contrast=[condition, 
                                                            test, ref], 
                                            quiet=quiet)
                        stat_res.summary()
                        print(stat_res.contrast_vector.index[1])
                        if shrink_LFC:
                            stat_res.lfc_shrink(stat_res.contrast_vector.index[1])
                            # stat_res.lfc_shrink(coeff=condition+"[T."+conds[1]+"]")
                        results_df = stat_res.results_df

                        # sign_thr=0.05, lFCs_thr=0.5
                        results_df["pvals"] = -np.log10(results_df["padj"])
                        # up_msk = (results_df["log2FoldChange"] >= lFCs_thr) & (results_df["pvals"] >= -np.log10(sign_thr))
                        # dw_msk = (results_df["log2FoldChange"] <= -lFCs_thr) & (results_df["pvals"] >= -np.log10(sign_thr))
                        # signs = results_df[up_msk | dw_msk].sort_values("pvals", ascending=False)
                        # signs = signs.iloc[:top_volcano]
                        # signs = signs.sort_values("log2FoldChange", ascending=False)
                        
                        # concatenate to total
                        # signs[groups_key] = ct
                        # results_df[groups_key] = ct
                        results_df["cell_type"] = ct
                        results_df["condition"] = test + "_" + ref
                        results_df["cond_1"] = test
                        results_df["cond_2"] = ref
                        # adconds[(adconds.obs[groups_key] == ct) & (adconds.obs['smoking'] == conds[1]), df_basal['index']].layers['counts'].sum(axis=0)
                        # adconds[(adconds.obs[groups_key] == ct) & (adconds.obs['smoking'] == conds[0]), df_basal['index']].layers['counts'].sum(axis=0)

                        # adconds[(adconds.obs[groups_key] == ct) & (adconds.obs['smoking'] == conds[1])] # 5376  
                        # adconds[(adconds.obs[groups_key] == ct) & (adconds.obs['smoking'] == conds[0])] # 6072   

                        nb_cells_1 = int(sub.obs.loc[sub.obs[condition] == test, 'psbulk_n_cells'].sum())
                        nb_cells_2 = int(sub.obs.loc[sub.obs[condition] == ref, 'psbulk_n_cells'].sum())

                        results_df['nbCellsTotal_1'] = nb_cells_1   
                        results_df['nbCellsTotal_2'] = nb_cells_2 

                        results_df['sum_1'] = sub[sub.obs[condition] == test].X.sum(axis=0)
                        results_df['sum_2'] = sub[sub.obs[condition] == ref].X.sum(axis=0)
                        # results_df['sum_1'] = sub[sub.obs['smoking'] == conds[0], results_df['index']].X.sum(axis=0)
                        # results_df['sum_2'] = sub[sub.obs['smoking'] == conds[1], results_df['index']].X.sum(axis=0)
                        # results_df['pct_1'] = np.round((adconds[(adconds.obs[groups_key] == ct) & (adconds.obs[condition] == conds[0]), df_basal['index']].layers[layer] > 0).sum(axis=0) / nb_cells_1, decimals=digits).T
                        # results_df['pct_2'] = np.round((adconds[(adconds.obs[groups_key] == ct) & (adconds.obs[condition] == conds[1]), df_basal['index']].layers[layer] > 0).sum(axis=0) / nb_cells_2, decimals=digits).T
                        
                        results_df['pct_1'] = np.round((adconds[(adconds.obs[groups_key] == ct) & (adconds.obs[condition] == test), results_df.index].layers[layer] > 0).sum(axis=0) / nb_cells_1, decimals=digits).T
                        results_df['pct_2'] = np.round((adconds[(adconds.obs[groups_key] == ct) & (adconds.obs[condition] == ref), results_df.index].layers[layer] > 0).sum(axis=0) / nb_cells_2, decimals=digits).T

                        df_total = pd.concat([df_total, results_df.reset_index(names="gene")])
                        # df_total = pd.concat([df_total, results_df.reset_index()])
                        print("============================================================================")
                        print("============================================================================")
                        if plots :
                            if len(signs.index.tolist()) > 0:
                                fig, axs = plt.subplots(1, 2, figsize=figsize)
                                dc.plot_volcano_df(results_df, x="log2FoldChange", y="padj", ax=axs[0], top=top_volcano)
                                axs[0].set_title(ct + "("+conds[1]+"-"+conds[0]+")")
                                sc.pp.normalize_total(sub)
                                sc.pp.log1p(sub)
                                sc.pp.scale(sub, max_value=10)
                                sc.pl.matrixplot(sub, signs.index, groupby=replicate, ax=axs[1])
                                plt.tight_layout()

                            if save is True:
                                results_df.to_csv(save_prefix + "_" + ct + ".csv")
                                fig.savefig(save_prefix + "_" + ct + ".pdf", bbox_inches="tight")

    adata.uns["scispy"][key_added] = df_total.reset_index(drop=True)
    # adata.uns["scispy"]["pseudobulk"][key_added] ???
    print(f'results stored in adata.uns["scispy"][{key_added}]')
    print(f'--> scis.pl.plot_pseudobulk(adata, key={key_added})')


# def pseudobulk(
#     adata: an.AnnData,
#     replicate: str,
#     condition: str, # obv. need to be unique per replicate
#     conds: tuple = [],
#     groups_key: str = "scmusk",
#     groups: tuple = [],
#     key_added: str = 'results',
#     layer: str = "counts",
#     min_cells: int = 5,
#     top_volcano: int = 20,
#     min_counts: int = 200,
#     sign_thr: float = 0.05,
#     lFCs_thr: int = 0.5,
#     save: bool = False,
#     save_prefix: str = "decoupler",
#     figsize: tuple = (8,3),
# ) -> pd.DataFrame:
#     """Decoupler pydeseq2 pseudobulk handler.

#     Parameters
#     ----------
#     adata
#         AnnData object.
#     replicate
#         replicate key
#     condition
#         condition key
#     conds
#         list of the 2 conditions to compare
#     groups_key
#         sdata.table.obs key, i.e. cell types
#     groups
#         specify the cell types to work with
#     key_added
#         The key added to `adata.uns['scispy']` result is saved to.
#     layer
#         sdata.table count values layer
#     min_cells
#         minimum cell number to keep replicate
#     min_counts
#         minimum total count to keep replicate
#     sign_thr
#         significant value threshold
#     lFCs_thr
#         log-foldchange value threshold
#     save
#         wether or not to save plots and tables
#     save_prefix
#         prefix for saved plot and tables
#     figsize
#         figure size

#     Returns
#     -------
#     Return a global pd.DataFrame containing the pseudobulk analysis for plotting.
#     """
#     # https://decoupler-py.readthedocs.io/en/latest/notebooks/pseudobulk.html
#     # sns.set(font_scale=0.5)
    
#     conds.sort()
#     adconds = adata[adata.obs[condition].isin(conds)].copy()

#     pdata = dc.get_pseudobulk(
#         adconds,
#         sample_col=replicate,  # "pseudoname"
#         groups_col=groups_key,  # celltype
#         layer=layer,
#         mode="sum",
#         min_cells=min_cells,
#         min_counts=min_counts,
#     )
#     # dc.plot_psbulk_samples(pdata, groupby=[replicate, groups_key], figsize=figsize)

#     if groups is None:
#         groups = adconds.obs[groups_key].cat.categories.tolist()

#     df_total = pd.DataFrame()
#     for ct in groups:
#         print(ct)
        
#         sub = pdata[pdata.obs[groups_key] == ct].copy()

#         if len(sub.obs[condition].to_list()) > 1:
#             # Obtain genes that pass the thresholds
#             genes = dc.filter_by_expr(sub, group=condition, min_count=5, min_total_count=5)
#             # Filter by these genes
#             sub = sub[:, genes].copy()

#             if len(sub.obs[condition].unique().tolist()) > 1:
#                 # Build DESeq2 object
#                 dds = DeseqDataSet(
#                     adata=sub,
#                     design_factors=condition,
#                     ref_level=[condition, conds[1]],
#                     refit_cooks=True,
#                     quiet=True,
#                 )

#                 if len(sub.obs[replicate].unique().tolist()) > 2:
#                     dds.deseq2()
#                     stat_res = DeseqStats(dds, contrast=[condition, conds[1], conds[0]], quiet=True)
#                     stat_res.summary()
#                     # might be cond_2
#                     stat_res.lfc_shrink(coeff=condition+"[T."+conds[1]+"]")
#                     results_df = stat_res.results_df

#                     # sign_thr=0.05, lFCs_thr=0.5
#                     results_df["pvals"] = -np.log10(results_df["padj"])
#                     up_msk = (results_df["log2FoldChange"] >= lFCs_thr) & (results_df["pvals"] >= -np.log10(sign_thr))
#                     dw_msk = (results_df["log2FoldChange"] <= -lFCs_thr) & (results_df["pvals"] >= -np.log10(sign_thr))
#                     signs = results_df[up_msk | dw_msk].sort_values("pvals", ascending=False)
#                     signs = signs.iloc[:top_volcano]
#                     signs = signs.sort_values("log2FoldChange", ascending=False)

#                     if len(signs.index.tolist()) > 0:
#                         fig, axs = plt.subplots(1, 2, figsize=figsize)
#                         dc.plot_volcano_df(results_df, x="log2FoldChange", y="padj", ax=axs[0], top=top_volcano)
#                         axs[0].set_title(ct + "("+conds[1]+"-"+conds[0]+")")
#                         sc.pp.normalize_total(sub)
#                         sc.pp.log1p(sub)
#                         sc.pp.scale(sub, max_value=10)
#                         sc.pl.matrixplot(sub, signs.index, groupby=replicate, ax=axs[1])
#                         plt.tight_layout()

#                         # concatenate to total
#                         signs[groups_key] = ct
#                         results_df[groups_key] = ct
#                         df_total = pd.concat([df_total, results_df.reset_index()])

#                         if save is True:
#                             results_df.to_csv(save_prefix + "_" + ct + ".csv")
#                             fig.savefig(save_prefix + "_" + ct + ".pdf", bbox_inches="tight")
    
#     adata.uns["scispy"] = {}
#     adata.uns["scispy"][key_added] = df_total.reset_index(drop=True)
#     print("results stored in adata.uns['scispy']['",key_added,"']")
#     print("--> scis.pl.plot_pseudobulk(adata, key='",key_added,"')")


def sdata_rotate(
    sdata: sd.SpatialData,
    rotation_angle: int = 0,
    obs_x: str = "center_x",
    obs_y: str = "center_y",
    obsm_key: str = "spatial",
    target_coordinates: str = "microns",
):
    """Apply a rotation to sdata object elements + [obs_x,obs_y] sdata.table.obs + sdata.table.obsm[obsm_key]

    Parameters
    ----------
    sdata
        SpatialData object.
    rotation_angle
        horary rotation angle
    obs_x
        x coordinate in sdata.table.obs
    obs_y
        y coordinate in sdata.table.obs
    obsm_key
        key in sdata.table.obsm storing spatial coordinates for squidpy plots
    target_coordinates
        target_coordinates system of sdata object

    """
    # 360∘ = 2π  rad
    # 180∘ = π   rad
    #  90∘ = π/2 rad
    #  60∘ = π/3 rad
    #  30∘ = π/6 rad

    # rotate the shape along x axis
    if rotation_angle != 0:
        theta = math.pi / (180 / rotation_angle)
        # perform rotation of shape
        rotation = Affine(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ],
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
        # translation = Translation([0, 0], axes=("x", "y"))
        # sequence = Sequence([rotation, translation])

        # for element in sdata._gen_elements_values():
        #    set_transformation(element, rotation, set_all=True)

        elements = list(sdata.images.keys()) + list(sdata.points.keys()) + list(sdata.shapes.keys())
        for i in range(0, len(elements)):
            set_transformation(sdata[elements[i]], rotation, to_coordinate_system=target_coordinates)

        # synchronization for obs and squidpy coordinates
        A = np.vstack((sdata.table.obs[obs_x], sdata.table.obs[obs_y]))

        theta = np.pi / (180 / rotation_angle)
        rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Translation vector is the mean of all xs and ys.
        translate = A.mean(axis=1, keepdims=True)

        out = A - translate  # Step 1
        out = rotate @ out  # Step 2
        out = out + translate  # Step 3

        sdata.table.obs[obs_x] = out[0]
        sdata.table.obs[obs_y] = out[1]
        spatial = sdata.table.obs[[obs_x, obs_y]].to_numpy()
        sdata.table.obsm[obsm_key] = spatial

    # convert to final coordinates
    # sdata_out = sdata.transform_to_coordinate_system(target_coordinates)

    # if 'celltypes' in list(sdata.points.keys()):
    #    sdata_out.pl.render_points(elements='celltypes', color='celltype').pl.show(figsize=(10,4))

    # return sdata_out


def sdata_querybox(
    sdata: sd.SpatialData,
    xmin: int = 0,
    xmax: int = 0,
    ymin: int = 0,
    ymax: int = 0,
    set_origin: bool = True,
    x_origin: int = 0,
    y_origin: int = 0,
    obs_x: str = "center_x",
    obs_y: str = "center_y",
    target_coordinates: str = "microns",
) -> sd.SpatialData:
    """Subset an sdata object to the coordinates box received, then set origin to (x_origin,y_origin)

    Parameters
    ----------
    sdata
        SpatialData object.
    xmin
        xmin
    xmax
        xmax
    ymin
        ymin
    ymax
        ymax
    set_origin
        wether or not translate coordinate to origin (x_origin,y_origin)
    x_origin
        define new x origin
    y_origin
        define new y origin
    obs_x
        x coordinate in sdata.table.obs
    obs_y
        y coordinate in sdata.table.obs
    target_coordinates
        target_coordinates

    Returns
    -------
    SpatialData object
    """
    # convert to real coordinates
    sdata2 = sdata.transform_to_coordinate_system(target_coordinates)

    sdata_crop = sdata2.query.bounding_box(
        axes=["x", "y"],
        min_coordinate=[xmin, ymin],
        max_coordinate=[xmax, ymax],
        target_coordinate_system=target_coordinates,
        filter_table=True,
    )

    if set_origin is True:
        sdata_crop.table.obs[obs_x] = sdata_crop[sdata_crop.table.uns["spatialdata_attrs"]["region"][0]].centroid.x
        sdata_crop.table.obs[obs_y] = sdata_crop[sdata_crop.table.uns["spatialdata_attrs"]["region"][0]].centroid.y
        sdata_crop.table.obsm["spatial"] = sdata_crop.table.obs[[obs_x, obs_y]].to_numpy()

        translation = Translation(
            [-sdata_crop.table.obs[obs_x].min() + x_origin, -sdata_crop.table.obs[obs_y].min() + y_origin],
            axes=("x", "y"),
        )
        elements = list(sdata_crop.images.keys()) + list(sdata_crop.points.keys()) + list(sdata_crop.shapes.keys())
        for i in range(0, len(elements)):
            set_transformation(sdata_crop[elements[i]], translation, to_coordinate_system=target_coordinates)

        # convert to final coordinates
        sdata_out = sdata_crop.transform_to_coordinate_system(target_coordinates)

        sdata_out.table.obs[obs_x] = sdata_out[sdata_out.table.uns["spatialdata_attrs"]["region"][0]].centroid.x
        sdata_out.table.obs[obs_y] = sdata_out[sdata_out.table.uns["spatialdata_attrs"]["region"][0]].centroid.y
        sdata_out.table.obsm["spatial"] = sdata_out.table.obs[[obs_x, obs_y]].to_numpy()

        return sdata_out

    else:
        return sdata_crop


def scis_prop(
    adata: ad.AnnData,
    group_by: str = "scmusk_T4",
    group_only: str = None,
    split_by: str = "anatomy",
    split_only: str = "",
    split_by_top: int = 5,
    replicate: str = "sample",
    condition: str = "group",
    condition_order: tuple = ["CTRL", "PAH"],  # might be possible to provide more conditions
    test: str = "t-test_ind", #t-test_ind, t-test_welch, t-test_paired, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, Levene, Wilcoxon, Kruskal, Brunner-Munzel
    figsize: tuple = (6, 3),
):
    """Compute per zone celltype proportion between 2 conditions using replicate for statistical testing

    Parameters
    ----------
    adata
        AnnData object.
    group_by
        group
    group_only
        just plot this group
    split_by
        x value split_by
    split_only
        focus on this split_by
    split_by_top
        top split_by to consider
    replicate
        replicate key in adata.obs
    condition
        condition key in adata.obs
    condition_order
        tuple of the x conditions to test
    test
        statistical test to use
    figsize
        figure size
    Returns
    -------

    """

    #print("group_by="+group_by)
    #sns.set_theme(style="whitegrid", palette="pastel")
    l = list(adata.obs[group_by].unique())
    if group_only is not None:
        l = [group_only]
    
    #print(l)

    for n in l:
        print(n)
        df = adata[adata.obs[group_by] == n].obs[[replicate, condition, split_by]]
        df2 = df.groupby([replicate, condition, split_by])[split_by].count().unstack()
        df2 = df2.div(df2.sum(axis=1), axis=0).reset_index()
        df2 = df2.melt(id_vars=[replicate, condition])
        df2 = df2.dropna()
        df2 = df2[df2.value > 0]
        
        hits = list(df[split_by].value_counts().head(split_by_top).keys())
        df2 = df2[df2[split_by].isin(hits)]

        if split_only:
            df2 = df2[df2[split_by] == split_only]
            hits = [split_only]

        split_order = hits

        pairs = []
        for s in split_order:
            if len(df2[df2[split_by] == s][condition].unique()) > 1:
                pairs.append([(s, condition_order[0]), (s, condition_order[1])])
                if(len(condition_order) > 2):
                    pairs.append([(s, condition_order[0]), (s, condition_order[2])])
                    pairs.append([(s, condition_order[1]), (s, condition_order[2])])

        hue_plot_params = {
            "data": df2,
            "x": split_by,
            "y": "value",
            "order": split_order,
            "hue": condition,
            "hue_order": condition_order,
            # "palette": pal_group,
        }

        if len(pairs) > 0:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            sns.boxplot(ax=ax, **hue_plot_params, boxprops={"alpha": 0.8}, showfliers=False, linewidth=0.5)
            sns.stripplot(ax=ax, **hue_plot_params, dodge=True, edgecolor="black", linewidth=0.5, size=3)

            annotator = Annotator(ax, pairs, **hue_plot_params)
            annotator.configure(test=test, text_format="star")
            annotator.apply_and_annotate()

            handles, labels = ax.get_legend_handles_labels()
            l = plt.legend(handles[0:len(condition_order)], labels[0:len(condition_order)], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=6)
            ax.set_yticklabels(ax.get_yticklabels(), size=6)
            ax.xaxis.grid(True)
            ax.yaxis.grid(True)
            ax.set(ylabel="")
            ax.set_title(str(n))
            plt.tight_layout()


def fromAxisMedialToDf(
    sdata: sd.SpatialData,
    axisMedial: shapely.LineString,
    nb_interval: int = 10,
    shape_key: str = 'cell_boundaries',
    group_by: str = "cell_type_pred",
    # group_lst: '[]|None' = None,
    # scale_factor: float = 1/0.2125, # because xenium in microns and sdata in global
    return_df: bool = False,
):
    """Compute 'nb_interval' regular intervals along the centerline

    Parameters
    ----------        
    sdata (sd.SpatialData): _description_
    axisMedial (shapely.LineString): _description_
    nb_interval (int, optional): _description_. Defaults to 10.
    shape_key (str, optional): _description_. Defaults to 'cell_boundaries'.
    group_by (str, optional): _description_. Defaults to "cell_type_pred".
    scale_factor (float, optional): _description_. Defaults to 1/0.2125.

    Returns:
        _type_: _description_
    """
    # length = axisMedial.length
    # start = shapely.get_coordinates(axisMedial)[0]
    # end = shapely.get_coordinates(axisMedial)[-1]
    
    labels = [str(i) for i in range(nb_interval)]
    bin_size = axisMedial.length / nb_interval
    x = np.arange(0, axisMedial.length, bin_size)
    x = np.append(x, axisMedial.length)
    # interval_dict = {pos: shapely.line_interpolate_point(axis_medial, pos) for pos in x}
    
    # if group_lst == None:
    #     group_lst = list(sdata.table.obs[group_by].unique())

    # find pts sur la ligne le plus proche de la cellule
    sdata[shape_key]["closest_point_on_line"] = sdata[shape_key].apply(
        lambda row: shapely.ops.nearest_points(row.geometry.centroid, axisMedial)[1] , axis = 1)
    # distance du pts start au point closest (plus proche de la cellule)
    # ATTENTION il faut faire la distance sur la courbe
    sdata[shape_key]["distance_from_start"] = sdata[shape_key]['closest_point_on_line'].apply(
        lambda row: axisMedial.project(row))

    # colonne distance en colonne categorie par rapport a x si distance entre 0 et 1 label 0 etc.
    sdata[shape_key]['cat'] = pd.cut(sdata[shape_key]['distance_from_start'], 
                                 bins=x, labels=labels, right=True, include_lowest=True)
    # sdata[shape_key]['cat'] = sdata[shape_key]['cat'].astype(int)
    
    if return_df:
        return sdata[shape_key]
    else: 
        return
    
    
def df_for_genes(
    sdata: sd.SpatialData,
    axisMedial: shapely.LineString,
    genes: str | list,
    nb_interval: int = 10,
    transcript_key: str = 'transcripts',
    feature_key: str = 'feature_name',
    qv: int = 20,
    # shape_key: 'str' = 'cell_boundaries',
    group_by: str = "cell_type",
    # group_lst: '[]|None' = None,
    # scale_factor: 'float' = 1/0.2125, # because xenium in microns and sdata in global
    # return_df: 'bool' = False,
):
    """Calculate the number of transcripts for a list of genes in 'nb_interval' regulars intervals

    Args:
        sdata (sd.SpatialData): _description_
        axisMedial (shapely.LineString): _description_
        genes (str | list): _description_
        nb_interval (int, optional): _description_. Defaults to 10.
        transcript_key (str, optional): _description_. Defaults to 'transcripts'.
        feature_key (str, optional): _description_. Defaults to 'feature_name'.
        qv (int, optional): _description_. Defaults to 20.
        group_by (str, optional): _description_. Defaults to "cell_type".

    Returns:
        df_trans_sub: _description_
    """
    labels = [str(i) for i in range(nb_interval)]

    bin_size = axisMedial.length / nb_interval
    x = np.arange(0, axisMedial.length, bin_size)
    x = np.append(x, axisMedial.length)
    # interval_df = pd.DataFrame(x, columns= ["position"])

    # # depuis le start -> find pts a une distance de 'position'
    # interval_df['point'] = interval_df.apply(lambda row: shapely.line_interpolate_point(axis_medial, row.position) , axis = 1)
    # interval_df
    
    df_transcripts = sdata[transcript_key].compute()
    df_trans_sub = df_transcripts[df_transcripts[feature_key].isin(genes)]
    df_trans_sub = df_trans_sub[df_trans_sub['qv'] >= qv]

    # find pts sur la ligne le plus proche de la cellule
    df_trans_sub["closest"] = df_trans_sub.apply(lambda row: 
        shapely.ops.nearest_points(
            shapely.Point([row.x, row.y]),
            axisMedial)[1] , axis = 1)

    # distance du pts start au point closest (plus proche de la cellule)
    # ATTENTION il faut faire la distance sur la courbe
    df_trans_sub["distance"] = df_trans_sub['closest'].apply(
        lambda row: axisMedial.project(row))

    # colonne distance en colonne categorie par rapport a x si distance entre 0 et 1 label 0 etc.
    df_trans_sub['cat'] = pd.cut(df_trans_sub['distance'], 
                                 bins=x, labels=labels, right=True)
    df_trans_sub[feature_key] = df_trans_sub[feature_key].cat.remove_unused_categories()
    df_trans_sub = df_trans_sub.merge(sdata['table'].obs[['cell_id', group_by]], 
                                      on = 'cell_id', how = 'left')
    # how = left -> else remove transcript not assign to a cell ? 
    # can be put by default on inner to remove transcript not assigned to a cell
    return df_trans_sub


# def find_polygon(geometry, up, down):
#     if up.intersects(geometry.centroid):
#         return 1
#     elif down.intersects(geometry.centroid):
#         return 2
#     elif up.intersects(geometry):
#         return 1
#     elif down.intersects(geometry):
#         return 2
#     else:
#         return 0
    
    

def orthogonalDistance(
    data: pd.DataFrame | gpd.GeoDataFrame,
    # sdata: sd.SpatialData,
    polygon: shapely.Polygon, 
    centerline: shapely.LineString,
    # shape_key: str = 'cell_boundaries',
    group_by: str | None = None,
    distance: int = 30,
    round: int = 3,
) -> gpd.GeoDataFrame:  
    """Normalize the distance by following the othogonal axis

    Args:
        data (pd.DataFrame | gpd.GeoDataFrame): _description_
        polygon (shapely.Polygon): _description_
        centerline (shapely.LineString): _description_
        group_by (str | None, optional): _description_. Defaults to None.
        distance (int, optional): _description_. Defaults to 30.
        round (int, optional): _description_. Defaults to 3.

    Returns:
        gpd.GeoDataFrame: _description_
    """

    df_compute = data.copy()
    if not isinstance(df_compute, gpd.GeoDataFrame):
        df_compute = gpd.GeoDataFrame(df_compute, geometry=gpd.points_from_xy(df_compute['x'], df_compute['y']))

    if len(shapely.ops.split(polygon, centerline).geoms) == 1 :
        print('Extend line...')
        order_centers= shapely.get_coordinates(centerline)
        extendedLine_start = extendLine(order_centers[0, :], 
                                        order_centers[1, :], distance=distance)
        extendedLine_end = extendLine(order_centers[-1, :], 
                                        order_centers[-2, :], distance=distance)
        lineFinal = shapely.LineString(np.vstack([shapely.get_coordinates(extendedLine_start)[0], 
                                                order_centers,
                                                shapely.get_coordinates(extendedLine_end)[0]]))
        split_shapes = shapely.ops.split(polygon, lineFinal)
        
    if len(split_shapes.geoms) == 2:
        up_shape = split_shapes.geoms[0]
        down_shape = split_shapes.geoms[1]
    else:
        print(len(split_shapes.geoms))
        print("Increase distance")
        return
    
    df_compute['distance_pts_line'] = df_compute.distance(centerline)
    gdf_polygons = gpd.GeoDataFrame({'cat_layers': [1, 2]}, geometry=[up_shape, down_shape])
    df_compute = gpd.sjoin(df_compute, gdf_polygons, predicate="intersects", how="left")
    # type(df_trans_sub) # geopandas.geodataframe.GeoDataFrame

    df_compute.loc[df_compute['cat_layers'] == 1, 'distance_pts_line'] *= -1
    df_compute['distance_pts_line'] -= df_compute['distance_pts_line'].min()
    # print(df_compute['distance_pts_line'].min())
    df_compute['distance_normalize']  = (df_compute['distance_pts_line'] / df_compute['distance_pts_line'].max()).round(round)
    
    return df_compute



    
# def orthogonalDistance(
#     sdata: sd.SpatialData,
#     polygon: shapely.Polygon, 
#     centerline: shapely.LineString,
#     shape_key: str = 'cell_boundaries',
#     distance: int = 30,
#     round: int = 3,
# ):
#     if len(shapely.ops.split(polygon, centerline).geoms) == 1 :
#         order_centers= shapely.get_coordinates(centerline)
#         extendedLine_start = scis.tl.unfolding.extendLine(order_centers[0, :], 
#                                         order_centers[1, :], distance=distance)
#         extendedLine_end = scis.tl.unfolding.extendLine(order_centers[-1, :], 
#                                         order_centers[-2, :], distance=distance)
#         lineFinal = shapely.LineString(np.vstack([shapely.get_coordinates(extendedLine_start)[0], 
#                                                 order_centers,
#                                                 shapely.get_coordinates(extendedLine_end)[0]]))
#         split_shapes = shapely.ops.split(polygon, lineFinal)
        
#         if len(split_shapes.geoms) == 2:
#             up_shape = split_shapes.geoms[0]
#             down_shape = split_shapes.geoms[1]
#         else:
#             print(len(split_shapes.geoms))
#             print("Increase distance")
#             return
    
#     sdata[shape_key]["distance_pts_line"] = sdata[shape_key]["geometry"].apply(
#         lambda row: shapely.distance(row.centroid, centerline))
#     sdata[shape_key]['cat_layers']  = sdata[shape_key]["geometry"].apply(
#         lambda row: find_polygon(row, up_shape,down_shape))    
#     sdata[shape_key].loc[sdata[shape_key]['cat_layers'] == 1, 'distance_pts_line'] *= -1
#     sdata[shape_key]['distance_pts_line'] -= sdata[shape_key]['distance_pts_line'].min()
#     sdata[shape_key]['distance_normalize']  = (sdata[shape_key]['distance_pts_line'] / sdata[shape_key]['distance_pts_line'].max()).round(round)
#     # print(sdata[shape_key])