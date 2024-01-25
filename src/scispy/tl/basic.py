import anndata as an
import dask.dataframe as dd
import decoupler as dc
import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spatialdata as sd
import squidpy as sq
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as mplPolygon
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import Identity

from scispy.io.basic import load_bounds_pixel


def add_to_shapes(
    sdata: sd.SpatialData,
    shape_file: str,
    shape_key: str = "myshapes",
    scale_factor: float = 0.50825,  # if shapes comes from xenium explorer
    target_coordinates: str = "microns",
):
    """Add anatomical shapes to sdata.

    Parameters
    ----------
    sdata
        SpatialData object.
    shape_file
        coordinates.csv file from xenium explorer (region = "normal_1")
        # vi coordinates.csv -> remove 2 first # lines
        # dos2unix coordinates.csv
    shape_name
        name of shapes
    scale_factor
        scale factor to get back to real microns
    target_coordinates
        target_coordinates system of sdata object

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

    # gdf.regionType = gdf.regionType.astype("category")

    sdata.add_shapes(
        shape_key, ShapesModel.parse(gdf, transformations={target_coordinates: Identity()}), overwrite=True
    )


def add_to_obs_for_pseudobulk(
    sdata: sd.SpatialData,
    shape_key: str = "myshapes",
    myname_key: str = "myname",
    mytype_key: str = "mytype",
    myreplicate_key: str = "myreplicate",
):
    """Add anatomical shape annotation to sdata.table.obs

    Parameters
    ----------
    sdata
        SpatialData object.
    shape_name
        sdata shape element key
    myname_key
        key of group
    mytype_key
        key of type
    myreplicate_key
        key of replicate

    """
    sdata.table.obs[myname_key] = "#NA"
    sdata.table.obs[mytype_key] = "#NA"
    sdata.table.obs[myreplicate_key] = "#NA"

    for i in range(0, len(sdata[shape_key])):
        poly = sdata[shape_key].geometry[i]
        myname = sdata[shape_key][myname_key][i]
        mytype = sdata[shape_key][mytype_key][i]
        myreplicate = sdata[shape_key][myreplicate_key][i]

        sdata2 = sd.polygon_query(
            sdata, poly, target_coordinate_system="microns", filter_table=True, points=False, shapes=True, images=True
        )
        sdata.table.obs.loc[sdata2.table.obs.index.to_list(), myname_key] = myname
        sdata.table.obs.loc[sdata2.table.obs.index.to_list(), mytype_key] = mytype
        sdata.table.obs.loc[sdata2.table.obs.index.to_list(), myreplicate_key] = myreplicate


def add_to_points(
    sdata: sd.SpatialData,
    label_obs_key: str = "cell_type",
    x_obs_key: str = "center_x",
    y_obs_key: str = "center_y",
    sdata_group_key: str = "celltypes",
    target_coordinates: str = "microns",
):
    """Add anatomical shapes to sdata.

    Parameters
    ----------
    sdata
        SpatialData object.
    label_obs_key
        label_key in sdata.table.obs to add
    x_obs_key
        x coordinate in sdata.table.obs
    y_obs_key
        y coordinate in sdata.table.obs
    point_key
        point key to add to sdata
    target_coordinates
        target_coordinates system of sdata object

    """
    # on ne peux pas faire ça
    # sdata['PGW9-2-2A_region_0_polygons']['cell_type'] = sdata.table.obs.cell_type

    # could also be done using centroid on polygons but ['x','y'] columns is great for counting along x axis in scis.pl.plot_shape_along_axis()
    # gdf = sdata['PGW9-2-2A_region_0_polygons'].centroid

    df = pd.DataFrame(sdata.table.obs[[label_obs_key, x_obs_key, y_obs_key]])
    ddf = dd.from_pandas(df, npartitions=1)
    points = PointsModel.parse(
        ddf,
        coordinates={"x": x_obs_key, "y": y_obs_key},
        transformations={target_coordinates: Identity()},
    )
    sdata.add_points(sdata_group_key, points, overwrite=True)


def get_sdata_polygon(
    sdata: sd.SpatialData,
    shape_name_key: str = "myshapes",
    polygon_name_key: str = "name",
    polygon_name: str = None,
    color_key: str = "cell_type",
    target_coordinates: str = "microns",
    figsize: tuple = (8, 2),
) -> sd.SpatialData:
    """Get sdata polygon object

    Parameters
    ----------
    sdata
        SpatialData object.
    polygon_name_key
        polygon name key
    polygon_name
        polygon name
    Returns
    -------
    Return sdata polygon object.
    """
    poly = sdata[shape_name_key][sdata[shape_name_key][polygon_name_key] == polygon_name].geometry.item()
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
    sdata_poly.pp.get_elements([shape_name_key]).pl.render_shapes(
        outline=True, fill_alpha=0.25, outline_color="red"
    ).pl.show(ax=ax1)
    sc.pl.embedding(sdata_poly.table, "umap", color=color_key, ax=ax2)
    plt.tight_layout()

    return sdata_poly


def do_pseudobulk(
    adata: an.AnnData,
    sample_col: str,
    groups_col: str,
    condition_col: str,
    condition_1: str,
    condition_2: str,
    celltypes: tuple = None,
    layer: str = "counts",
    min_cells: int = 5,
    min_counts: int = 200,
    save: bool = False,
    save_prefix: str = "decoupler",
    figsize: tuple = (8, 2),
) -> pd.DataFrame:
    """Decoupler pydeseq2 pseudobulk handler.

    Parameters
    ----------
    adata
        Anndata object.

    Returns
    -------
    Return pseudobulk analysis according to parameters.
    """
    # https://decoupler-py.readthedocs.io/en/latest/notebooks/pseudobulk.html

    sns.set(font_scale=0.5)

    adata_clean = adata[adata.obs[condition_col].isin([condition_1, condition_2])].copy()

    pdata = dc.get_pseudobulk(
        adata_clean,
        sample_col=sample_col,  # zone
        groups_col=groups_col,  # celltype
        layer=layer,
        mode="sum",
        min_cells=min_cells,
        min_counts=min_counts,
    )
    dc.plot_psbulk_samples(pdata, groupby=[sample_col, groups_col], figsize=figsize)

    if celltypes is None:
        celltypes = adata_clean.obs[groups_col].cat.categories.tolist()

    df_total = pd.DataFrame()
    for ct in celltypes:
        sub = pdata[pdata.obs[groups_col] == ct].copy()

        if len(sub.obs[condition_col].to_list()) > 1:
            # Obtain genes that pass the thresholds
            genes = dc.filter_by_expr(sub, group=condition_col, min_count=5, min_total_count=5)
            # Filter by these genes
            sub = sub[:, genes].copy()

            if len(sub.obs[condition_col].unique().tolist()) > 1:
                # Build DESeq2 object
                dds = DeseqDataSet(
                    adata=sub,
                    design_factors=condition_col,
                    ref_level=[condition_col, condition_1],
                    refit_cooks=True,
                    quiet=True,
                )
                dds.deseq2()
                stat_res = DeseqStats(dds, contrast=[condition_col, condition_1, condition_2], quiet=True)

                stat_res.summary()
                coeff_str = condition_col + "_" + condition_2 + "_vs_" + condition_1
                stat_res.lfc_shrink(coeff=coeff_str)

                results_df = stat_res.results_df

                fig, axs = plt.subplots(1, 2, figsize=figsize)
                dc.plot_volcano_df(results_df, x="log2FoldChange", y="padj", ax=axs[0], top=20)
                axs[0].set_title(ct)

                # sign_thr=0.05, lFCs_thr=0.5
                results_df["pvals"] = -np.log10(results_df["padj"])

                up_msk = (results_df["log2FoldChange"] >= 0.5) & (results_df["pvals"] >= -np.log10(0.05))
                dw_msk = (results_df["log2FoldChange"] <= -0.5) & (results_df["pvals"] >= -np.log10(0.05))
                signs = results_df[up_msk | dw_msk].sort_values("pvals", ascending=False)
                signs = signs.iloc[:20]
                signs = signs.sort_values("log2FoldChange", ascending=False)

                # concatenate to total
                signs[groups_col] = ct
                df_total = pd.concat([df_total, signs.reset_index()])

                if len(signs.index.tolist()) > 0:
                    sc.pp.normalize_total(sub)
                    sc.pp.log1p(sub)
                    sc.pp.scale(sub, max_value=10)
                    sc.pl.matrixplot(sub, signs.index, groupby=sample_col, ax=axs[1])

                plt.tight_layout()

                if save is True:
                    results_df.to_csv(save_prefix + "_" + ct + ".csv")
                    fig.savefig(save_prefix + "_" + ct + ".pdf", bbox_inches="tight")

    # pivlfc = pd.pivot_table(df_total, values=["log2FoldChange"], index=["index"], columns=[groups_col], fill_value=0)
    # pd.pivot_table(df_total, values=["pvals"], index=["index"], columns=[groups_col], fill_value=0)
    ## plot pivot table as heatmap using seaborn
    # sns.clustermap(pivlfc, cmap="vlag", figsize=(6, 6))
    ## plt.setp( ax.xaxis.get_majorticklabels(), rotation=90)
    # plt.tight_layout()
    # plt.show()

    return df_total


def compare_seg(
    adata_1: an.AnnData,
    adata_2: an.AnnData,
    color_key: str,
    x: int,
    y: int,
    size_x: int,
    size_y: int,
    lw: float = 1,
    image_cmap: str = "bone",
    mypal: dict = None,
    figsize: tuple = (8, 8),
) -> int:
    """Compare segmentation between 2 anndata object of the same slice.

    Parameters
    ----------
    adata
        Anndata object.

    Returns
    -------
    Return segmentation comparison
    """
    fig, ax = plt.subplots(figsize=figsize)

    library_id = adata_1.obs.library_id.cat.categories.tolist()[0]
    img = sq.im.ImageContainer(adata_1.uns["spatial"][library_id]["images"]["hires"], library_id=library_id)
    crop1 = img.crop_corner(y, x, size=(size_y, size_x))
    adata_crop = crop1.subset(adata_1, spatial_key="pixel")
    crop1.show(layer="image", ax=ax, cmap=image_cmap)

    ser_color = pd.Series(mypal)
    pts = adata_crop.obs[["x_pix", "y_pix", color_key]]
    pts.x_pix -= x
    pts.y_pix -= y
    ax = sns.scatterplot(
        x="x_pix", y="y_pix", s=0, alpha=0.5, edgecolors=color_key, data=pts, hue=color_key, palette=mypal
    )

    # full cell segmentation
    adata_crop = load_bounds_pixel(adata_crop, library_id)

    currentCells = []
    typedCells = []
    for cell_id in adata_crop.obs.index:
        currentCells.append(adata_crop.obs.bounds[cell_id])
        typedCells.append(adata_crop.obs[color_key][cell_id])

    polygon_data = []
    for inst_index in range(len(currentCells)):
        inst_cell = currentCells[inst_index]
        df_poly_z = pd.DataFrame(inst_cell).transpose()
        df_poly_z.columns.tolist()
        xxx = df_poly_z.values - [x, y]
        inst_poly = np.array(xxx.tolist())
        polygon_data.append(inst_poly)

    polygons = [mplPolygon(polygon_data[i].reshape(-1, 2)) for i in range(len(polygon_data))]
    ax.add_collection(PatchCollection(polygons, fc=ser_color[typedCells], ec="None", alpha=0.5))
    # axs[1].add_collection(PatchCollection(polygons, fc="none", ec=ser_color[typedCells], lw=1))

    # then nuclei segmentation
    adata_crop_nuc = crop1.subset(adata_2, spatial_key="pixel")
    adata_crop_nuc = load_bounds_pixel(adata_crop_nuc, library_id)

    currentCells = []
    typedCells = []
    for cell_id in adata_crop_nuc.obs.index:
        currentCells.append(adata_crop_nuc.obs.bounds[cell_id])
        typedCells.append(adata_crop_nuc.obs[color_key][cell_id])

    polygon_data = []
    for inst_index in range(len(currentCells)):
        inst_cell = currentCells[inst_index]
        df_poly_z = pd.DataFrame(inst_cell).transpose()
        df_poly_z.columns.tolist()
        xxx = df_poly_z.values - [x, y]
        inst_poly = np.array(xxx.tolist())
        polygon_data.append(inst_poly)

    polygons = [mplPolygon(polygon_data[i].reshape(-1, 2)) for i in range(len(polygon_data))]
    ax.add_collection(PatchCollection(polygons, fc="none", ec=ser_color[typedCells], lw=1, ls="--"))

    plt.tight_layout()

    return 0


def compare_label_knn(
    adata_1: an.AnnData,
    adata_2: an.AnnData,
    color_key: str = "celltype",
    image_cmap: str = "viridis",
    figsize: tuple = (8, 7),
) -> int:
    """Compare segmentation between 2 anndata object of the same slice.

    Parameters
    ----------
    adata
        Anndata object.

    Returns
    -------
    Return segmentation comparison
    """
    df = adata_1.obs[[color_key, "center_x", "center_y"]]
    gpd1 = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.center_x, df.center_y))

    dfnuc = adata_2.obs[[color_key, "center_x", "center_y"]]
    gpd2 = gpd.GeoDataFrame(dfnuc, geometry=gpd.points_from_xy(dfnuc.center_x, dfnuc.center_y))

    # unary union of the gpd2 geomtries
    pts3 = gpd2.geometry.unary_union

    def near(point, pts=pts3):
        # find the nearest point and return the corresponding Place value
        nearest = gpd2.geometry == nearest_points(point, pts)[1]
        return gpd2[nearest][color_key][0]

    gpd1["Nearest"] = gpd1.apply(lambda row: near(row.geometry), axis=1)

    print("entity_1  = ", len(gpd1))
    print("entity_2  = ", len(gpd2))
    print("identical = ", len(gpd1[gpd1[color_key] == gpd1.Nearest]))

    # Build co-occurrence matrix and set diagonal to zero.
    ct = 100 * pd.crosstab(gpd1["celltype"], gpd1["Nearest"], normalize="index")

    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(ct, cmap=image_cmap)
    plt.show()

    return 0
