import math

import dask.dataframe as dd
import decoupler as dc
import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from matplotlib import pyplot as plt
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from shapely import affinity
from shapely.geometry import Polygon
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import Affine, Identity, Translation, set_transformation


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

    sdata.add_shapes(
        shape_key, ShapesModel.parse(gdf, transformations={target_coordinates: Identity()}), overwrite=True
    )


def add_to_points(
    sdata: sd.SpatialData,
    label_key: str = "celltype",
    x_key: str = "center_x",
    y_key: str = "center_y",
    element_key: str = "celltypes",
    target_coordinates: str = "microns",
):
    """Add anatomical shapes to sdata.

    Parameters
    ----------
    sdata
        SpatialData object.
    label_key
        label_key in sdata.table.obs to add as shape element
    x_key
        x coordinate in sdata.table.obs to add as shape element x coordinate
    y_key
        y coordinate in sdata.table.obs to add as shape element y coordinate
    element_key
        element point key to add to sdata
    target_coordinates
        target_coordinates system of sdata object

    """
    # can't do that
    # sdata['PGW9-2-2A_region_0_polygons']['cell_type'] = sdata.table.obs.cell_type

    # could also be done using centroid on polygons but ['x','y'] columns is great for counting along x axis in scis.pl.plot_shape_along_axis()
    # gdf = sdata['PGW9-2-2A_region_0_polygons'].centroid

    df = pd.DataFrame(sdata.table.obs[[label_key, x_key, y_key]])
    df = df.rename(columns={label_key: "ct"})
    ddf = dd.from_pandas(df, npartitions=1)
    points = PointsModel.parse(
        ddf,
        coordinates={"x": x_key, "y": y_key},
        transformations={target_coordinates: Identity()},
    )
    sdata.add_points(element_key, points, overwrite=True)


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
    sdata_poly.pp.get_elements([shape_key]).pl.render_shapes(
        outline=True, fill_alpha=0.25, outline_color="red"
    ).pl.show(ax=ax1)
    sc.pl.embedding(sdata_poly.table, "umap", color=color_key, ax=ax2)
    plt.tight_layout()

    return sdata_poly


def prep_pseudobulk(
    sdata: sd.SpatialData,
    shape_key: str = "myshapes",
    myname_key: str = "pseudoname",
    mytype_key: str = "pseudotype",
    target_coordinates: str = "microns",
):
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

    """
    sdata[shape_key][["type", "replicate"]] = sdata[shape_key]["name"].str.split("_", expand=True)

    sdata.table.obs[myname_key] = "#NA"
    sdata.table.obs[mytype_key] = "#NA"
    # sdata.table.obs[myreplicate_key] = "#NA"

    for i in range(0, len(sdata[shape_key])):
        poly = sdata[shape_key].geometry[i]
        myname = sdata[shape_key]["name"][i]
        mytype = sdata[shape_key]["type"][i]
        # myreplicate = sdata[shape_key]["replicate"][i]

        sdata2 = sd.polygon_query(
            sdata,
            poly,
            target_coordinate_system=target_coordinates,
            filter_table=True,
            points=False,
            shapes=True,
            images=True,
        )
        sdata.table.obs.loc[sdata2.table.obs.index.to_list(), myname_key] = myname
        sdata.table.obs.loc[sdata2.table.obs.index.to_list(), mytype_key] = mytype
        # sdata.table.obs.loc[sdata2.table.obs.index.to_list(), myreplicate_key] = myreplicate


def run_pseudobulk(
    sdata: sd.SpatialData,
    pseudotype_1: str,
    pseudotype_2: str,
    pseudotype_key: str = "pseudotype",
    pseudoname_key: str = "pseudoname",
    groups_key: str = "celltype_spatial",
    groups: tuple = [],
    layer: str = "counts",
    min_cells: int = 5,
    min_counts: int = 200,
    sign_thr: float = 0.05,
    lFCs_thr: int = 0.5,
    save: bool = False,
    save_prefix: str = "decoupler",
    figsize: tuple = (10, 4),
) -> pd.DataFrame:
    """Decoupler pydeseq2 pseudobulk handler.

    Parameters
    ----------
    sdata
        SpatialData object.
    pseudotype_1
        pseudobulk condition (pseudotype) 1
    pseudotype_2
        pseudobulk condition (pseudotype) 2
    pseudotype_key
        sdata.table.obs key, i.e. condition
    pseudoname_key
        sdata.table.obs key, i.e. replicate name
    groups_key
        sdata.table.obs key, i.e. cell types
    groups
        specify the cell types to work with
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

    adata = sdata.table[sdata.table.obs[pseudotype_key].isin([pseudotype_1, pseudotype_2])].copy()

    pdata = dc.get_pseudobulk(
        adata,
        sample_col=pseudoname_key,  # "pseudoname"
        groups_col=groups_key,  # celltype
        layer=layer,
        mode="sum",
        min_cells=min_cells,
        min_counts=min_counts,
    )
    # dc.plot_psbulk_samples(pdata, groupby=[pseudoname_key, groups_key], figsize=figsize)

    if groups is None:
        groups = adata.obs[groups_key].cat.categories.tolist()

    df_total = pd.DataFrame()
    for ct in groups:
        sub = pdata[pdata.obs[groups_key] == ct].copy()

        if len(sub.obs[pseudotype_key].to_list()) > 1:
            # Obtain genes that pass the thresholds
            genes = dc.filter_by_expr(sub, group=pseudotype_key, min_count=5, min_total_count=5)
            # Filter by these genes
            sub = sub[:, genes].copy()

            if len(sub.obs[pseudotype_key].unique().tolist()) > 1:
                # Build DESeq2 object
                dds = DeseqDataSet(
                    adata=sub,
                    design_factors=pseudotype_key,
                    ref_level=[pseudotype_key, pseudotype_1],
                    refit_cooks=True,
                    quiet=True,
                )
                dds.deseq2()
                stat_res = DeseqStats(dds, contrast=[pseudotype_key, pseudotype_1, pseudotype_2], quiet=True)

                stat_res.summary()
                coeff_str = pseudotype_key + "_" + pseudotype_2 + "_vs_" + pseudotype_1
                stat_res.lfc_shrink(coeff=coeff_str)

                results_df = stat_res.results_df

                fig, axs = plt.subplots(1, 2, figsize=figsize)
                dc.plot_volcano_df(results_df, x="log2FoldChange", y="padj", ax=axs[0], top=20)
                axs[0].set_title(ct)

                # sign_thr=0.05, lFCs_thr=0.5
                results_df["pvals"] = -np.log10(results_df["padj"])

                up_msk = (results_df["log2FoldChange"] >= lFCs_thr) & (results_df["pvals"] >= -np.log10(sign_thr))
                dw_msk = (results_df["log2FoldChange"] <= -lFCs_thr) & (results_df["pvals"] >= -np.log10(sign_thr))
                signs = results_df[up_msk | dw_msk].sort_values("pvals", ascending=False)
                signs = signs.iloc[:20]
                signs = signs.sort_values("log2FoldChange", ascending=False)

                # concatenate to total
                signs[groups_key] = ct
                df_total = pd.concat([df_total, signs.reset_index()])

                if len(signs.index.tolist()) > 0:
                    sc.pp.normalize_total(sub)
                    sc.pp.log1p(sub)
                    sc.pp.scale(sub, max_value=10)
                    sc.pl.matrixplot(sub, signs.index, groupby=pseudoname_key, ax=axs[1])

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


def sdata_rotate(
    sdata: sd.SpatialData,
    rotation_angle: int = 0,
    target_coordinates: str = "microns",
) -> sd.SpatialData:
    """Rotate sdata object

    Parameters
    ----------
    sdata
        SpatialData object.
    rotation_angle
        horary rotation angle
    target_coordinates
        target_coordinates system of sdata object

    Returns
    -------
    SpatialData object
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

    # convert to final coordinates
    sdata_out = sdata.transform_to_coordinate_system(target_coordinates)

    # if 'celltypes' in list(sdata.points.keys()):
    #    sdata_out.pl.render_points(elements='celltypes', color='celltype').pl.show(figsize=(10,4))

    return sdata_out


def sdata_querybox(
    sdata: sd.SpatialData,
    xmin: int = 0,
    xmax: int = 0,
    ymin: int = 0,
    ymax: int = 0,
    set_origin: bool = True,
    x_origin: int = 0,
    y_origin: int = 0,
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
    target_coordinates
        target_coordinates

    Returns
    -------
    SpatialData object
    """
    sdata_crop = sdata.query.bounding_box(
        axes=["x", "y"],
        min_coordinate=[xmin, ymin],
        max_coordinate=[xmax, ymax],
        target_coordinate_system=target_coordinates,
        filter_table=True,
    )

    if set_origin is True:
        sdata_crop.table.obs.center_x = sdata_crop[sdata_crop.table.uns["spatialdata_attrs"]["region"][0]].centroid.x
        sdata_crop.table.obs.center_y = sdata_crop[sdata_crop.table.uns["spatialdata_attrs"]["region"][0]].centroid.y
        sdata_crop.table.obsm["spatial"] = sdata_crop.table.obs[["center_x", "center_y"]].to_numpy()

        translation = Translation(
            [-sdata_crop.table.obs.center_x.min() + x_origin, -sdata_crop.table.obs.center_y.min() + y_origin],
            axes=("x", "y"),
        )
        elements = list(sdata_crop.images.keys()) + list(sdata_crop.points.keys()) + list(sdata_crop.shapes.keys())
        for i in range(0, len(elements)):
            set_transformation(sdata_crop[elements[i]], translation, to_coordinate_system=target_coordinates)

        # convert to final coordinates
        sdata_out = sdata_crop.transform_to_coordinate_system(target_coordinates)

        sdata_out.table.obs.center_x = sdata_out[sdata_out.table.uns["spatialdata_attrs"]["region"][0]].centroid.x
        sdata_out.table.obs.center_y = sdata_out[sdata_out.table.uns["spatialdata_attrs"]["region"][0]].centroid.y
        sdata_out.table.obsm["spatial"] = sdata_out.table.obs[["center_x", "center_y"]].to_numpy()

        # cropped_sdata.table.obs.celltype = cropped_sdata.table.obs.celltype.cat.remove_unused_categories()
        # cropped_sdata.points['celltypes'].celltype = cropped_sdata.points['celltypes'].celltype.cat.remove_unused_categories()
        # del cropped_sdata.table.uns['celltype_colors']
        return sdata_out

    else:
        return sdata_crop
