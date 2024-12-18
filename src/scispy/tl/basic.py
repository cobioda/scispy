import math

import anndata as an
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
from shapely import affinity
from shapely.geometry import Polygon
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import Affine, Identity, Translation, set_transformation
from statannotations.Annotator import Annotator


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
    point_key: str = "celltypes",
    label_key: str = "celltype",
    x_key: str = "center_x",
    y_key: str = "center_y",
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

    sdata.points[point_key] = PointsModel.parse(
        ddf, coordinates={"x": x_key, "y": y_key}, transformations={target_coordinates: Identity()}
    )


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


def run_pseudobulk(
    adata: an.AnnData,
    cond_1: str,
    cond_2: str,
    cond_key: str,
    replicate_key: str,
    groups_key: str = "celltype",
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
    adata
        AnnData object.
    cond_1
        pseudobulk cond_1
    cond_2
        pseudobulk cond_2
    cond_key
        sdata.table.obs cond_key
    replicate_key
        sdata.table.obs replicate_key
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

    adata = adata[adata.obs[cond_key].isin([cond_1, cond_2])].copy()

    pdata = dc.get_pseudobulk(
        adata,
        sample_col=replicate_key,  # "pseudoname"
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

        if len(sub.obs[cond_key].to_list()) > 1:
            # Obtain genes that pass the thresholds
            genes = dc.filter_by_expr(sub, group=cond_key, min_count=5, min_total_count=5)
            # Filter by these genes
            sub = sub[:, genes].copy()

            if len(sub.obs[cond_key].unique().tolist()) > 1:
                # Build DESeq2 object
                dds = DeseqDataSet(
                    adata=sub,
                    design_factors=cond_key,
                    ref_level=[cond_key, cond_1],
                    refit_cooks=True,
                    quiet=True,
                )
                print(len(sub.obs[replicate_key].unique().tolist()))
                if len(sub.obs[replicate_key].unique().tolist()) > 2:
                    dds.deseq2()
                    stat_res = DeseqStats(dds, contrast=[cond_key, cond_1, cond_2], quiet=True)

                    stat_res.summary()
                    coeff_str = cond_key + "_" + cond_2 + "_vs_" + cond_1
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
                        sc.pl.matrixplot(sub, signs.index, groupby=replicate_key, ax=axs[1])

                    plt.tight_layout()

                    if save is True:
                        results_df.to_csv(save_prefix + "_" + ct + ".csv")
                        fig.savefig(save_prefix + "_" + ct + ".pdf", bbox_inches="tight")

    if not df_total.empty:
        if len(df_total[groups_key].unique()) > 2:
            pd.pivot_table(df_total, values=["log2FoldChange"], index=["index"], columns=[groups_key], fill_value=0)

    return df_total


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
    adata: an.AnnData,
    celltype: str = "ct_musk",
    zone: str = "cc_niches",
    replicate: str = "sample",
    condition: str = "group",
    condition_order: tuple = ["CTRL", "PAH"],  # might be possible to provide more conditions
    top: int = 5,
    figsize: tuple = (6, 3),
):
    """Compute per zone celltype proportion between 2 conditions using replicate for statistical testing

    Parameters
    ----------
    adata
        AnnData object.
    celltype
        celltype key in adata.obs
    zone
        zone key in adata.obs
    replicate
        replicate key in adata.obs
    condition
        condition key in adata.obs
    condition_order
        tuple of the 2 conditions to test
    top
        top celltype to consider
    figsize
        figure size
    Returns
    -------

    """
    sns.set_theme(style="whitegrid", palette="pastel")
    l = list(adata.obs[zone].unique())
    for n in l:
        # print(n)
        df = adata[adata.obs[zone] == n].obs[[replicate, condition, celltype]]
        df2 = df.groupby([replicate, condition, celltype])[celltype].count().unstack()
        df2 = df2.div(df2.sum(axis=1), axis=0).reset_index()
        df2 = df2.melt(id_vars=[replicate, condition])
        df2 = df2.dropna()
        df2 = df2[df2.value > 0]

        hits = list(df[celltype].value_counts().head(top).keys())
        df2 = df2[df2[celltype].isin(hits)]

        pairs = []
        for ct in hits:
            if len(df2[df2[celltype] == ct][condition].unique()) > 1:
                pairs.append([(ct, condition_order[0]), (ct, condition_order[1])])

        subcat_order = hits
        hue_plot_params = {
            "data": df2,
            "x": celltype,
            "y": "value",
            "order": subcat_order,
            "hue": "group",
            "hue_order": condition_order,
            # "palette": pal_group,
        }

        if len(pairs) > 0:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            sns.boxplot(ax=ax, **hue_plot_params, boxprops={"alpha": 0.8}, showfliers=False, linewidth=0.5)
            sns.stripplot(ax=ax, **hue_plot_params, dodge=True, edgecolor="black", linewidth=0.5, size=3)

            annotator = Annotator(ax, pairs, **hue_plot_params)
            annotator.configure(test="Mann-Whitney", text_format="star")
            annotator.apply_and_annotate()

            # When creating the legend, only use the first 2 elements
            handles, labels = ax.get_legend_handles_labels()
            l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=6)
            ax.set_yticklabels(ax.get_yticklabels(), size=6)
            ax.xaxis.grid(True)
            ax.yaxis.grid(True)
            ax.set(ylabel="")
            ax.set_title("zone " + str(n))
            plt.tight_layout()
