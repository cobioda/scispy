import base64
import json
import math
import zlib

import anndata as an
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spatialdata as sd
import squidpy as sq
from clustergrammer2 import CGM2, Network
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as mplPolygon
from observable_jupyter import embed
from skimage import exposure, img_as_float
from spatialdata.transformations import (
    Affine,
    set_transformation,
)

from scispy.io.basic import load_bounds_pixel


def plot_shape_along_axis(
    sdata: sd.SpatialData,
    group_lst: tuple = ["eOSNs", "Deuterosomal"],  # the cell types to consider
    gene_lst: tuple = ["KRT19", "SOX2"],  # the genes to consider
    label_obs_key: str = "celltype_spatial",
    poly_name: str = "polygone_name",
    poly_name_key: str = "name",
    sdata_shape_key: str = "myshapes",
    sdata_group_key: str = "celltypes",
    target_coordinates: str = "microns",
    scale_expr: bool = False,
    bin_size: int = 50,
    rotation_angle: int = 0,
    save: bool = False,
):
    """Analyse a polygon shape stored, rotate it and view celltype or gene expression along x coordinate

    Parameters
    ----------
    sdata
        SpatialData object.
    label_obs_key
        label_key in sdata.table.obs to add
    shape_key
        name of shape layer
    poly_name
        name of polygone in shape layer
    poly_name_key
        where to find poly_name information in geodataframe shape layer
    groups
        group list to consider (related to label_obs_key)
    target_coordinates
        target_coordinates system of sdata object

    """
    if group_lst is None:
        group_lst = sdata.table.obs[label_obs_key].unique().tolist()

    # extract polygon to work with
    # poly = sdata[sdata_shape_key][sdata[sdata_shape_key][poly_name_key] == poly_name].geometry.item()
    # sdata2 = sd.polygon_query(
    #    sdata,
    #    poly,
    #    target_coordinate_system=target_coordinates,
    #    filter_table=True,
    #    points=True,
    #    shapes=True,
    #    images=True,
    # )

    # rotate the shape along x axis
    if rotation_angle != 0:
        theta = math.pi / (360 / rotation_angle)
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

        elements = list(sdata.points.keys()) + list(sdata.shapes.keys())
        for i in range(0, len(elements)):
            set_transformation(sdata[elements[i]], rotation, to_coordinate_system=target_coordinates)

    # convert to final coordinates
    sdata3 = sdata.transform_to_coordinate_system(target_coordinates)

    # get elements key, probably need to do better here in the futur !!
    sdata_transcript_key = sdata3.table.obs.dataset_id.unique().tolist()[0] + "_transcripts"
    sdata_polygon_key = sdata3.table.obs.dataset_id.unique().tolist()[0] + "_polygons"

    # compute dataframes
    df_transcripts = sdata3[sdata_transcript_key].compute()
    df_celltypes = sdata3[sdata_group_key].compute()

    # parametrage
    x_min = df_transcripts.x.min()
    total_x = df_transcripts.x.max() - df_transcripts.x.min()
    step_number = int(total_x / bin_size)

    # init color palette
    cats = sdata.table.obs[label_obs_key].cat.categories.tolist()
    colors = list(sdata.table.uns[label_obs_key + "_colors"])
    mypal = dict(zip(cats, colors))
    mypal = {x: mypal[x] for x in group_lst}

    # compute values dataframes
    vals = pd.DataFrame({"microns": [], "count": [], "gene": []})
    for g in range(0, len(gene_lst)):
        df2 = df_transcripts[df_transcripts.gene == gene_lst[g]]
        df2.shape[0] / step_number
        for i in range(0, step_number):
            new_row = {
                "microns": (x_min + (i + 0.5) * bin_size),
                "count": df2[(df2.x > (x_min + i * bin_size)) & (df2.x < (x_min + (i + 1) * bin_size))].shape[0],
                "gene": gene_lst[g],
            }
            vals = pd.concat([vals, pd.DataFrame([new_row])], ignore_index=True)

    valct = pd.DataFrame({"microns": [], "count": [], "cell_type": []})
    for ct in range(0, len(group_lst)):
        df2 = df_celltypes[df_celltypes[label_obs_key] == group_lst[ct]]
        for i in range(0, step_number):
            new_row = {
                "microns": (x_min + (i + 0.5) * bin_size),
                "count": df2[(df2.x > (x_min + i * bin_size)) & (df2.x < (x_min + (i + 1) * bin_size))].shape[0],
                "cell_type": group_lst[ct],
            }
            valct = pd.concat([valct, pd.DataFrame([new_row])], ignore_index=True)

    # normalization
    means_stds = vals.groupby(["gene"])["count"].agg(["mean", "std", "max"]).reset_index()
    vals = vals.merge(means_stds, on=["gene"])
    vals["norm_count"] = vals["count"] / vals["max"]

    # draw figure
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    sdata3.pl.render_shapes(
        elements=sdata_polygon_key, color=label_obs_key, palette=list(mypal.values()), groups=group_lst
    ).pl.show(ax=ax1)

    sns.lineplot(data=valct, x="microns", y="count", hue="cell_type", linewidth=0.9, palette=mypal, ax=ax2)
    ax2.get_legend().remove()

    if scale_expr is True:
        sns.lineplot(
            data=vals,
            x="microns",
            y="norm_count",
            hue="gene",
            linewidth=0.9,
            palette=sns.color_palette("Paired"),
            ax=ax3,
        )
    else:
        sns.lineplot(
            data=vals, x="microns", y="count", hue="gene", linewidth=0.9, palette=sns.color_palette("Paired"), ax=ax3
        )

    ax3.legend(bbox_to_anchor=(1, 1.1))
    ax1.set_title(poly_name + " shape (" + sdata.table.obs.slide.unique().tolist()[0] + ")")
    ax2.set_title("Number of cells per cell types (" + str(int(bin_size)) + " µm bins)")
    ax3.set_title("Gene expression (" + str(int(bin_size)) + " µm bins)")

    plt.tight_layout()

    # save figure
    if save is True:
        print("saving " + poly_name + ".pdf")
        plt.savefig(poly_name + ".pdf", format="pdf", bbox_inches="tight")


def view_region(
    adata: an.AnnData,
    color_key: str,
    geneNames: tuple,
    pt_size: float,
    x: int,
    y: int,
    size_x: int,
    size_y: int,
    show_loc: bool = True,
    lw: float = 1,
    image_cmap: str = "viridis",
    fill_polygons: bool = True,
    all_transcripts: bool = False,
    noimage: bool = False,
    save: bool = False,
    mypal: dict = None,
    highlight_top: int = None,
    cat_kept: str = None,
    figsize: tuple = (8, 8),
) -> an.AnnData:
    """Scis crop region plot.

    Parameters
    ----------
    adata
        Anndata object.

    Returns
    -------
    Return the AnnData object of the crop region.
    """
    library_id = adata.obs.library_id.cat.categories.tolist()[0]
    img = sq.im.ImageContainer(adata.uns["spatial"][library_id]["images"]["hires"], library_id=library_id)

    if (img.shape[0] < y + size_y) or (img.shape[1] < x + size_x):
        print("Crop outside range img width/height = ", img.shape)
        return 0
    else:
        crop1 = img.crop_corner(y, x, size=(size_y, size_x))
        adata_crop = crop1.subset(adata, spatial_key="pixel")

        if adata_crop.shape[0] == 0:
            print("No cells found in region [x=", x, ", y=", y, ", size_x=", size_x, ", size_y=", size_y)
            return adata_crop
        else:
            print(adata_crop.shape[0], "cells to plot")

            # select categories to plot, ghost all others as 'others'
            if cat_kept is not None:
                lst = adata_crop.obs[color_key].value_counts().keys().tolist()
                tokeep = cat_kept
                toghost = list(set(lst) - set(cat_kept))
                my_dict = {}
                for _index, element in enumerate(tokeep):
                    my_dict[element] = element
                for _index, element in enumerate(toghost):
                    my_dict[element] = "others"
                adata_crop.obs[color_key] = adata_crop.obs[color_key].map(my_dict)
                adata_crop.uns.pop(color_key + "_colors", None)
                adata_crop.obs[color_key] = adata_crop.obs[color_key].astype("category")
            if highlight_top is not None:
                lst = adata_crop.obs[color_key].value_counts().keys().tolist()
                tokeep = lst[0:highlight_top]
                toghost = lst[highlight_top : len(lst)]
                my_dict = {}
                for _index, element in enumerate(tokeep):
                    my_dict[element] = element
                for _index, element in enumerate(toghost):
                    my_dict[element] = "others"
                adata_crop.obs[color_key] = adata_crop.obs[color_key].map(my_dict)
                adata_crop.uns.pop(color_key + "_colors", None)
                adata_crop.obs[color_key] = adata_crop.obs[color_key].astype("category")

            print("load_bounds_pixel start")

            # load cell boundaries --> need to have this in anndata object somewhere : 'obsm' ?
            adata_crop = load_bounds_pixel(adata_crop, library_id)

            print("load_bounds_pixel end")

            currentCells = []
            typedCells = []
            for cell_id in adata_crop.obs.index:
                currentCells.append(adata_crop.obs.bounds[cell_id])
                typedCells.append(adata_crop.obs[color_key][cell_id])

            # minCoord = np.min([np.min(x, axis=1) for x in currentCells], axis=0).astype(int)
            # maxCoord = np.max([np.max(x, axis=1) for x in currentCells], axis=0).astype(int)

            # segmentation data
            polygon_data = []
            for inst_index in range(len(currentCells)):
                inst_cell = currentCells[inst_index]
                df_poly_z = pd.DataFrame(inst_cell).transpose()
                df_poly_z.columns.tolist()

                xxx = df_poly_z.values - [x, y]
                inst_poly = np.array(xxx.tolist())
                polygon_data.append(inst_poly)

            print("polygon_data end")

            # generate colors for categories by plotting
            cats = adata_crop.obs[color_key].cat.categories.tolist()
            if mypal is None:
                colors = list(adata_crop.uns[color_key + "_colors"])
                mypal = dict(zip(cats, colors))

            ser_color = pd.Series(mypal)

            if noimage is True:
                plt.style.use("dark_background")

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, gridspec_kw={"width_ratios": [1, 10]})

            # fig, ax = plt.subplots(figsize=figsize)
            # sns.scatterplot(x='x_pix', y='y_pix', data=adata_crop.obs, s=0.5, hue = color_key, ax=axs[0])
            # sq.pl.spatial_scatter(adata, color=color_key, shape=None, size=1, ax=axs[0], palette=ListedColormap(list(colors.values())))

            if show_loc is True:
                sq.pl.spatial_scatter(adata, color=color_key, shape=None, size=1, ax=axs[0])

            axs[0].add_patch(
                patches.Rectangle((x, y), size_x, size_y, ls="--", edgecolor="black", facecolor="none", linewidth=1)
            )
            # axs[0].get_legend().remove()
            title = library_id + "[x=" + str(x) + ",y=" + str(y) + "]"
            axs[0].set_title(title, fontsize=6)
            # axs[0].grid(True)
            axs[0].get_xaxis().set_visible(False)
            axs[0].get_yaxis().set_visible(False)
            axs[0].plot(x, y, "go")

            if noimage is False:
                crop1.show(layer="image", ax=axs[1], cmap=image_cmap)

            # -- 1. plot cells --#
            pts = adata_crop.obs[["x_pix", "y_pix", color_key]]
            pts.x_pix -= x
            pts.y_pix -= y
            axs[1] = sns.scatterplot(
                x="x_pix", y="y_pix", s=0, alpha=0.5, edgecolors=color_key, data=pts, hue=color_key, palette=mypal
            )

            # -- 2. plot polygons fill --#
            polygons = [mplPolygon(polygon_data[i].reshape(-1, 2)) for i in range(len(polygon_data))]
            if fill_polygons is True:
                axs[1].add_collection(PatchCollection(polygons, fc=ser_color[typedCells], ec="None", alpha=0.5))

            # -- 3. plot transcripts --#
            transcripts = adata_crop.uns["transcripts"][library_id]
            tr = transcripts[
                (transcripts.x_pix > x)
                & (transcripts.x_pix < x + size_x)
                & (transcripts.y_pix > y)
                & (transcripts.y_pix < y + size_y)
            ]
            tr.loc[:, "x_pix"] = tr.loc[:, "x_pix"] - x
            tr.loc[:, "y_pix"] = tr.loc[:, "y_pix"] - y
            pts = tr[["x_pix", "y_pix"]].to_numpy()

            if all_transcripts:
                axs[1].scatter(pts[:, 0], pts[:, 1], marker=".", color="grey", s=pt_size, label="all")

            i = 0
            cols = sns.color_palette("bright", len(geneNames))
            for gn in geneNames:
                tr2 = tr[tr.gene == gn]
                pts = tr2[["x_pix", "y_pix"]].to_numpy()
                pts = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0) & (pts[:, 0] < size_x) & (pts[:, 1] < size_y)]
                axs[1].scatter(pts[:, 0], pts[:, 1], marker="o", color=cols[i], s=pt_size, label=gn)
                i = i + 1

            # -- 4. plot polygons edges --#
            axs[1].add_collection(PatchCollection(polygons, fc="none", ec=ser_color[typedCells], lw=lw))

            h, l = axs[1].get_legend_handles_labels()
            l1 = axs[1].legend(h[: len(cats)], l[: len(cats)], loc="upper right", bbox_to_anchor=(1, 1), fontsize=6)
            axs[1].legend(h[len(cats) :], l[len(cats) :], loc="upper left", bbox_to_anchor=(0, 1), fontsize=6)
            axs[1].add_artist(l1)  # we need this because the 2nd call to legend() erases the first
            axs[1].set_title("")
            plt.tight_layout()

            if save is True:
                print("saving " + title + ".pdf")
                plt.savefig(title + ".pdf", format="pdf", bbox_inches="tight")

    return adata_crop


def get_regions(
    adata: an.AnnData, df: pd.DataFrame, color_key: str = "celltype", pt_size: float = 20, save: bool = False
) -> an.AnnData:
    """Scis get region plot.

    Parameters
    ----------
    adata
        Anndata object.
    df
        dataframe of regions.

    Returns
    -------
    Return the concatenate AnnData object of the regions.
    """
    library_id = adata.obs.library_id.cat.categories.tolist()[0]
    img = sq.im.ImageContainer(adata.uns["spatial"][library_id]["images"]["hires"], library_id=library_id)

    fig, axs = plt.subplots(nrows=1, ncols=df.shape[0], figsize=(16, 4))

    for index, row in df.iterrows():
        x = row.x
        y = row.y
        size_x = row.size_x
        size_y = row.size_y

        if (img.shape[0] < y + size_y) or (img.shape[1] < x + size_x):
            print("Crop outside range img width/height = ", img.shape)
            return 0
        else:
            crop1 = img.crop_corner(y, x, size=(size_y, size_x))
            adata_crop = crop1.subset(adata, spatial_key="pixel")
            adata_crop.obs["zone"] = row.zone
            sq.pl.spatial_scatter(adata_crop, color=color_key, shape=None, size=pt_size, ax=axs[index])
            axs[index].set_title(row.zone, fontsize=12)
            if index < df.shape[0] - 1:
                axs[index].get_legend().remove()
            if index == 0:
                concat = adata_crop.copy()
            else:
                concat = concat.concatenate(adata_crop)

    plt.tight_layout()

    if save is True:
        print("saving " + library_id + "_regions.pdf")
        plt.savefig(library_id + "_regions.pdf", format="pdf", bbox_inches="tight")

    return concat


def view_qc(adata: an.AnnData, library_id: str) -> int:
    """Scinsit quality control plot.

    Parameters
    ----------
    adata
        Anndata object.
    library_id
        library id.

    Returns
    -------
    Return the Anndata object of the crop region.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.subplot(2, 2, 1)
    bins = np.logspace(0, 4, 100)
    plt.hist(adata.obs["volume"], alpha=0.2, bins=bins, label=library_id, color="red")
    plt.xlabel("Volume")
    plt.ylabel("Cell count")
    plt.xscale("log")
    # Transcript count by cell
    plt.subplot(2, 2, 2)
    bins = np.logspace(0, 4, 100)
    plt.hist(adata.obs["barcodeCount"], alpha=0.2, bins=bins, label=library_id, color="red")
    plt.xlabel("Transcript count")
    plt.ylabel("Cell count")
    plt.xscale("log")
    plt.yscale("log")
    plt.subplot(2, 2, 3)
    barcodeCount = adata.obs["barcodeCount"]
    sns.distplot(barcodeCount, label=library_id, color="red")
    ax1 = plt.subplot(2, 2, 4)
    sc.pl.violin(adata, keys="barcodeCount", ax=ax1)

    return 0


def plot_compartments(
    adata: an.AnnData,
    celltypelabel: str = "cell_type",
    metalabel: str = "population",
    pt_size: float = 2,
    figsize: tuple = (20, 5),
    zoom: tuple = None,
    mypal: dict = None,
):
    """Plot cells by compartment (i.e. population).

    Parameters
    ----------
    adata
        Anndata object.
    zoom
        [x, y, x_size, y_size].

    Returns
    -------
    Return the Anndata object of the crop region.
    """
    if zoom is not None:
        library_id = adata.obs.library_id.cat.categories.tolist()[0]
        img = sq.im.ImageContainer(adata.uns["spatial"][library_id]["images"]["hires"], library_id=library_id)
        if (img.shape[0] < zoom[1] + zoom[3]) or (img.shape[1] < zoom[0] + zoom[2]):
            print("Crop outside range img width/height = ", img.shape)
            return 0
        else:
            crop1 = img.crop_corner(zoom[1], zoom[0], size=(zoom[3], zoom[2]))
            adata = crop1.subset(adata, spatial_key="pixel")

    maliste = adata.obs[metalabel].cat.categories.tolist()
    fig, axs = plt.subplots(1, len(maliste), figsize=figsize)
    for i in range(len(maliste)):
        ad = adata[adata.obs[metalabel].isin([maliste[i]])]
        if mypal is None:
            sns.scatterplot(x="x_pix", y="y_pix", data=ad.obs, s=pt_size, hue=celltypelabel, ax=axs[i]).set_title(
                maliste[i]
            )
        else:
            sns.scatterplot(
                x="x_pix", y="y_pix", data=ad.obs, s=pt_size, hue=celltypelabel, ax=axs[i], palette=mypal
            ).set_title(maliste[i])
        plt.setp(axs[i].get_legend().get_texts(), fontsize="6")


def cluster_small_multiples(adata, clust_key, size=60, frameon=False, legend_loc=None, **kwargs):
    """cluster_small_multiples

    Parameters
    ----------
    adata
        Anndata object.
    clust_key
        key to plot

    Returns
    -------
    Return the Anndata object of the crop region.
    """
    tmp = adata.copy()

    for i, clust in enumerate(adata.obs[clust_key].cat.categories):
        tmp.obs[clust] = adata.obs[clust_key].isin([clust]).astype("category")
        tmp.uns[clust + "_colors"] = ["#d3d3d3", adata.uns[clust_key + "_colors"][i]]

    sc.pl.umap(
        tmp,
        groups=tmp.obs[clust].cat.categories[1:].values,
        color=adata.obs[clust_key].cat.categories.tolist(),
        size=size,
        frameon=frameon,
        legend_loc=legend_loc,
        **kwargs,
    )


def json_zip(j):
    """Json zipper"""
    zip_json_string = base64.b64encode(zlib.compress(json.dumps(j).encode("utf-8"))).decode("ascii")
    return zip_json_string


def embed_vizgen(adata: an.AnnData, color_key: str):
    """Embed vizgen assay (Nicolas Fernandez credit).

    Parameters
    ----------
    adata
        Anndata object.
    color_key
        color key.

    Returns
    -------
    Return the Anndata object of the crop region.
    """
    cats = adata.obs[color_key].cat.categories.tolist()
    colors = list(adata.uns[color_key + "_colors"])
    cat_colors = dict(zip(cats, colors))
    ser_color = pd.Series(cat_colors)
    ser_color.name = "color"
    df_colors = pd.DataFrame(ser_color)
    df_colors.index = [str(x) for x in df_colors.index.tolist()]

    ser_counts = adata.obs[color_key].value_counts()
    ser_counts.name = "cell counts"
    meta_leiden = pd.DataFrame(ser_counts)
    sig_leiden = pd.DataFrame(columns=adata.var_names, index=adata.obs[color_key].cat.categories)
    for clust in adata.obs[color_key].cat.categories:
        sig_leiden.loc[clust] = adata[adata.obs[color_key].isin([clust]), :].X.mean(0)

    sig_leiden = sig_leiden.transpose()
    leiden_clusters = [str(x) for x in sig_leiden.columns.tolist()]
    sig_leiden.columns = leiden_clusters
    meta_leiden.index = sig_leiden.columns.tolist()
    meta_leiden[color_key] = pd.Series(meta_leiden.index.tolist(), index=meta_leiden.index.tolist())

    net = Network(CGM2)
    # net.load_df(sig_leiden, meta_col=meta_leiden, col_cats=['cell counts'])
    net.load_df(
        sig_leiden,
        meta_col=meta_leiden,
        col_cats=[color_key, "cell counts"],
        meta_row=adata.var,
        row_cats=["mean", "expression"],
    )
    net.filter_threshold(0.01, axis="row")
    net.normalize(axis="row", norm_type="zscore")
    net.set_global_cat_colors(df_colors)
    net.cluster()

    gex_int = pd.DataFrame(adata.layers["counts"], index=adata.obs.index.copy(), columns=adata.var_names).astype(np.int)
    gex_dict = {}
    for inst_gene in gex_int.columns.tolist():
        if "Blank" not in inst_gene:
            ser_gene = gex_int[inst_gene]
            ser_gene = ser_gene[ser_gene > 0]
            ser_gene = ser_gene.astype(np.int8)
            gex_dict[inst_gene] = ser_gene.to_dict()

    df_pos = adata.obs[["center_x", "center_y", color_key]]
    df_pos[["center_x", "center_y"]] = df_pos[["center_x", "center_y"]].round(2)
    df_pos.columns = ["x", "y", "leiden"]
    df_pos["y"] = -df_pos["y"]
    df_umap = adata.obsm.to_df()[["X_umap1", "X_umap2"]].round(2)
    df_umap.columns = ["umap-x", "umap-y"]

    # rotate the mouse brain to the upright position
    # theta = np.deg2rad(-15)
    # rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    # df_pos[['x', 'y']] = df_pos[['x', 'y']].dot(rot)

    df_name = pd.DataFrame(df_pos.index.tolist(), index=df_pos.index.tolist(), columns=["name"])

    df_obs = pd.concat([df_name, df_pos, df_umap], axis=1)
    data = df_obs.to_dict("records")

    obs_data = {"gex_dict": gex_dict, "data": data, "cat_colors": cat_colors, "network": net.viz}

    zip_obs_data = json_zip(obs_data)

    inputs = {
        "zoom": -3.5,
        "ini_cat": color_key,
        "ini_map_type": "UMAP",
        "ini_min_radius": 1.75,
        "zip_obs_data": zip_obs_data,
        "gex_opacity_contrast_scale": 0.85,
    }

    embed(
        "@vizgen/umap-spatial-heatmap-single-cell-0-3-0",
        cells=["viewof cgm", "dashboard"],
        inputs=inputs,
        display_logo=False,
    )


def getFovCoordinates(fov: int, meta_cell: pd.DataFrame) -> tuple:
    """Return fov coordinates

    Parameters
    ----------
    fov
        fov number.
    meta_cell
        cell metadata.

    Returns
    -------
    Return fov coordinates.
    """
    xmin = meta_cell.x[meta_cell.fov == fov].min()
    ymin = meta_cell.y[meta_cell.fov == fov].min()
    xmax = meta_cell.x[meta_cell.fov == fov].max()
    ymax = meta_cell.y[meta_cell.fov == fov].max()

    return (xmin, ymin, xmax, ymax)


def plot_img_and_hist(image: np.ndarray, axes: int, bins: int = 256):
    """Plot an image along with its histogram and cumulative histogram."""
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype="step", color="black")
    ax_hist.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax_hist.set_xlabel("Pixel intensity")
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, "r")
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def plot_contrast_panels(img: np.ndarray) -> int:
    """Test different contrasts for image received."""
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

    # Equalization
    img_eq = exposure.equalize_hist(img)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    # Display results
    fig = plt.figure(figsize=(12, 8))
    axes = np.zeros((2, 4), dtype=object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5 + i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title("Original image")

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
    ax_img.set_title("Contrast stretching")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    ax_img.set_title("Histogram equalization")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
    ax_img.set_title("Adaptive equalization")

    ax_cdf.set_ylabel("Fraction of total intensity")
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()

    return 1


def plot_gamma_panels(img: np.ndarray) -> int:
    """Test different correction for image received."""
    # Gamma
    gamma_corrected = exposure.adjust_gamma(img, 2)

    # Logarithmic
    logarithmic_corrected = exposure.adjust_log(img, 1)

    # Display results
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 3), dtype=object)
    axes[0, 0] = plt.subplot(2, 3, 1)
    axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 0] = plt.subplot(2, 3, 4)
    axes[1, 1] = plt.subplot(2, 3, 5)
    axes[1, 2] = plt.subplot(2, 3, 6)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title("Original image")

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(gamma_corrected, axes[:, 1])
    ax_img.set_title("Gamma correction")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(logarithmic_corrected, axes[:, 2])
    ax_img.set_title("Logarithmic correction")

    ax_cdf.set_ylabel("Fraction of total intensity")
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()

    return 1


def plot_adaptive_size_panels(img: np.ndarray, clip_limit: float) -> int:
    """Test different kernel_size for exposure.equalize_adapthist of image received."""
    # Adaptive Equalization
    img_1 = exposure.equalize_adapthist(img, clip_limit=clip_limit)
    img_2 = exposure.equalize_adapthist(img, clip_limit=clip_limit, kernel_size=[100, 100])
    img_3 = exposure.equalize_adapthist(img, clip_limit=clip_limit, kernel_size=[1000, 1000])

    # Display results
    fig = plt.figure(figsize=(12, 8))
    axes = np.zeros((2, 4), dtype=object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5 + i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title("Original image")

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_1, axes[:, 1])
    ax_img.set_title("adapt " + str(clip_limit) + ", default size 1/8")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_2, axes[:, 2])
    ax_img.set_title("adapt " + str(clip_limit) + ", size 100p")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_3, axes[:, 3])
    ax_img.set_title("adapt " + str(clip_limit) + ", size 1kp")

    ax_cdf.set_ylabel("Fraction of total intensity")
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()

    return 1


def plot_adaptive_panels(img: np.ndarray) -> int:
    """Test different clip_limit for exposure.equalize_adapthist of image received."""
    # Adaptive Equalization
    img_001 = exposure.equalize_adapthist(img, clip_limit=0.01)
    img_003 = exposure.equalize_adapthist(img, clip_limit=0.03)
    img_01 = exposure.equalize_adapthist(img, clip_limit=0.1)

    # Display results
    fig = plt.figure(figsize=(12, 8))
    axes = np.zeros((2, 4), dtype=object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5 + i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title("Original image")

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_001, axes[:, 1])
    ax_img.set_title("adapt 0.01")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_003, axes[:, 2])
    ax_img.set_title("adapt 0.03")

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_01, axes[:, 3])
    ax_img.set_title("adapt 0.1")

    ax_cdf.set_ylabel("Fraction of total intensity")
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()

    return 1
