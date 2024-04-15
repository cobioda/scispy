import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spatialdata as sd
import squidpy as sq
from matplotlib import pyplot as plt

from scispy.tl.basic import sdata_rotate


def plot_shapes(
    sdata: sd.SpatialData,
    group_lst: tuple = [],  # the cell types to consider
    shapes_lst: tuple = [],  # the shapes to plot
    label_obs_key: str = "celltype_spatial",
    shape_key: str = "arteries",
    color_dict: tuple = [],
    target_coordinates: str = "microns",
    figsize: tuple = (12, 6),
    save: bool = False,
):
    """Plot list of shapes

    Parameters
    ----------
    sdata
        SpatialData object obtained by tl.get_sdata_polygon()
    group_lst
        group list to consider (related to label_obs_key)
    shapes_lst
        shapes list to plot
    label_obs_key
        label_key in sdata.table.obs to consider
    shape_key
        SpatialData shape element to consider
    color_dict
        dictionary of colors to use
    target_coordinates
        target_coordinates system of sdata object
    figsize
        figure size
    save
        wether or not to save the figure

    """
    region_key = sdata.table.uns["spatialdata_attrs"]["region"]

    fig, axs = plt.subplots(ncols=len(shapes_lst), nrows=1, figsize=figsize)
    for i in range(0, len(shapes_lst)):
        poly = sdata[shape_key][sdata[shape_key].name == shapes_lst[i]].geometry.item()
        sdata2 = sd.polygon_query(
            sdata,
            poly,
            target_coordinate_system=target_coordinates,
            filter_table=True,
            points=False,
            shapes=True,
            images=True,
        )

        sdata2.pl.render_images().pl.show(ax=axs[i])
        if group_lst is None:
            group_lst = sdata.table.obs[label_obs_key].unique().tolist()

        if color_dict is not None:
            mypal = [color_dict[x] for x in group_lst]
            sdata2.pl.render_shapes(elements=region_key, color=label_obs_key, groups=group_lst, palette=mypal).pl.show(
                ax=axs[i]
            )
        else:
            sdata2.pl.render_shapes(elements=region_key, color=label_obs_key, groups=group_lst).pl.show(ax=axs[i])

        axs[i].set_title(shapes_lst[i])
        # if(i < len(shapes_lst)):
        #    axs[i].get_legend().remove()

    plt.tight_layout()


def plot_shape_along_axis(
    sdata: sd.SpatialData,
    group_lst: tuple = [],  # the cell types to consider
    gene_lst: tuple = [],  # the genes to consider
    color_key: str = "celltype",
    sdata_group_key: str = "celltypes",
    target_coordinates: str = "microns",
    palette: tuple = [],
    scale_expr: bool = False,
    bin_size: int = 50,
    rotation_angle: int = 0,
    heatmap: bool = False,
    save: bool = False,
):
    """Analyse cell types occurence and gene expression along x axis of a sdata polygon shape after an evenual rotation

    Parameters
    ----------
    sdata
        SpatialData object obtained by tl.get_sdata_polygon()
    group_lst
        group list to consider (related to label_obs_key)
    gene_lst
        gene list to consider
    color_key
        label_key in sdata.table.obs to consider
    sdata_group_key
        SpatialData element where to find the group label
    target_coordinates
        target_coordinates system of sdata object
    scale_expr
        wether or not to scale the gene expression plot
    bin_size
        size of bins for plotting (µm)
    rotation_angle
        horary rotation angle of the shape before computing along x axis
    save
        wether or not to save the figure
    """
    if group_lst is None:
        group_lst = sdata.table.obs[color_key].unique().tolist()

    sdata_rotate(sdata, rotation_angle)
    sdata2 = sdata.transform_to_coordinate_system(target_coordinates)

    dataset_id = sdata2.table.obs.dataset_id.unique()[0]
    feature_key = sdata2.table.uns["spatialdata_attrs"]["feature_key"]
    polygon_key = sdata2.table.uns["spatialdata_attrs"]["region"]
    transcript_key = list(sdata.points.keys())[0]  # need to be in spatialdata_attrs

    # compute dataframes
    df_transcripts = sdata2[transcript_key].compute()
    df_celltypes = sdata2[sdata_group_key].compute()

    # parametrage
    x_min = df_transcripts.x.min()
    total_x = df_transcripts.x.max() - df_transcripts.x.min()
    step_number = int(total_x / bin_size)

    # init color palette
    cats = sdata.table.obs[color_key].cat.categories.tolist()
    colors = list(sdata.table.uns[color_key + "_colors"])
    mypal = dict(zip(cats, colors))
    if palette is not None:
        mypal = palette

    mypal = {x: mypal[x] for x in group_lst}

    # compute values dataframes
    vals = pd.DataFrame({"microns": [], "count": [], feature_key: []})
    for g in range(0, len(gene_lst)):
        df2 = df_transcripts[df_transcripts[feature_key] == gene_lst[g]]
        df2.shape[0] / step_number
        for i in range(0, step_number):
            new_row = {
                "microns": (x_min + (i + 0.5) * bin_size),
                "count": df2[(df2.x > (x_min + i * bin_size)) & (df2.x < (x_min + (i + 1) * bin_size))].shape[0],
                feature_key: gene_lst[g],
            }
            vals = pd.concat([vals, pd.DataFrame([new_row])], ignore_index=True)

    valct = pd.DataFrame({"microns": [], "count": [], "celltype": []})
    for ct in range(0, len(group_lst)):
        df2 = df_celltypes[df_celltypes["ct"] == group_lst[ct]]
        for i in range(0, step_number):
            new_row = {
                "microns": (x_min + (i + 0.5) * bin_size),
                "count": df2[(df2.x > (x_min + i * bin_size)) & (df2.x < (x_min + (i + 1) * bin_size))].shape[0],
                "celltype": group_lst[ct],
            }
            valct = pd.concat([valct, pd.DataFrame([new_row])], ignore_index=True)

    # draw figure
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

    sdata.pl.render_shapes(
        elements=polygon_key, color=color_key, palette=list(mypal.values()), groups=group_lst
    ).pl.show(ax=ax1)

    sns.lineplot(data=valct, x="microns", y="count", hue="celltype", linewidth=0.9, palette=mypal, ax=ax2)
    ax2.get_legend().remove()

    if scale_expr is True:
        means_stds = vals.groupby([feature_key])["count"].agg(["mean", "std", "max"]).reset_index()
        vals = vals.merge(means_stds, on=[feature_key])
        vals["norm_count"] = vals["count"] / vals["max"]

        if heatmap is True:
            sns.set(font_scale=0.5)
            sns.heatmap(vals.pivot(index="gene", columns="microns", values="norm_count"), cmap="viridis", ax=ax3)
        else:
            sns.lineplot(
                data=vals,
                x="microns",
                y="norm_count",
                hue=feature_key,
                linewidth=0.9,
                palette=sns.color_palette("Paired"),
                ax=ax3,
            )
            ax3.legend(bbox_to_anchor=(1, 1.1))
    else:
        if heatmap is True:
            sns.heatmap(vals.pivot(index="gene", columns="microns", values="count"), cmap="viridis", ax=ax3)
        else:
            sns.lineplot(
                data=vals,
                x="microns",
                y="count",
                hue=feature_key,
                linewidth=0.9,
                palette=sns.color_palette("Paired"),
                ax=ax3,
            )
            ax3.legend(bbox_to_anchor=(1, 1.1))

    ax1.set_title(dataset_id)
    ax2.set_title("Number of cells per cell types (" + str(int(bin_size)) + " µm bins)")
    ax3.set_title("Gene expression (" + str(int(bin_size)) + " µm bins)")
    plt.tight_layout()

    # save figure
    if save is True:
        print("saving " + dataset_id + ".pdf")
        plt.savefig(dataset_id + ".pdf", format="pdf", bbox_inches="tight")


def plot_sdata(
    sdata: sd.SpatialData,
    color_key: str = "celltype",
    feature_key: str = "gene",  # or feature_name for xenium
    point_size: int = 1,
    figsize: tuple = (12, 6),
    outline: bool = False,
    outline_width: float = 1.0,
    outline_color: str = "red",
    shape_keys: str = None,
    shape_palette: tuple = None,
    shape_groups: tuple = None,  # need shape_palette to be defined
    cmap: tuple = None,
    point_groups: tuple = None,  # 10 maximum
    grid: bool = False,
    save: bool = False,
    save_format: str = "pdf",
):
    """Plot sdata object (i.e. embedding and polygons). This should always works if well synchronized sdata object

    Parameters
    ----------
    sdata
        SpatialData object.
    color_key
        color key from .table.obs
    """
    if shape_keys is None:
        shape_keys = list(sdata.shapes.keys())  # better would be to get the list of spatialdata_attrs 'cell_regions'
    if feature_key is None:
        feature_key = sdata.tables[list(sdata.tables.keys())[0]].uns["spatialdata_attrs"]["feature_key"]

    args_shapes = {"elements": shape_keys, "color": color_key}
    args_points = {"size": point_size, "color": feature_key}

    # size=0.01, linewidth=None, marker=".", edgecolor = 'none', markeredgewidth=0.0

    fig, ax = plt.subplots(figsize=figsize)
    if shape_palette is not None:
        args_shapes["palette"] = list(shape_palette.values())
        args_shapes["groups"] = list(shape_palette.keys())
        if shape_groups is not None:
            mypal = {x: shape_palette[x] for x in shape_groups}
            args_shapes["palette"] = list(mypal.values())
            args_shapes["groups"] = list(mypal.keys())
    if outline is True:
        args_shapes["outline"] = True
        args_shapes["outline_width"] = outline_width
        args_shapes["outline_color"] = outline_color
    if cmap is not None:  # case on continuous value like gene expression or .obs metadata
        args_shapes["cmap"] = cmap

    sdata.pl.render_shapes(**args_shapes).pl.show(ax=ax)

    if point_groups is not None:
        point_dict = get_palette("fluo")
        if len(point_groups) > 10:
            point_groups = point_groups[0:10]
        point_pal = list(point_dict.values())[0 : len(point_groups)]
        sdata.pl.render_points(**args_points, groups=point_groups, palette=point_pal).pl.show(ax=ax)

    if grid is True:
        ax.grid(which="both", linestyle="dashed")
        ax.minorticks_on()
        ax.tick_params(which="minor", bottom=False, left=False)

    legend_without_duplicate_labels(ax)
    plt.tight_layout()

    # save figure
    if save is True:
        if save_format == "pdf":
            print("saving plot_sdata.pdf")
            plt.savefig("plot_sdata.pdf", bbox_inches="tight")
        elif save_format == "png":
            print("saving plot_sdata.png")
            plt.savefig("plot_sdata.png", dpi=300, bbox_inches="tight")


def plot_multi_sdata(
    sdata: sd.SpatialData,
    color_key: str = "celltype",
    save: bool = False,
):
    """Plot sdata object (i.e. embedding and polygons). This should always works if well synchronized sdata object

    Parameters
    ----------
    sdata
        SpatialData object.
    color_key
        color key from .table.obs
    """
    sdata.table.obs[color_key] = sdata.table.obs[color_key].cat.remove_unused_categories()
    sdata.table.uns.pop(color_key + "_colors", None)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    sns.scatterplot(x="center_x", y="center_y", data=sdata.table.obs, s=1, hue=color_key, ax=ax1)
    ax1.axis("equal")
    ax1.get_legend().remove()
    ax1.set_title("anndata.obs coordinates")
    ax1.invert_yaxis()

    sdata.pl.render_shapes(elements=sdata.table.uns["spatialdata_attrs"]["region"], color=color_key).pl.show(ax=ax2)
    ax2.get_legend().remove()
    ax2.set_title("spatialdata polygons")

    sq.pl.spatial_scatter(sdata.table, color=color_key, shape=None, size=1, ax=ax3)
    # ax2.get_legend().remove()
    ax3.set_title("squidpy spatial")

    plt.tight_layout()

    # save figure
    if save is True:
        print("saving plot_multi_sdata.pdf")
        plt.savefig("plot_multi_sdata.pdf", format="pdf", bbox_inches="tight")


def get_palette(color_key: str) -> dict:
    """Palette definition for specific projects.

    Parameters
    ----------
    color_key
        color key (might be 'group', 'population' or 'celltype').

    Returns
    -------
    Return palette dictionary.
    """
    if color_key == "group":
        palette = {"CTRL": "#006E82", "PAH": "#AA0A3C"}
    elif color_key == "population":
        palette = {"Endothelial": "#0077b6", "Epithelial": "#5e548e", "Immune": "#606c38", "Stroma": "#bb3e03"}
    elif color_key == "compartment":
        palette = {
            "cartilage nasal": "#fb8500",
            "vascular lymphatic": "#ef233c",
            "olfactory epithelium": "#344966",
            "migrating neuron": "#606c38",
        }
    elif color_key == "celltype":
        palette = {
            # htap
            "Basal": "#7209b7",
            "Multiciliated": "#b5179e",
            "Neuroendocrine": "#d0d1ff",
            "Secretory": "#9d4edd",
            "AT0": "#e0aaff",
            "AT2": "#6a4c93",
            "AT1": "#4d194d",
            "Lymphatic": "#124e78",
            "aCap": "#00bbf9",
            "gCap": "#0466c8",
            "ArtEC": "#6096ba",
            "VeinEC": "#657ed4",
            "AlvMacro": "#ffd29d",
            "Dendritic": "#d6ce93",
            "Monocyte": "#b1cc74",
            "InterMacro": "#38b000",
            "CD4": "#7c6a0a",
            "CD8": "#bcbd8b",
            "NK": "#e8fcc2",
            "Mast": "#4f6d7a",
            "Plasma": "#829399",
            "B": "#fffbbd",
            "Megak": "#006400",
            "Pericyte": "#9c6644",
            "Pericytes": "#9c6644",
            "SMC": "#d81159",
            "AdvFibro": "#ef6351",
            "AlvFibro": "#d58936",
            "MyoFibro": "#69140e",
        }
    elif color_key == "celltype paolo":
        # paolo
        palette = {
            "Early OSNs": "#18DED2",
            "Sustentaculars": "#C09ACA",
            "HBCs": "#E41A1C",
            "GBCs": "#F48B5A",
            "Precursors": "#706fd3",
            "Duct/MUC": "#efe13c",
            "Multiciliated": "#1f618d",
            "Deuterosomal": "#3498db",
            "GnRH neurons": "#E00EF1",
            "Migratory neurons": "#457b9d",
            "Neuron progenitors": "#6A0B78",
            "Excitatory neurons": "#61559E",
            "Inhibitory neurons": "#800EF1",
            "Neurons ALK+": "#95819F",
            "VNO neurons": "#0D16C8",
            "junk neurons": "#f1faee",
            "Olf. ensh. glia": "#AE8C0D",
            "Glia progenitors": "#E3D9AC",
            "Schwann cells": "#C0AC51",
            "Myeloid": "#736376",
            "Microglia": "#91BFB7",
            "Vascular EC": "#E788C2",
            "Lymphatic EC": "#F78896",
            "Satellites": "#CB7647",
            "Skeletal muscle": "#926B54",
            "Stromal0": "#99D6A9",
            "Stromal1": "#1B8F76",
            "Stromal2": "#9DAF07",
            "Stromal3": "#4CAD4C",
            "Pericytes": "#BBD870",
            "Cartilages": "#0B4B19",
        }
    elif color_key == "population paolo":
        palette = {
            "Olfactory epithelium": "#EF1B4F",
            "Respiratory epithelium": "#5562B7",
            "Neurons": "#6E5489",
            "Glia": "#919976",
            "Immune": "#2EECDB",
            "Endothelium": "#CC79A7",
            "Myocytes": "#803800",
            "Stroma": "#736376",
        }
    elif color_key == "fluo":
        palette = {
            "blue": "#382aff",
            "green": "#82ff78",
            "purple": "#a900d7",
            "orange": "#ffa421",
            "darkblue": "#006b94",
            "red": "#c70015",
            "cyan": "#00b5b9",
            "brown": "#954600",
            "yellow": "#e9ffae",
            "pink": "#ff9eda",
        }
    elif color_key == "leiden":  # default is 40 colors returned
        l = list(range(0, 39, 1))
        ll = list(map(str, l))
        palette = dict(zip(ll, sns.color_palette("husl", 40).as_hex()))

    palette["others"] = "#ffffff"

    return palette


def plot_qc(sdata: sd.SpatialData):
    """Plot quality control analysis.

    Parameters
    ----------
    sdata
        SpatialData object.

    """
    dataset_id = sdata.table.obs.dataset_id.unique().tolist()[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.subplot(2, 2, 1)
    bins = np.logspace(0, 4, 100)
    plt.hist(sdata.table.obs["volume"], alpha=0.2, bins=bins, label=dataset_id, color="red")
    plt.xlabel("Volume")
    plt.ylabel("Cell count")
    plt.xscale("log")
    # Transcript count by cell
    plt.subplot(2, 2, 2)
    bins = np.logspace(0, 4, 100)
    plt.hist(sdata.table.obs["transcript_count"], alpha=0.2, bins=bins, label=dataset_id, color="red")
    plt.xlabel("Transcript count")
    plt.ylabel("Cell count")
    plt.xscale("log")
    plt.yscale("log")
    plt.subplot(2, 2, 3)
    barcodeCount = sdata.table.obs["transcript_count"]
    sns.distplot(barcodeCount, label=dataset_id, color="red")
    ax1 = plt.subplot(2, 2, 4)
    sc.pl.violin(sdata.table.obs, keys="transcript_count", ax=ax1)
    plt.tight_layout()


def plot_per_groups(adata, clust_key, size=60, is_spatial=False, frameon=False, legend_loc=None, **kwargs):
    """Plot UMAP splitted by clust_key

    Parameters
    ----------
    adata
        Anndata object.
    clust_key
        key to plot
    is_spatial
        UMAP plot if False,

    """
    tmp = adata.copy()

    for i, clust in enumerate(adata.obs[clust_key].cat.categories):
        tmp.obs[clust] = adata.obs[clust_key].isin([clust]).astype("category")
        tmp.uns[clust + "_colors"] = ["#d3d3d3", adata.uns[clust_key + "_colors"][i]]

    if is_spatial is False:
        sc.pl.umap(
            tmp,
            groups=tmp.obs[clust].cat.categories[1:].values,
            color=adata.obs[clust_key].cat.categories.tolist(),
            size=size,
            frameon=frameon,
            legend_loc=legend_loc,
            **kwargs,
        )
    else:
        # not working !....
        tmp.uns["spatial"] = tmp.obsm["spatial"]
        sq.pl.spatial_scatter(
            tmp,
            groups=tmp.obs[clust].cat.categories[1:].values,
            color=adata.obs[clust_key].cat.categories.tolist(),
            size=size,
            frameon=frameon,
            legend_loc=legend_loc,
            **kwargs,
        )


def legend_without_duplicate_labels(figure):
    """Remove duplicated labels in figure legend

    Parameters
    ----------
    figure
        matplotlib figure.
    """
    # code from here
    # https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=6, ncol=1)
