import math

import anndata as an
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spatialdata as sd
from matplotlib import pyplot as plt
from spatialdata.transformations import Affine, set_transformation


def plot_shape_along_axis(
    sdata: sd.SpatialData,
    group_lst: tuple = ["eOSNs", "Deuterosomal"],  # the cell types to consider
    gene_lst: tuple = ["KRT19", "SOX2"],  # the genes to consider
    label_obs_key: str = "celltype_spatial",
    poly_name: str = "polygone_name",
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


def get_palette(color_key: str) -> dict:
    """Scispy palette definition for a specific PAH project.

    Parameters
    ----------
    color_key
        color key from Anndata.obs (might be 'group', 'population' or 'celltype').

    Returns
    -------
    Return the Anndata object of the crop region.
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
            "SMC": "#d81159",
            "AdvFibro": "#ef6351",
            "AlvFibro": "#d58936",
            "MyoFibro": "#69140e",
        }
    elif color_key == "celltype2":
        palette = {
            # paolo
            "cartilage": "#005f73",
            "myeloid": "#0a9396",
            "skeletal muscle": "#94d2bd",
            "stromal": "#e9d8a6",
            "endothelial": "#64a6bd",
            "lymphatic": "#90a8c3",
            "pericyte": "#d7b9d5",
            "vascular": "#f4cae0",
            "basal": "#003049",
            "ciliated": "#d62828",
            "deuterosomal": "#f77f00",
            "secretory": "#fcbf49",
            "excitatory neuron": "#797d62",
            "neuron": "#4a4e69",
            "glia": "#9b9b7a",
            "olfactory sensory neuron": "#d9ae94",
            "satelite": "#ffcb69",
            "sustentacular": "#b58463",
            # per compartment
            #'cartilage nasal': '#fb8500',
            #'vascular lymphatic': '#ef233c',
            #'olfactory epithelium': '#344966',
            #'migrating neuron': '#606c38',
        }
    elif color_key == "leiden":  # default is 40 colors returned
        l = list(range(0, 39, 1))
        ll = list(map(str, l))
        palette = dict(zip(ll, sns.color_palette("husl", 40).as_hex()))

    palette["others"] = "#ffffff"

    return palette


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
