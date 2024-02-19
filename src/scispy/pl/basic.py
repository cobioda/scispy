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
    target_coordinates
        target_coordinates system of sdata object
    figsize
        figure size
    save
        wether or not to save the figure

    """
    region_key = sdata.table.uns["spatialdata_attrs"]["region"]

    if group_lst is None:
        group_lst = sdata.table.obs[label_obs_key].unique().tolist()

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
        sdata2.pl.render_shapes(elements=shape_key, outline=True, fill_alpha=0.25, outline_color="red").pl.show(
            ax=axs[i]
        )
        # sdata2.pl.render_shapes(elements=region_key, color=label_obs_key, groups=group_lst).pl.show(ax=axs[i])
        # sdata2.pp.get_elements([region_key]).pl.render_shapes(color=label_obs_key, groups=group_lst).pl.show(ax=axs[i])
        sdata2.pl.render_shapes(elements=region_key).pl.show(ax=axs[i])

        axs[i].set_title(shapes_lst[i])
        # if(i < len(shapes_lst)):
        #    axs[i].get_legend().remove()

    plt.tight_layout()


def plot_shape_along_axis(
    sdata: sd.SpatialData,
    group_lst: tuple = [],  # the cell types to consider
    gene_lst: tuple = [],  # the genes to consider
    label_obs_key: str = "celltype_spatial",
    sdata_group_key: str = "celltypes",
    target_coordinates: str = "microns",
    scale_expr: bool = False,
    bin_size: int = 50,
    rotation_angle: int = 0,
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
    label_obs_key
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
        group_lst = sdata.table.obs[label_obs_key].unique().tolist()

    # extract polygon to work with (should have been done before using tl.get_sdata_polygon)
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

    sdata2 = sdata_rotate(sdata, rotation_angle, target_coordinates)

    # get elements key, probably need to do better here in the futur !!
    dataset_id = sdata2.table.obs.dataset_id.unique().tolist()[0]
    sdata_transcript_key = dataset_id + "_transcripts"
    sdata_polygon_key = dataset_id + "_polygons"

    # compute dataframes
    df_transcripts = sdata2[sdata_transcript_key].compute()
    df_celltypes = sdata2[sdata_group_key].compute()

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
        df2 = df_celltypes[df_celltypes["ct"] == group_lst[ct]]
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
    sdata.pl.render_shapes(
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
):
    """Plot sdata object (i.e. embedding and polygons). This should always works if well synchronized sdata object

    Parameters
    ----------
    sdata
        SpatialData object.
    color_key
        color key from .table.obs
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.embedding(sdata.table, "umap", color=color_key, ax=ax1, show=False)
    ax1.get_legend().remove()
    if sdata.contains_element(sdata.table.uns["spatialdata_attrs"]["region"]):
        sdata.pl.render_shapes(elements=sdata.table.uns["spatialdata_attrs"]["region"], color=color_key).pl.show(ax=ax2)
    else:
        sq.pl.spatial_scatter(sdata.table, color=color_key, shape=None, size=1, ax=ax2)
    plt.tight_layout()


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
