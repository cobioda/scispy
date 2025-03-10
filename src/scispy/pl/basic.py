import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spatialdata as sd
import anndata as ad
from matplotlib import pyplot as plt
from spatialdata import SpatialData
import spatialdata_plot

from scispy.tl.basic import sdata_rotate
from scispy.tl.basic import add_to_points


def plot_shapes(
    sdata: sd.SpatialData,
    group_lst: tuple = None,  # the cell types to consider
    shapes_lst: tuple = None,  # the shapes to plot
    color_key: str = "celltype_spatial",
    shape_key: str = "arteries",
    target_coordinates: str = "microns",
    figsize: tuple = (12, 6),
    palette: tuple = None,
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
    color_key
        label_key in sdata.table.obs to consider
    shape_key
        SpatialData shape element to consider
    palette
        dictionary of colors to use
    target_coordinates
        target_coordinates system of sdata object
    figsize
        figure size
    save
        wether or not to save the figure

    """
    region_key = sdata.table.uns["spatialdata_attrs"]["region"]
    my_shapes = {region_key: sdata[region_key], shape_key: sdata[shape_key]}
    my_tables = {"table": sdata["table"]}
    sdata2 = SpatialData(shapes=my_shapes, tables=my_tables)

    fig, axs = plt.subplots(ncols=len(shapes_lst), nrows=1, figsize=figsize)
    for i in range(0, len(shapes_lst)):
        poly = sdata2[shape_key][sdata2[shape_key].name == shapes_lst[i]].geometry.item()
        sdata3 = sd.polygon_query(
            sdata2,
            poly,
            target_coordinate_system=target_coordinates,
            filter_table=True,
        )

        # sdata3.pl.render_images().pl.show(ax=axs[i])
        if group_lst is None:
            group_lst = sdata2.table.obs[color_key].unique().tolist()

        if palette is not None:
            mypal = [palette[x] for x in group_lst]
            sdata3.pl.render_shapes(elements=region_key, color=color_key, groups=group_lst, palette=mypal).pl.show(
                ax=axs[i]
            )
        else:
            sdata3.pl.render_shapes(elements=region_key, color=color_key, groups=group_lst).pl.show(ax=axs[i])

        axs[i].set_title(shapes_lst[i])
        if i < len(shapes_lst) - 1:
            axs[i].get_legend().remove()

    plt.tight_layout()


def plot_shape_along_axis(
    sdata: sd.SpatialData,
    group_lst: tuple = [],  # the cell types to consider
    gene_lst: tuple = [],  # the genes to consider
    color_key: str = "scmusk",
    sdata_group_key: str = "scmusk",
    target_coordinates: str = "global",
    palette: tuple = [],
    scale_expr: bool = False,
    bin_size: int = 50,
    rotation_angle: int = 0,
    heatmap: bool = False,
    save: bool = False,
    figsize: tuple = (10, 6),
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

    sdata_rotate(sdata, target_coordinates=target_coordinates, rotation_angle=rotation_angle)
    sdata2 = sdata.transform_to_coordinate_system(target_coordinates)
    
    dataset_id = sdata2.table.obs.dataset_id.unique()[0]
    feature_key = sdata2.table.uns["spatialdata_attrs"]["feature_key"]
    polygon_key = sdata2.table.uns["spatialdata_attrs"]["region"][0]
    transcript_key = list(sdata.points.keys())[0]  # need to be in spatialdata_attrs

    # compute dataframes
    df_transcripts = sdata2[transcript_key].compute()
    df_celltypes = sdata2[sdata_group_key].compute()

    # parametrage
    x_min = df_transcripts.x.min()
    total_x = df_transcripts.x.max() - df_transcripts.x.min()
    step_number = int(total_x / bin_size)

    # init color palette
    #cats = sdata.table.obs[color_key].cat.categories.tolist()
    #colors = list(sdata.table.uns[color_key + "_colors"])
    #mypal = dict(zip(cats, colors))
    if palette is not None:
        mypal = palette
        mypal = {x: mypal[x] for x in group_lst}

    # compute values dataframes
    vals = pd.DataFrame({target_coordinates: [], "count": [], feature_key: []})
    for g in range(0, len(gene_lst)):
        df2 = df_transcripts[df_transcripts[feature_key] == gene_lst[g]]
        df2.shape[0] / step_number
        for i in range(0, step_number):
            new_row = {
                target_coordinates: (x_min + (i + 0.5) * bin_size),
                "count": df2[(df2.x > (x_min + i * bin_size)) & (df2.x < (x_min + (i + 1) * bin_size))].shape[0],
                feature_key: gene_lst[g],
            }
            vals = pd.concat([vals, pd.DataFrame([new_row])], ignore_index=True)

    valct = pd.DataFrame({target_coordinates: [], "count": [], "celltype": []})
    for ct in range(0, len(group_lst)):
        df2 = df_celltypes[df_celltypes["ct"] == group_lst[ct]]
        for i in range(0, step_number):
            new_row = {
                target_coordinates: (x_min + (i + 0.5) * bin_size),
                "count": df2[(df2.x > (x_min + i * bin_size)) & (df2.x < (x_min + (i + 1) * bin_size))].shape[0],
                "celltype": group_lst[ct],
            }
            valct = pd.concat([valct, pd.DataFrame([new_row])], ignore_index=True)

    # draw figure
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=figsize, sharex=True)

    sdata.pl.render_shapes(
        elements=polygon_key, color=color_key, palette=list(mypal.values()), groups=group_lst,
        fill_alpha=1, outline_width=0.5, outline_color='#000000', outline_alpha=1,
    ).pl.show(ax=ax1)
    #ax1.legend(bbox_to_anchor=(1.0, 1.0))
    ax1.get_legend().remove()

    sns.lineplot(data=valct, x=target_coordinates, y="count", hue="celltype", linewidth=0.9, palette=mypal, ax=ax2)
    ax2.legend(bbox_to_anchor=(1.0, 1.0), fontsize='x-small')

    if scale_expr is True:
        means_stds = vals.groupby([feature_key])["count"].agg(["mean", "std", "max"]).reset_index()
        vals = vals.merge(means_stds, on=[feature_key])
        vals["norm_count"] = vals["count"] / vals["max"]

        if heatmap is True:
            #sns.set(font_scale=0.5)
            sns.heatmap(vals.pivot(index="gene", columns=target_coordinates, values="norm_count"), cmap="viridis", ax=ax3)
        else:
            sns.lineplot(
                data=vals,
                x=target_coordinates,
                y="norm_count",
                hue=feature_key,
                linewidth=0.9,
                palette=sns.color_palette("Paired"),
                ax=ax3,
            )
            ax3.legend(bbox_to_anchor=(1.0, 1.0), fontsize='x-small')

    else:
        if heatmap is True:
            sns.heatmap(vals.pivot(index="gene", columns=target_coordinates, values="count"), cmap="viridis", ax=ax3)
        else:
            sns.lineplot(
                data=vals,
                x=target_coordinates,
                y="count",
                hue=feature_key,
                linewidth=0.9,
                palette=sns.color_palette("Paired"),
                ax=ax3,
            )
            ax3.legend(bbox_to_anchor=(1.0, 1.0), fontsize='x-small')

    ax1.set_title(dataset_id)
    ax2.set_title("Number of cells per cell types (" + str(int(bin_size)) + " µm bins)")
    ax3.set_title("Gene expression (" + str(int(bin_size)) + " µm bins)")
    #plt.legend(fontsize='small', title_fontsize='6')
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
    target_coordinates: str = "microns",
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
        shape_keys = sdata.table.uns['spatialdata_attrs']['region']
        #shape_keys = list(sdata.shapes.keys())  # better would be to get the list of spatialdata_attrs 'cell_regions'
    if feature_key is None:
        feature_key = sdata.tables[list(sdata.tables.keys())[0]].uns["spatialdata_attrs"]["feature_key"]

    args_shapes = {"element": shape_keys, "method": "matplotlib", "color": color_key}
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

    args_shapes["coordinate_system"] = target_coordinates
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

    sdata.pl.render_shapes(element=sdata.table.uns["spatialdata_attrs"]["region"][0], color=color_key).pl.show(ax=ax2)
    ax2.get_legend().remove()
    ax2.set_title("spatialdata polygons")

    sc.pl.embedding(sdata.table, "spatial", color=color_key, size=1, ax=ax3)
    #sq.pl.spatial_scatter(sdata.table, basis='spatial', color=color_key, shape=None, size=1, ax=ax3)
    #ax2.get_legend().remove()
    #ax3.set_title("squidpy spatial")

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
    elif color_key == "HTAP":
        palette = {
            # htap
            "AT2": "#3E8F91",
            "AT1": "#6F5D85",
            "Basal": "#E41A1C",
            "Multiciliated": "#1f618d",
            "Pre-TB secretory": "#3b683f",
            "Secretory": "#E6AB02",
            "AT0": "#BA6866",
            "AT1-AT2": "#F2920D",
            "Rare": "#DF8CC4",
            "EC general capillary": "#4EA2D7",
            "Plasma cells": "#78281f",
            "EC venous pulmonary": "#1E6275",
            "EC venous systemic": "#2FA679",
            "EC aerocyte capillary": "#95D286",
            "Lymphatic EC": "#2d7687",
            "EC arterial": "#C9CE46",
            "Smooth muscle": "#ec7063",
            "Alveolar fibroblasts": "#af801d",
            "Adventitial fibroblasts": "#D6217C",
            "Myofibroblasts": "#426F8E",
            "Pericytes": "#7b241c",
            "Mast cells": "#F79F80",
            "Alveolar macrophages": "#BC6399",
            "C1Q+ macrophages": "#B22070",
            "CD4 T cells": "#674A9C",
            "CD8 T cells": "#79838A",
            "B cells": "#668C61",
            "NK cells": "#8AA20A",
            "Monocytes": "#D1DC1F",
            "DC": "#AB674F",
            "Interstitial Mph perivascular": "#ff00a2",
            "Megakaryocytes": "#d68a1c",
        }
    elif color_key == "ann_level_2":
        # paolo
        palette = {
            "Cartilages": "#0B4B19",
            "Stromal0": "#99D6A9",
            "Stromal1": "#1B8F76",
            "Stromal2": "#9DAF07",
            "Osteoblasts": "#4CAD4C",
            "Progenitor cells": "#03045e",
            "Schwann cells": "#95ccff",
            "Lymphatic EC": "#F78896",
            "Vascular EC": "#E788C2",
            "Pericytes": "#BBD870",
            "Satellites": "#CB7647",
            "Skeletal muscle": "#926B54",
            "Neural crest": "#E3D9AC",
            "Olf. ensh. glia": "#cd6889",
            "Glia progenitors": "#FF4500",
            "ALK neurons": "#95819F",
            "NOS1 neurons": "#95819F",
            "Olfactory HBCs": "#E41A1C",
            "Respiratory HBCs": "#C82C73",
            "Olf. microvillars": "#efe13c",
            "Multiciliated": "#1f618d",
            "Deuterosomal": "#3498db",
            "Sustentaculars": "#C09ACA",
            "GBCs": "#F48B5A",
            "preOSNs": "#E69F00",
            "iOSNs": "#f05b43",
            "mOSNs": "#33b8ff",
            "Neural progenitors": "#6A0B78",
            "Excitatory neurons": "#706fd3",
            "Inhibitory neurons": "#800EF1",
            "GnRH neurons": "#2EECDB",
            "Myeloid": "#736376",
            "Microglia": "#91BFB7",

            #"Cycling HBCs": "#C2A523",
            #"Tufts": "#eb10fd",
            #"Duct": "#efe13c",
        }
    elif color_key == "ann_level_1":
        palette = {
            "Progenitor cells": "#03045e",
            "Olfactory epithelium": "#EF1B4F",
            "Respiratory epithelium": "#5562B7",
            "Neurons": "#6E5489",
            "Glial": "#919976",
            "Stroma": "#009E73",
            "Immune": "#2EECDB",
            "Vasculars": "#CC79A7",
            "Myocytes": "#803800",
            "Immune": "#736376",
            "Pericytes": "#BBD870",
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
        
        #sc.pl.embedding(sdata.table, "spatial", color=key, size=1, ax=axs[1])
        
        #sq.pl.spatial_scatter(
        #    tmp,
        #    groups=tmp.obs[clust].cat.categories[1:].values,
        #    color=adata.obs[clust_key].cat.categories.tolist(),
        #    size=size,
        #    frameon=frameon,
        #    legend_loc=legend_loc,
        #    **kwargs,
        #)


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


def plot_pseudobulk(
    adata: ad.AnnData,
    x_key: str = 'scmusk',
    y_key: str = 'log2FoldChange',
    palette: tuple = None,
    key: str = 'results',
    title: str = None,
    padj: float = 0.05,
    log2FoldChange: float = 1,
    baseMean: float = 50,
    figsize: tuple = (8,3),
    save: bool = False,
    save_format: str = 'pdf',
):
    """Plot DEG dataframe from pseudobulk analysis

    Parameters
    ----------
    adata
        anndata object
    x_key
        x key
    y_key
        y key
    key
        key in adata.uns['scispy'] storing the results to plot
    padj
        p adjusted to be significant
    log2FoldChange
        log2FoldChange to be significant
    figsize
        figure size
    save
        wether or not to save the figure
    save_format
        pdf or png
    """
    if palette is None:
        palette="deep"

    df = adata.uns['scispy'][key].copy()

    df['significative'] = 0
    df.loc[df.padj < padj, 'significative'] = 1
    df.loc[abs(df.log2FoldChange) < log2FoldChange, 'significative'] = 0
    df.loc[abs(df.baseMean) < baseMean, 'significative'] = 0

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    tmp = df[df.significative==1]
    tmp = tmp.reset_index(drop=True)
    order = list(tmp.groupby([x_key]).groups.keys())

    sns.stripplot(data=tmp, x=x_key, y=y_key, hue=x_key, palette=palette,
                orient='v', order=order, alpha=0.6, size=5, linewidth=1, edgecolor='black', jitter=0.4)

    grouped = tmp.groupby([x_key])
    # Label the points within each group
    for collection, (group_key, group_data) in zip(ax.collections, grouped):
        for i, (x, y) in enumerate(collection.get_offsets()):
            y_value = group_data.iloc[i]['index']
            ax.text(x, y+0.1, y_value, ha='left', va='bottom', color='grey', size='x-small')

    tmp = df[df.significative==0]
    sns.stripplot(data=tmp, x=x_key, y=y_key, color="grey", 
                orient='v', alpha=0.6, size=2, jitter=0.4)

    ax.tick_params(axis='x', rotation=90)
    ax.set_title(title)

    # save figure
    if save is True:
        if save_format == "pdf":
            print("saving plot_pseudobulk.pdf")
            plt.savefig("plot_pseudobulk.pdf", bbox_inches="tight")
        elif save_format == "png":
            print("saving plot_pseudobulk.png")
            plt.savefig("plot_pseudobulk.png", dpi=300, bbox_inches="tight")