import os

import anndata as an
import cv2
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile


def load_merscope(folder: str, library_id: str, scale_percent: int) -> an.AnnData:
    """Load vizgen merscope data.

    Parameters
    ----------
    folder
        path to folder.
    library_id
        library id.
    scale_percent
        scaling factor for image and pixel coordinates reduction.

    Returns
    -------
    Anndata initialized object.
    """
    # transformation matrix micron to mosaic pixel
    transformation_matrix = pd.read_csv(
        folder + "/images/micron_to_mosaic_pixel_transform.csv", header=None, sep=" "
    ).values
    # genes
    data = pd.read_csv(folder + "/cell_by_gene.csv", index_col=0)
    datanoblank = data.drop(data.filter(regex="^Blank-").columns, axis=1)
    meta_gene = pd.DataFrame(index=datanoblank.columns.tolist())
    meta_gene["expression"] = datanoblank.sum(axis=0)
    # cells
    meta = pd.read_csv(folder + "/cell_metadata.csv", index_col=0)
    meta = meta.loc[data.index.tolist()]
    meta["cell_cov"] = datanoblank.sum(axis=1)
    meta["barcodeCount"] = datanoblank.sum(axis=1)
    meta["width"] = meta["max_x"].to_numpy() - meta["min_x"].to_numpy()
    meta["height"] = meta["max_y"].to_numpy() - meta["min_y"].to_numpy()

    # Transform coordinates to mosaic pixel coordinates
    temp = meta[["center_x", "center_y"]].values
    cell_positions = np.ones((temp.shape[0], temp.shape[1] + 1))
    cell_positions[:, :-1] = temp
    transformed_positions = np.matmul(transformation_matrix, np.transpose(cell_positions))[:-1]
    meta["x_pix"] = transformed_positions[0, :] * (scale_percent / 100)
    meta["y_pix"] = transformed_positions[1, :] * (scale_percent / 100)
    meta["library_id"] = library_id
    meta = meta.drop(columns=["min_x", "max_x", "min_y", "max_y"])

    # init annData pixel coordinates
    coord_pix = meta[["x_pix", "y_pix"]].to_numpy()
    coord_mic = meta[["center_x", "center_y"]].to_numpy()
    # coordinates.rename(columns={"x_pix": "x", "y_pix": "y"})
    adata = sc.AnnData(
        X=datanoblank.values,
        obsm={"spatial": coord_mic, "X_spatial": coord_mic, "pixel": coord_pix},
        obs=meta,
        var=meta_gene,
    )
    adata.layers["counts"] = adata.X.copy()

    # transcripts
    transcripts = load_transcript(folder + "/detected_transcripts.csv", transformation_matrix, scale_percent)
    adata.uns["transcripts"] = {library_id: {}}
    adata.uns["transcripts"][library_id] = transcripts

    percent_in_cell = adata.obs.barcodeCount.sum(axis=0) * 100 / adata.uns["transcripts"][library_id].shape[0]
    print("\n" + library_id)
    print("total cells=", adata.shape[0])
    print("total transcripts=", transcripts.shape[0])
    print("% in cells=", percent_in_cell)
    print("mean transcripts per cell=", meta["barcodeCount"].mean())
    print("median transcripts per cell=", meta["barcodeCount"].median())

    # image Container
    image = tifffile.imread(folder + "/images/mosaic_DAPI_z2.tif")
    # print('loaded image into memory')
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # print('resized image')
    adata.uns["spatial"] = {library_id: {}}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"] = {"hires": resized}
    adata.uns["spatial"][library_id]["scalefactors"] = {
        "tissue_hires_scalef": 1,
        "spot_diameter_fullres": 5,
        "scale_percent": scale_percent,
        "transformation_matrix": transformation_matrix,
        "folder": folder,
    }
    image = None
    resized = None

    return adata


def save_merscope(adata: an.AnnData, file: str) -> int:
    """Save Anndata object.

    Parameters
    ----------
    adata
        Anndata object.
    file
        path file.

    Returns
    -------
    Some integer value.
    """
    adata.obs = adata.obs.drop(["bounds"], axis=1)
    # del adata.uns['transcripts']
    adata.write(file)
    return 0


def load_transcript(path: str, transformation_matrix: str, scale_percent: int) -> pd.DataFrame:
    """Load detected transcripts.

    Parameters
    ----------
    path
        path to detected transcripts file.
    transformation_matrix
        transformation matrix for pixel to micron coordinate conversion.
    scale_percent
        scaling factor for image and pixel coordinates reduction.

    Returns
    -------
    Detected transcripts pd.DataFrame
    """
    transcripts = pd.read_csv(path, index_col=0)

    # Transform coordinates to mosaic pixel coordinates
    temp = transcripts[["global_x", "global_y"]].values
    transcript_positions = np.ones((temp.shape[0], temp.shape[1] + 1))
    transcript_positions[:, :-1] = temp
    transformed_positions = np.matmul(transformation_matrix, np.transpose(transcript_positions))[:-1]
    transcripts["x_pix"] = transformed_positions[0, :] * (scale_percent / 100)
    transcripts["y_pix"] = transformed_positions[1, :] * (scale_percent / 100)

    # global_x, _y -> micron coordinates
    # x_pix, y_pix -> pixels coordinates
    return transcripts


def load_bounds_pixel(adata: an.AnnData, library_id: str) -> an.AnnData:
    """Load cell boundaries and convert from micron to pixel coordinates in adata.obs["bounds"].

    Parameters
    ----------
    adata
        Anndata object.
    library_id
        library id.

    Returns
    -------
    Anndata object
    """
    folder = adata.uns["spatial"][library_id]["scalefactors"].get("folder")
    scale_percent = adata.uns["spatial"][library_id]["scalefactors"].get("scale_percent")
    transformation_matrix = adata.uns["spatial"][library_id]["scalefactors"].get("transformation_matrix")

    adata.obs["bounds"] = np.array(adata.obs.shape[0], dtype=object)

    if not any(fname.endswith(".parquet") for fname in os.listdir(folder)):
        z_indexes = ["0", "1", "2", "3", "4", "5", "6"]
        for fov in pd.unique(adata.obs["fov"]):
            # try:
            with h5py.File(f"{folder}/cell_boundaries/feature_data_{fov}.hdf5", "r") as f:
                # print(fov)
                for cell_id in adata.obs.index[adata.obs["fov"] == fov]:
                    # print(cell_id)
                    for z in z_indexes:
                        node = f"featuredata/{cell_id}/zIndex_{z}/p_0/coordinates"
                        # print(node)
                        if node in f.keys():
                            temp = f[node][0]
                            boundaryPolygon = np.ones((temp.shape[0], temp.shape[1] + 1))
                            boundaryPolygon[:, :-1] = temp
                            transformedBoundary = np.matmul(transformation_matrix, np.transpose(boundaryPolygon))[
                                :-1
                            ] * (scale_percent / 100)
                            adata.obs["bounds"][cell_id] = transformedBoundary
    else:
        parquet_file = ""
        if os.path.isfile(folder + "/cellpose_micron_space.parquet"):
            parquet_file = folder + "/cellpose_micron_space.parquet"
        elif os.path.isfile(folder + "/cell_boundaries.parquet"):
            parquet_file = folder + "/cell_boundaries.parquet"

        df = gpd.read_parquet(parquet_file)
        for cell_id in df.index:
            if str(df.EntityID[cell_id]) in adata.obs.index:
                temp = np.asarray(df.Geometry[cell_id][0].exterior.coords.xy)
                boundaryPolygon = np.array([temp[0], temp[1], np.ones((temp.shape[1],))]).T
                transformedBoundary = np.matmul(transformation_matrix, np.transpose(boundaryPolygon))[:-1] * (
                    scale_percent / 100
                )
                # case its mosaic -> no conversion
                # transformedBoundary = np.transpose(boundaryPolygon)[:-1] * (scale_percent / 100)
                adata.obs.bounds.loc[str(df.EntityID[cell_id])] = transformedBoundary

    return adata


def load_parquet(adata: an.AnnData, library_id: str) -> an.AnnData:
    """Load cell boundaries as parquet file in pixel coordinates in adata.obs["bounds"].

    Parameters
    ----------
    adata
        Anndata object.
    library_id
        library id.

    Returns
    -------
    Anndata object
    """
    folder = adata.uns["spatial"][library_id]["scalefactors"].get("folder")
    scale_percent = adata.uns["spatial"][library_id]["scalefactors"].get("scale_percent")
    adata.uns["spatial"][library_id]["scalefactors"].get("transformation_matrix")
    adata.obs["bounds"] = np.array(adata.obs.shape[0], dtype=object)

    df = gpd.read_parquet(folder + "/cellpose_mosaic_space.parquet")
    for cell_id in df.index:
        if str(df.EntityID[cell_id]) in adata.obs.index:
            temp = np.asarray(df.Geometry[cell_id][0].exterior.coords.xy)
            boundaryPolygon = np.array([temp[0], temp[1], np.ones((temp.shape[1],))]).T
            transformedBoundary = np.transpose(boundaryPolygon)[:-1] * (scale_percent / 100)
            adata.obs.bounds.loc[str(df.EntityID[cell_id])] = transformedBoundary

    return adata


def get_palette(color_key: str) -> dict:
    """Scinsit palette definition for a specific Htap project.

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
        palette = {"Endothelial": "#00A087FF", "Epithelial": "#3C5488FF", "Immune": "#E64B35FF", "Stroma": "#7E6148FF"}
    elif color_key == "celltype":
        palette = {
            "Basal": "#E41A1C",
            "PNEC": "#377EB8",
            "Pre-TB-SC": "#4DAF4A",
            "TRB-SC": "#984EA3",
            "AT0": "#ffccd5",
            "AT2a": "#F781BF",
            "AT2b": "#BC9DCC",
            "AT1-AT2": "#A65628",
            "AT1-imm": "#54B0E4",
            "AT1": "#84a59d",
            "MCC": "#1B9E77",
            "Lymph": "#a6d8f5",
            "aCap": "#7FCDBB",
            "gCap": "#41B6C4",
            "Art": "#1D91C0",
            "V-Pulm": "#225EA8",
            "V-Sys": "#669bbc",
            "AM1": "#C9CE46",
            "AM2": "#E6B818",
            "AM-prolif": "#ADDD8E",
            "DC1": "#78C679",
            "DC2": "#41AB5D",
            "Mono": "#739154",
            "Inter-Macro": "#01736b",
            "CD4": "#a4ac86",
            "CD8": "#656d4a",
            "T-prolif": "#46c798",
            "NK": "#ebd513",
            "Mast": "#ccbf47",
            "Plasma": "#ff7d00",
            "pDC": "#e670b4",
            "B": "#6F5D85",
            "Megak": "#960c5a",
            "Peri": "#fee0d2",
            "SMC": "#eaac8b",
            "AdvFibro": "#5e1e33",
            "AlvFibro": "#f2ae5a",
            "MyoFibro": "#e37e12",
        }

    return palette
