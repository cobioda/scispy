import os

import anndata as an
import cv2
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import tifffile

""" from dask import array as da
import dask.dataframe as dd
from dask_image.imread import imread
from dask.dataframe import read_csv

from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, Identity
 """


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
        dtype=np.float32,
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
            "ghost": "#cfcfcf",
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

    return palette


"""
def merfish(path: str, library_id: str, scale_percent: int) -> SpatialData:

    Read *MERFISH* data from Vizgen.
    Parameters
    ----------
    path
        Path to merscope or vpt output directory containing :
        - cell_by_gene.csv
        - cell_metadata.csv
        - detected_transcripts.csv
        - cellpose_micron_space.parquet or cell_boundaries.parquet
        - images
            - micron_to_mosaic_pixel_transform.csv
            - mosaic_DAPI_z2.tif

    Returns
    -------
    :class:`spatialdata.SpatialData`

    path = Path(path)
    count_path = path / 'cell_by_gene.csv'
    obs_path = path / 'cell_metadata.csv'
    transcript_path = path / 'detected_transcripts.csv'
    boundaries_path = path / 'cell_boundaries.parquet'
    if os.path.isfile(path / 'cellpose_mosaic_space.parquet'):
        boundaries_path = path / 'cellpose_mosaic_space.parquet'
    images_dir = path / 'images'
    microns_to_pixels = np.genfromtxt(images_dir / 'micron_to_mosaic_pixel_transform.csv')
    microns_to_pixels = Affine(microns_to_pixels, input_axes=("x", "y"), output_axes=("x", "y"))

    ### Images
    images = {}
    exp = r"mosaic_(?P<stain>[\\w|-]+[0-9]?)_z(?P<z>[0-9]+).tif"
    matches = [re.search(exp, file.name) for file in images_dir.iterdir()]
    stainings = {match.group("stain") for match in matches if match}
    z_levels = {match.group("z") for match in matches if match}
    for z in z_levels:
        im = da.stack([imread(images_dir / f"mosaic_{stain}_z{z}.tif").squeeze() for stain in stainings], axis=0)
        parsed_im = Image2DModel.parse(
            im,
            dims=("c", "y", "x"),
            transformations={"global": Identity()},
            # transformations={"pixels": Identity(), "microns": microns_to_pixels.inverse()},
            c_coords=stainings,
        )
        images[f"z{z}"] = parsed_im

    ### Transcripts
    transcript_df = dd.read_csv(transcript_path)
    transcripts = PointsModel.parse(
        transcript_df,
        coordinates={"x": 'global_x', "y": 'global_y', "z": 'global_z'},
        transformations={"global": Identity()},
        # transformations={"microns": Identity(), "pixels": microns_to_pixels},
    )
    points = {}
    gene_categorical = dd.from_pandas(transcripts['gene'].compute().astype('category'), npartitions=transcripts.npartitions).reset_index(drop=True)
    transcripts['gene'] = gene_categorical

    points = {"transcripts": transcripts}

    # split the transcripts into the different z-levels
    #z = transcripts['z'].compute()
    #z_levels = z.value_counts().index
    #z_levels = sorted(z_levels, key=lambda x: int(x))
    #for z_level in z_levels:
    #    transcripts_subset = transcripts[z == z_level]
    #    # temporary solution until the 3D support is better developed
    #    transcripts_subset = transcripts_subset.drop('z', axis=1)
    #    points[f"transcripts_z{int(z_level)}"] = transcripts_subset

    ### Polygons
    geo_df = gpd.read_parquet(boundaries_path)
    geo_df = geo_df[geo_df.ZIndex == 2]
    geo_df = geo_df.rename_geometry("geometry")
    geo_df.index = geo_df['EntityID'].astype(str)
    geo_df = geo_df.drop(columns=['EntityID','ZIndex','ID','Type','ZLevel','Name','ParentID','ParentType'])

    polygons = ShapesModel.parse(geo_df, transformations={"global": Identity()})
    # polygons = ShapesModel.parse(geo_df, transformations={"microns": Identity(), "pixels": microns_to_pixels})
    shapes = {"polygons": polygons}

    ### Table
    data = pd.read_csv(count_path, index_col=0, dtype={'cell': str})
    obs = pd.read_csv(obs_path, index_col=0, dtype={'EntityID': str})

    is_gene = ~data.columns.str.lower().str.contains("blank")
    adata = an.AnnData(data.loc[:, is_gene], dtype=data.values.dtype, obs=obs)

    adata.obsm["blank"] = data.loc[:, ~is_gene]  # blank fields are excluded from adata.X
    adata.obsm["spatial"] = adata.obs[['center_x', 'center_y']].values
    adata.uns["spatial"] = {'mylib': {}}
    adata.obs["region"] = pd.Series(path.stem, index=adata.obs_names, dtype="category")
    adata.obs['EntityID'] = adata.obs.index

    table = TableModel.parse(
        adata,
        region_key="region",
        region=adata.obs["region"].cat.categories.tolist(),
        instance_key='EntityID',
    )

    return SpatialData(shapes=shapes, points=points, images=images, table=table)
"""
