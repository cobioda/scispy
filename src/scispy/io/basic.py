import numpy as np
import pandas as pd
import spatialdata as sd
import spatialdata_io


def load_sdata_merscope(
    path: str, vpt_outputs: str, region_name: str, slide_name: str, z_layers: int = 2
) -> sd.SpatialData:
    """Load vizgen merscope data as a spatialdata object

    Parameters
    ----------
    path
        path to folder.
    vpt_outputs
        path to vpt folder.
    region_name
        region_name id.
    slide_name
        slide_name id.
    z_layers
        z layers to load.

    Returns
    -------
    SpatialData initialized object loaded first using spatialdata_io library
    """
    sdata = spatialdata_io.merscope(
        path=path, vpt_outputs=vpt_outputs, region_name=region_name, slide_name=slide_name, z_layers=z_layers
    )
    key = slide_name + "_" + region_name
    sdata.table.obs_names.name = None
    sdata[key + "_polygons"].index.name = None

    # transformation matrix micron to mosaic pixel
    transformation_matrix = pd.read_csv(
        path + "/images/micron_to_mosaic_pixel_transform.csv", header=None, sep=" "
    ).values

    # Transform coordinates to mosaic pixel coordinates
    temp = sdata.table.obs[["center_x", "center_y"]].values
    cell_positions = np.ones((temp.shape[0], temp.shape[1] + 1))
    cell_positions[:, :-1] = temp
    transformed_positions = np.matmul(transformation_matrix, np.transpose(cell_positions))[:-1]
    sdata.table.obs["center_x_pix"] = transformed_positions[0, :]
    sdata.table.obs["center_y_pix"] = transformed_positions[1, :]
    # sdata.table.obs = sdata.table.obs.drop(columns=["min_x", "max_x", "min_y", "max_y"])

    # coord_pixels = sdata.table[["center_x_pix", "center_x_pix"]].to_numpy()
    # coord_microns = sdata.table[["center_x", "center_y"]].to_numpy()
    # sdata.table.obsm={"microns": coord_microns, "pixels": coord_pixels},

    sdata.table.layers["counts"] = sdata.table.X.copy()
    # percent_in_cell = sdata.table.obs.n_Counts.sum(axis=0) * 100 / len(sdata[key + '_transcripts'])
    # print("\n" + slide_name)
    # print("total cells=", sdata.table.obs.shape[0])
    # print("total transcripts=", len(sdata[key + '_transcripts']))
    # print("% in cells=", percent_in_cell)
    # print("mean transcripts per cell=", sdata.table.obs["n_Counts"].mean())
    # print("median transcripts per cell=", sdata.table.obs["n_Counts"].median())

    return sdata
