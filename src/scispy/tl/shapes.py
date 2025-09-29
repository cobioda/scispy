import spatialdata as sd
import pandas as pd
import geopandas as gpd
from spatialdata.models import ShapesModel
from shapely import count_coordinates, get_coordinates
from spatialdata.transformations import Identity, get_transformation
import shapely
import numpy as np
import anndata as ad
import networkx as nx 
import squidpy as sq 
from ..tl.alpha_shape import alpha_shape_optimal
# from shapely.ops import cascaded_union
# from shapely.geometry import Polygon
# from spatialdata.models import PointsModel
# from shapely import count_coordinates
# from shapely import count_coordinates, get_coordinates
# from shapely import affinity
# from anndata import AnnData
# from scipy.sparse import csr_matrix

def add_to_shapes(
    sdata: sd.SpatialData,
    poly: list, 
    name: list,
    centerline: list = None,
    shape_key: str = "myshapes",
    scale_factor: float = 0.50825,  # if shapes comes from xenium explorer
    target_coordinates: str = "microns",
    transfo_object_key : str = 'images',
    # **kwargs,
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
    if shape_key in sdata.shapes:
        print(f'Shape "{shape_key}" is already present in the object.')
        return

    d = {"geometry": poly, "name": name}
    if centerline is not None:
        d["centerline"] = centerline

    gdf = gpd.GeoDataFrame(d)
    
    # scale because it comes from the xenium explorer !!!
    gdf.geometry = gdf.geometry.scale(
        xfact=scale_factor, yfact=scale_factor, origin=(0, 0)
    )

    # substract the initial image offset (x,y)
    # matrix = sd.transformations.get_transformation(
    #     sdata[transfo_object_key], target_coordinates
    # ).to_affine_matrix(input_axes=["x", "y"], output_axes=["x", "y"])
    # x_translation = matrix[0][2]
    # y_translation = matrix[1][2]
    # gdf.geometry = gdf.geometry.apply(
    #     affinity.translate, xoff=x_translation, yoff=y_translation
    # )
    transfo = sd.transformations.get_transformation(
        sdata[transfo_object_key], target_coordinates
    )

    if centerline is not None:
        gdf['centerline'] = gpd.GeoSeries(gdf['centerline'])
        gdf.centerline = gdf.centerline.scale(
            xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
        # gdf.centerline = gdf.centerline.apply(
            # affinity.translate, xoff=x_translation, yoff=y_translation)

    sdata.shapes[shape_key] = ShapesModel.parse(
        gdf, transformations = {target_coordinates: Identity()}
    )
    sd.transformations.set_transformation(
        sdata.shapes[shape_key], 
        transfo, 
        to_coordinate_system=target_coordinates
    )
    # print(sd.transformations.get_transformation(
    #     sdata[shape_key], target_coordinates
    # ))

    print(f"New shape added : '{shape_key}'")
    return


def add_metadata_to_shape(
    sdata: sd.SpatialData,
    obs_key: list,
    shape_key: str = "myshapes",
    target_coordinates="microns",
    right_on: str = None,
):
    """Add metadata to a shape in the sdata.shape.keys()

    Parameters
    ----------
    sdata
        SpatialData object.
    obs_key
        list of column's name that we want to add in the element shape
    shape_key
        key of element shape

    Return
    ----------
    Add some metadata in element shape.
    """
    for key in obs_key:
        if key in sdata.shapes[shape_key].columns:
            print(f'This column "{key}" is already present in the shape.')
            obs_key.remove(key)
            # return

    if right_on:
        obs_key.append(right_on)
        gdf = sdata.shapes[shape_key].merge(
            sdata.table.obs[obs_key], 
            how="left", left_index=True, right_on = right_on
        )
    else:
        gdf = sdata.shapes[shape_key].merge(
            sdata.table.obs[obs_key], 
            how="left", left_index=True, right_index=True
        )

    transfo = get_transformation(sdata[shape_key])
    print(transfo)
    sdata.shapes[shape_key] = ShapesModel.parse(
        gdf, transformations={target_coordinates: transfo}
    )
    print(get_transformation(sdata[shape_key]))

    sdata.shapes[shape_key]["len_shape"] = sdata.shapes[shape_key]["geometry"].apply(
        lambda x: count_coordinates(x)
    )
    return


def shapes_of_cell_type(
    sdata: sd.SpatialData,
    celltype: str,
    obs_key: str = "celltype_spatial",
    shape_key: str = "myshapes",
) -> list:
    """Extract shapes from a celltype. First step for the mean shape.

    Parameters
    ----------
    sdata
        SpatialData object.
    celltype
        name of the cell type we want to extract
    obs_key
        name of column where cell type can be found in sdata.table.obs
    shape_key
        key of element shape

    Returns
    -------
    List of boundary coordinates for each cell.
    """
    # Extract cell shapes of the defined cell type
    idx_cells = sdata.table.obs[sdata.table.obs[obs_key] == celltype].index
    gdf_shapes_cells = sdata[shape_key].loc[idx_cells]

    if len(gdf_shapes_cells["geometry"].geom_type.unique()) != 1:
        print("Geometry type is not unique !!!")

    # Extract the x and y coordinates of each shape
    shapes_coordinates = (
        gdf_shapes_cells["geometry"].apply(lambda x: get_coordinates(x)).to_list()
    )

    # OLD extract
    # shapes_coordinates = []
    # for shape in gdf_shapes_cells["geometry"]:
    #     if shape.geom_type == "Polygon":
    #         # coordinates = get_coordinates(shape).tolist()
    #         coordinates = list(shape.exterior.coords)
    #         shapes_coordinates.append(coordinates)
    #     elif shape.geom_type == "MultiPolygon":
    #         print("MultiPolygon")
    #         for polygon in shape:
    #             # coordinates = shapely.get_coordinates(polygon).tolist()
    #             coordinates = list(polygon.exterior.coords)
    #             shapes_coordinates.append(coordinates)

    return shapes_coordinates




def count_in_shape(
    sdata: sd.SpatialData,
    shape_key: str,
    transcript_key: str = 'transcripts',
    feature_key: str = 'feature_name',
    qv: int = 20,
    how: str = "inner",
    predicate: str = "intersects",
    coordinates: list = ['x', 'y'],
    gene_exclude_pattern: str = "Unassigned.*|Deprecated.*|Intergenic.*|Neg.*",
):
    df_transcripts = sdata[transcript_key].copy()
    df_transcripts = df_transcripts[(df_transcripts['qv'] >= qv) & (df_transcripts.is_gene)]
    df_transcripts = df_transcripts[~(df_transcripts[feature_key].str.contains(gene_exclude_pattern, regex=True))].compute()
    df_transcripts[feature_key] = df_transcripts[feature_key].cat.remove_unused_categories()

    df_transcripts = gpd.GeoDataFrame(
        df_transcripts, 
        geometry=gpd.points_from_xy(df_transcripts[coordinates[0]], 
                                    df_transcripts[coordinates[1]]))
    
    df_merge = gpd.sjoin(df_transcripts, sdata[shape_key], predicate=predicate, how=how)
    count = df_merge.groupby(feature_key).size().sort_index()
    print("nb_genes : ", len(count), np.any(count == 0))
    return count.to_numpy()



def shape_to_pseudobulk(
    sdata: sd.SpatialData,
    obs_val: int | str,
    obs_key: str,
    # library_key: str,
    table_key: str = 'table',
    cell_id: str = 'cell_id',
    samples: list | None = None,
    metadata: list | None = None,
    scale_factor: float = 1.0,
    convex_hull: bool = False,
    only_shape: bool = True,
    percentile: float = 99.0,
    region: str = 'region',
    option = 1,
    connectivity_key: str ='spatial_connectivities', 
    distances_key: str ='spatial_distances',
    neighs_key: str ='spatial_neighbors',    
    save: bool = True,
):
    if not samples:
        samples = list(sdata.tables.keys())
    print(len(samples))
    shape_pseudo_counts = {}
    samples_meta_dict= {}

    for _, table_key in enumerate(samples):
        print(table_key)

        id_sample = table_key.split('-')[1]
        transcript_key = f"transcripts-{id_sample}"
        shape_key = f"shape-{id_sample}" 
        shape_cells_key = sdata[table_key].uns['spatialdata_attrs']['region']

        if type(shape_cells_key) == list:
            shape_cells_key = shape_cells_key[0]

        if type(obs_val) != list:
            obs_val = [obs_val]

        if metadata:
            metadata_dict = {col: sdata[table_key].obs[col].unique()[0] for col in metadata}
            samples_meta_dict[id_sample] = metadata_dict

        shape = alpha_shape_optimal(
                sdata=sdata,
                group_by = obs_key,
                groups = obs_val,
                table_key = table_key,
                cell_id = cell_id,
                convex_hull = convex_hull,
                only_shape = only_shape,
                percentile = percentile,
                region = region,
                connectivity_key = connectivity_key, 
                distances_key = distances_key,
                neighs_key = neighs_key,
                option = option, 
            )
        
        print('add shape')
        add_to_shapes(
            sdata = sdata,
            poly = [shape],
            name = [shape_key],
            target_coordinates ="global",
            transfo_object_key= transcript_key,
            shape_key= shape_key,
            scale_factor = scale_factor,
        )
        print('count')
        shape_pseudo_counts[id_sample] = count_in_shape(
            sdata= sdata,
            shape_key = shape_key ,
            transcript_key = transcript_key,
        )
    pseudo_counts = pd.DataFrame(shape_pseudo_counts)
    pseudo_counts.index = sdata[table_key].var_names

    if metadata:
        pdata_shapes = ad.AnnData(
            X = pseudo_counts.T, 
            obs = pd.DataFrame(samples_meta_dict).T
        )
    else:
        pdata_shapes = ad.AnnData(
            X = pseudo_counts.T
        )

    if save:
        if type(obs_val) == list:
            file = f'shape_pseudocount_{obs_key}_{"_".join(map(str, obs_val))}_per_samples'
        else:
            file = f'shape_pseudocount_{obs_key}_{obs_val}_per_samples'
        pdata_shapes.write(file + '.h5ad')
    return pdata_shapes



# def old_add_to_shapes(
#     sdata: sd.SpatialData,
#     shape_file: str,
#     shape_key: str = "myshapes",
#     scale_factor: float = 0.50825,  # if shapes comes from xenium explorer
#     target_coordinates: str = "microns",
#     **kwargs,
# ):
#     """Add shape element to SpatialData.

#     Parameters
#     ----------
#     sdata
#         SpatialData object.
#     shape_file
#         coordinates.csv file from xenium explorer (region = "normal_1")
#         # vi coordinates.csv -> remove 2 first # lines
#         # dos2unix coordinates.csv
#     shape_key
#         key of element shape
#     scale_factor
#         scale factor conversion applied to x and y coordinates for real micron coordinates
#     target_coordinates
#         target_coordinates system

#     """
#     if shape_key in list(sdata.shapes.keys()):
#         print(f'Shape "{shape_key}" is already present in the object.')
#         return

#     d = {"geometry": [], "name": []}
#     df = pd.read_csv(shape_file, **kwargs) 
#     # if target_coordinates == 'global':
#     #     print(f'Convert shape in micron to pixels with a pixel size of : {pixel_size}')
#     #     df[['X', 'Y']] = df[['X', 'Y']] / pixel_size
        
#     for name, group in df.groupby("Selection"):
#         if len(group) >= 3:
#             poly = shapely.Polygon(zip(group.X, group.Y))
#             d["geometry"].append(poly)
#             d["name"].append(name)
#         else:
#             print("Shape with less than 3 points")
            
#     gdf = gpd.GeoDataFrame(d)

#     # scale because it comes from the xenium explorer !!!
#     gdf.geometry = gdf.geometry.scale(
#         xfact=scale_factor, yfact=scale_factor, origin=(0, 0)
#     )

#     # substract the initial image offset (x,y)
#     image_object_key = list(sdata.images.keys())[0]
#     matrix = sd.transformations.get_transformation(
#         sdata[image_object_key], target_coordinates
#     ).to_affine_matrix(input_axes=["x", "y"], output_axes=["x", "y"])
#     x_translation = matrix[0][2]
#     y_translation = matrix[1][2]
#     gdf.geometry = gdf.geometry.apply(
#         affinity.translate, xoff=x_translation, yoff=y_translation
#     )

#     sdata.shapes[shape_key] = ShapesModel.parse(
#         gdf, transformations ={target_coordinates: Identity()}
#     )
#     print(f"New shape added : '{shape_key}'")
#     return
