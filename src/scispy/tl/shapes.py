import spatialdata as sd
import pandas as pd
import geopandas as gpd
from spatialdata.models import PointsModel, ShapesModel
from shapely.geometry import Polygon
from shapely import count_coordinates, get_coordinates, affinity
from spatialdata.transformations import Identity
import shapely

def add_to_shapes(
    sdata: sd.SpatialData,
    shape_file: str,
    shape_key: str = "myshapes",
    scale_factor: float = 0.50825,  # if shapes comes from xenium explorer
    target_coordinates: str = "microns",
    **kwargs,
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
    if shape_key in list(sdata.shapes.keys()):
        print(f'Shape "{shape_key}" is already present in the object.')
        return

    d = {"geometry": [], "name": []}
    df = pd.read_csv(shape_file, **kwargs) 
    # if target_coordinates == 'global':
    #     print(f'Convert shape in micron to pixels with a pixel size of : {pixel_size}')
    #     df[['X', 'Y']] = df[['X', 'Y']] / pixel_size
        
    for name, group in df.groupby("Selection"):
        if len(group) >= 3:
            poly = shapely.Polygon(zip(group.X, group.Y))
            d["geometry"].append(poly)
            d["name"].append(name)
        else:
            print("Shape with less than 3 points")
            
    gdf = gpd.GeoDataFrame(d)

    # scale because it comes from the xenium explorer !!!
    gdf.geometry = gdf.geometry.scale(
        xfact=scale_factor, yfact=scale_factor, origin=(0, 0)
    )

    # substract the initial image offset (x,y)
    image_object_key = list(sdata.images.keys())[0]
    matrix = sd.transformations.get_transformation(
        sdata[image_object_key], target_coordinates
    ).to_affine_matrix(input_axes=["x", "y"], output_axes=["x", "y"])
    x_translation = matrix[0][2]
    y_translation = matrix[1][2]
    gdf.geometry = gdf.geometry.apply(
        affinity.translate, xoff=x_translation, yoff=y_translation
    )

    sdata.shapes[shape_key] = ShapesModel.parse(
        gdf, transformations ={target_coordinates: Identity()}
    )
    print(f"New shape added : '{shape_key}'")
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
