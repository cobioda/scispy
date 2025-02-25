import spatialdata as sd
import pandas as pd
import geopandas as gpd
from spatialdata.models import ShapesModel
from shapely import count_coordinates, get_coordinates, affinity
from spatialdata.transformations import Identity
import shapely
from shapely.ops import unary_union, polygonize
import shapely
from scipy.spatial import Delaunay
import numpy as np
import math
import warnings
# from shapely.ops import cascaded_union
# from shapely.geometry import Polygon
# from spatialdata.models import PointsModel
# from shapely import count_coordinates

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
        gdf = sdata.shapes[shape_key].merge(
            sdata.table.obs[[obs_key, right_on]], 
            how="left", left_index=True, right_on = right_on
        )
    else:
        gdf = sdata.shapes[shape_key].merge(
            sdata.table.obs[obs_key], 
            how="left", left_index=True, right_index=True
        )
    sdata.shapes[shape_key] = ShapesModel.parse(
        gdf, transformations={target_coordinates: Identity()}
    )
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



def alpha_shape(
    points: list, 
    alpha: float,
    threshold: int = None,
    only_shape: bool = True,
) -> tuple | shapely.Polygon | shapely.MultiPolygon:
    """Compute the alpha shape of a set of points.
    https://web.archive.org/web/20201013181320/http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
    ; https://gist.github.com/dwyerk/10561690 ; https://gist.github.com/jclosure/d93f39a6c7b1f24f8b92252800182889#file-concave_hulls-ipynb 
    ; https://github.com/mlichter2/concavity
    
    Parameters
    ----------
    points
        List of cell centroids
    alpha
        Value to influence the gooeyness of the border. Smaller numbersdon't fall inward as much as larger numbers.
        Too large, and you lose everything
    threshold
        Threshold to estimate the shape. Default none.
    only_shape
        By default return only the shape. If False return the shape with the edge_points (lines) 
        and all_circum_r (radius of all the circumcircle )
    
    Returns
    -------
    By default return only the shape (Polygon or MultiPolygon). 
    If only_shape is False return a tuple with the shape, all the lines used to calculate the shape 
    and all the radii of the circumcircle.
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        warnings.warn("Warning Message: Less than 4 points, simply compute the convex hull") 
        return shapely.MultiPoint(points).convex_hull
    
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
        
    if threshold:
        print('threshodl') 
        seuil = threshold
    else:
        print('alpha')
        seuil = 1.0/alpha
        
    coords = np.array([point.coords[0]
                       for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    all_circum_r = []

    for ia, ib, ic in tri.simplices: 
        # ia, ib, ic = indices of corner points of the triangle
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        
        s = (a + b + c)/2.0 # Semiperimeter of triangle
        area = math.sqrt(s*(s-a)*(s-b)*(s-c)) # Area of triangle by Heron's formula
        circum_r = a*b*c/(4.0*area) # radius of circumcircle
        all_circum_r.append(circum_r)
        
        if circum_r < seuil:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
            
    m = shapely.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    
    if only_shape:
        return unary_union(triangles)
    else:
        return unary_union(triangles), edge_points, all_circum_r

