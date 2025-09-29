from anndata import AnnData
from scipy.sparse import csr_matrix
import networkx as nx 
import squidpy as sq
import numpy as np 
import shapely
import math
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import warnings
import spatialdata as sd


def alpha_shape_optimal(
    sdata: sd.SpatialData,
    group_by: str,
    groups: int|str|list,
    table_key: str = 'table',
    cell_id: str = 'cell_id',
    convex_hull: bool = False,
    only_shape: bool = True,
    percentile: float = 99.0,
    region: str = 'region',
    connectivity_key: str ='spatial_connectivities', 
    distances_key: str ='spatial_distances',
    neighs_key: str ='spatial_neighbors',
    option = 1, #option 1 = remove long link et largest_cc
    #  option 2 = len(list_points) * percentile / 100
):  
    if type(groups) != list:
        groups = [groups]
    adata = sdata[table_key][sdata[table_key].obs[group_by].isin(groups)].copy()
    shape_key = adata.uns['spatialdata_attrs'][region]
    
    if type(shape_key) == list:
        # print(len(shape_key))
        shape_key = shape_key[0]
    
    if (option == 1) | (convex_hull):
        print(f'Remove long links > {percentile} percentile...')
        sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True)
        remove_long_links(
            adata,
            distance_percentile = percentile,
            connectivity_key=connectivity_key, 
            distances_key=distances_key,
            neighs_key=neighs_key)
        G = nx.from_numpy_array(adata.obsp[connectivity_key].todense())
        largest_cc = max(nx.connected_components(G), key=len)
    

    if convex_hull:
        print('Convexe hull...')
        sub_cells = adata[list(largest_cc),].obs[cell_id].values
        list_points = [poly.centroid for poly in sdata[shape_key].loc[sub_cells,].geometry.values]
        # print(len(list_points))
        pol = shapely.convex_hull(shapely.MultiPoint(list_points))
    else:
        if option == 1:
            nb_cells = len(largest_cc)
            print(nb_cells)
        ### MAYBE TO REMOVE OPTION 2 
        ### recall and precision lower than option 1 ????
        elif option == 2:    
            print('Option 2 avec percent of list_points')
            nb_cells = int(len(list_points) * percentile / 100)
            print(nb_cells)
        #######################################
        sub_cells = adata.obs[cell_id].values
        list_points = [poly.centroid for poly in sdata[shape_key].loc[sub_cells].geometry.values]

        pol, alpha, alpha_cells = trouver_alpha_dicho(
            points = list_points, 
            seuil = nb_cells, 
            borne_sup = 1000)
        print(f'{alpha}: {alpha_cells} cells')

    if convex_hull or only_shape:
        return pol
    else: 
        return pol, alpha, alpha_cells

    


def alpha_shape(
    points: list, 
    alpha: float,
    # threshold: int = None,
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
        
        if circum_r < alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
            
    m = shapely.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    
    if only_shape:
        return unary_union(triangles)
    else:
        return unary_union(triangles), edge_points, all_circum_r



def function_alpha_counts_cells(
    points: list,
    alpha: float,
):
    shapes = alpha_shape(points = points, alpha = alpha)
    if shapes.geom_type == 'MultiPolygon':
        pol = max(shapes.geoms, key=lambda g: g.area)
    else:
        pol = shapes
    count = shapely.covers(pol, points).sum()
    return pol, count



def trouver_alpha_dicho(points, seuil, borne_sup=1000):
    """
    Trouve le plus petit alpha ∈ N tel que f(alpha) == seuil
    en utilisant une recherche dichotomique (optimisée).
    
    Hypothèse : f est croissante.
    
    Paramètres :
    - f : fonction de N → N, croissante
    - seuil : entier, valeur cible
    - borne_sup : limite supérieure pour la recherche
    
    Retour :
    - alpha : entier tel que f(alpha) == seuil
    - None : si aucun alpha trouvé dans [0, borne_sup]
    """
    gauche, droite = 1, borne_sup
    best = None 
    
    while gauche <= droite:
        milieu = (gauche + droite) // 2
        pol, val = function_alpha_counts_cells(points=points, alpha= milieu)
        
        if val == seuil:
            best = (pol, milieu, val)
            droite = milieu - 1
        elif val < seuil:
            best = (pol, milieu, val)
            gauche = milieu + 1
        else:
            droite = milieu - 1
    return best



def remove_long_links(
    adata: AnnData,
    distance_percentile: float = 99.0,
    connectivity_key: str | None = None,
    distances_key: str | None = None,
    neighs_key: str | None = None,
    copy: bool = False,
) -> tuple[csr_matrix, csr_matrix] | None:
    """
    Remove links between cells at a distance bigger than a certain percentile of all positive distances.

    It is designed for data with generic coordinates.

    Parameters
    ----------
    %(adata)s

    distance_percentile
        Percentile of the distances between cells over which links are trimmed after the network is built.
    %(conn_key)s

    distances_key
        Key in :attr:`anndata.AnnData.obsp` where spatial distances are stored.
        Default is: :attr:`anndata.AnnData.obsp` ``['{{Key.obsp.spatial_dist()}}']``.
    neighs_key
        Key in :attr:`anndata.AnnData.uns` where the parameters from gr.spatial_neighbors are stored.
        Default is: :attr:`anndata.AnnData.uns` ``['{{Key.uns.spatial_neighs()}}']``.

    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`tuple` with the new spatial connectivities and distances matrices.

    Otherwise, modifies the ``adata`` with the following keys:
        - :attr:`anndata.AnnData.obsp` ``['{{connectivity_key}}']`` - the new spatial connectivities.
        - :attr:`anndata.AnnData.obsp` ``['{{distances_key}}']`` - the new spatial distances.
        - :attr:`anndata.AnnData.uns`  ``['{{neighs_key}}']`` - :class:`dict` containing parameters.
    """

    conns, dists = adata.obsp[connectivity_key], adata.obsp[distances_key]

    if copy:
        conns, dists = conns.copy(), dists.copy()

    threshold = np.percentile(np.array(dists[dists != 0]).squeeze(), distance_percentile)
    conns[dists > threshold] = 0
    dists[dists > threshold] = 0

    conns.eliminate_zeros()
    dists.eliminate_zeros()

    if copy:
        return conns, dists
    else:
        adata.uns[neighs_key]["params"]["radius"] = threshold
    # print(threshold)



# def shape_max(shapes):
#     max_area = 0
#     max_id = 0
#     for i in range(len(shapes.geoms)):
#         if shapes.geoms[i].area > max_area:
#             max_area = shapes.geoms[i].area
#             max_id = i
#     return shapes.geoms[max_id]
# IF MULTIPOLYGON ==> BUT BETTER WITH JUST MAX AREA
# shapeM = max(shape.geoms, key=lambda g: g.area)


# def count_points_in_polygon(points, polygon, include_boundary=True):
#     """
#     Count how many points lie inside (or on the boundary of) a polygon.

#     Args:
#         points (list of tuple): List of (x, y) points.
#         polygon_coords (list of tuple): List of (x, y) polygon vertices.
#         include_boundary (bool): If True, count points on boundary as inside.

#     Returns:
#         int: Number of points inside the polygon.
#     """
#     count = 0
#     for p in points:
#         point = Point(p)
#         if polygon.contains(point) or (include_boundary and polygon.touches(point)):
#             count += 1
#     return count
# count_points_in_polygon(list_points, shapeM)
# BUT BETTER WITH JUST sum du covers from shapely !!!
# sum(shapely.covers(shapeM, list_points))
# EVEN BETTER !!!!
# shapely.covers(shapeM, list_points).sum()



