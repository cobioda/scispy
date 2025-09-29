import math 
import matplotlib.pyplot as plt


def plot_shapes(
    poly,
    figsize=(5,5),
    ncols = 2,
):
    if poly.geom_type == 'Polygon':
        plt.figure(1, figsize=figsize)
        plt.plot(*poly.boundary.xy, c='red')
        plt.scatter(*poly.centroid.xy, c="blue", s=5)
        plt.title(f'threshold : {130}')
        plt.show()

    elif poly.geom_type == 'MultiPolygon':
        n = len(poly.geom_type)
        nrows = math.ceil(n / ncols)

        plt.figure(figsize=(13, 5 * nrows))
        plt.subplots_adjust(hspace =0.9, wspace=0.5)

        for n, pol in enumerate(poly.geoms):
            ax = plt.subplot(nrows, ncols, n+1)
            ax.plot(*pol.boundary.xy, c='red')
            ax.scatter(*pol.centroid.xy, c="blue", s=5)
            ax.set_title(f'Polygon {n+1}')
            ax.set_aspect('equal')
          
        plt.tight_layout()
        plt.show()
    
    else: 
        print('Shape is not a polygon or a multipolygon')


def plot_genes_expr_in_shapes(
  sdata,
#   along|orthogonal
    genes: list,
    group_by,
    cell_type,

):
    sdata



