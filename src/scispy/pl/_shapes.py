import matplotlib.pyplot as plt

def plot_shapes(
    shape, 
    title = "",
    only_max = False,
    plot_points = False,
    return_y_axis = True,
    figsize=(5,5),
):
    _, ax = plt.subplots(figsize=figsize)
    
    if (shape.geom_type == 'MultiPolygon') &  (only_max):
        shape = max(shape.geoms, key=lambda g: g.area)
    
    if shape.geom_type == 'Polygon':
        x, y = shape.exterior.xy
        ax.plot(x, y, color="blue")
        ax.fill(x, y, alpha=0.3, color="lightblue")

        if plot_points:
            ax.scatter(x, y, c="blue", s=5)
    elif shape.geom_type == 'MultiPolygon':
        for pol in shape.geoms:
            x, y = pol.exterior.xy
            ax.plot(x, y, color="blue")
            ax.fill(x, y, alpha=0.3, color="lightblue")

            if plot_points:
                ax.scatter(x, y, c="blue", s=5)

    if return_y_axis:
        ax.invert_yaxis()
    
    plt.title(title)
    plt.show()

