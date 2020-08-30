import chart_studio.plotly as ply
from plotly.offline import iplot
import plotly.graph_objs as go
import numpy as np

from lidar_segmentation.segmentation import LidarSegmentationResult
from colorsys import hsv_to_rgb

from lidar_segmentation.detections import CLASS_NAMES


def label_colors(labels, bg_label=0, s=0.9, v=0.8):
    """
    Define a color for each label. Colors are picked in HSV color space.
    Parameters
    ----------
    labels: ndarray
        Array of all unique labels.
    bg_label: int
        The label that corresponds to the background.
        This label will be colored gray.
    s: float
        Saturation for HSV colors, between 0 and 1.0
    v: float
        Value for HSV colors, between 0 and 1.0

    Returns
    -------
    dict
        Dictionary mapping integer labels to RGB colors as [r,g,b]
    """
    # pick colors with random hue and fixed saturation, value
#    print(labels)
    labels = [l for l in labels if l != bg_label]
    hues = np.linspace(0, 0.2, len(labels))
    color_dict = {i: [255*c for c in hsv_to_rgb(h, s, v)]
                  for (i,h) in zip(labels, hues)}
    # set background to gray
    color_dict[bg_label] = [220, 220, 220]
    return color_dict


def plot_segmentation_result(results=None, label_type='class', size=1.2, labels=None, points=None):
    """

    Parameters
    ----------
    points: ndarray
    results: LidarSegmentationResult
    label_type: str
        Should be 'instance' or 'class'

    Returns
    -------
    None

    """
    if labels is None or points is None:
        points = results.points

        if label_type == 'instance':
            labels = results.instance_labels()
        elif label_type == 'class':
            labels = results.class_labels()
        else:
            raise ValueError("label_type argument must be 'instance' or 'class'")
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    bg_label=0
    color_dict = label_colors(unique_labels, bg_label)

    # Put label colors into format required by Plotly Marker colorscale
    plot_color_dict = {label: 'rgb(%d, %d, %d)' % tuple(color_dict[label])
                       for label in color_dict.keys()}
    #print(plot_color_dict)
    colorscale = None
    colors = [plot_color_dict[label] for label in labels]

    return plot_points(points, size=1.2, opacity=1.0,
                color=colors, colorscale=colorscale)
    
def plot_points(points, size=1.0, opacity=0.8, color=None, colorscale=None,):
    data = [pointcloud(points, size, opacity, color, colorscale)]
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
   # xy_lim = np.max(np.abs([np.min(x), np.max(x), np.min(y), np.max(y)]))
    xy_lim = 10
    zmin = -7
    zmax = zmin + 2*xy_lim
    layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0),
                       paper_bgcolor='rgb(0,0,0)',
                       scene=dict(aspectmode='auto',
                           xaxis=dict(title='x', range=[0,2*xy_lim],visible=False),
                                yaxis=dict(title='y', range=[-xy_lim,xy_lim],visible=False),
                                zaxis=dict(title='z', range=[zmin,zmax],visible=False),
                                camera=dict(eye=dict(x=-0.97,y=0.1,z=0))))
                       
    fig = go.Figure(data=data,layout=layout)
    
    #fig.show()
    return fig
def pointcloud(points, size=1.0, opacity=0.8, color=None, colorscale='Rainbow'):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    return go.Scatter3d(x=x,y=y,z=z, mode='markers',
                         marker=dict(size=size, opacity=opacity,
                                     color=color, colorscale=colorscale),
                         )
