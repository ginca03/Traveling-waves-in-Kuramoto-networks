#!/usr/bin/env python
""" Visualization functions

This module contains functions to visualize results.
"""

import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors
import matplotlib.animation as animation
import seaborn as sns
import pyvista as pv

plt.rcParams['image.cmap'] = 'cividis'


__author__ = "Dominik Koller"
__date__ = "02. August 2021"
__status__ = "Prototype"


def plot_brain_data(v, f, data, filename=None, cmap=None, clim=None, data_points=None, window_size=(2256, 912), glyph_scaling=0.01, return_plotter=False):
    """ Plot data projected on brain locations.
    
    This function plots data at the specified locations (vertices) of the mesh.
    
    Parameters
    ----------
    v : numpy.array
        #vertices x 3 array of vertex coordinates in 3D.
    f : numpy.array
        #faces x 3 array of faces indexing the vertices of a triangle.
    data : numpy.array
        #vertices array of data to project onto the brain.
    filename : String
        Filename for saving the output image.
    cmap : matplotlib.cm
        Matplotlib colormap to use for the data.
    clim : tuple
        Tuple of color limits
    data_points : numpy.array
        #vertices x 3 array of vertex coordinates in 3D for data positions. This can be used if the background mesh is not equal to the data point locations.
    window_size : (int, int)
        Window size for plotting and saving an image.
    glyph_scaling : float
        Float that specifies the scaling of the glyph data points.
        
    Returns
    -------
    p : pyvista.Plotter
        Pyvista plotter.
    """
    pv.set_plot_theme('document')

    # Build pyvista mesh
    mesh = pv.PolyData(v)

    # convert faces to pyvista
    mesh.faces = np.insert(f.flatten(), np.arange(0,len(f.flatten()),3), 3)
    
    # separate mesh and data points
    if data_points is not None:
        points = pv.PolyData(data_points)
    else:
        points = mesh.copy()
    
    # assign data
    points.point_data['data'] = data
    
    # create point glyphs
    glyphs = points.glyph(geom=pv.Sphere(), scale=False, factor=glyph_scaling)

    # fill medial wall hole
    mesh = mesh.fill_holes(1.).fill_holes(1.)

    # create plotter
    p = pv.Plotter(shape=(1,4), col_weights=[1,1,1,0.8], off_screen=True, notebook=True, border=False)

    # left
    p.subplot(0,0)
    p.add_mesh(mesh, color='white', point_size=0, show_scalar_bar=False, smooth_shading=True, ambient=0.2)
    p.add_mesh(glyphs, scalars='data', cmap=cmap, clim=clim, show_scalar_bar=False, lighting=False)

    p.camera_position = 'yz'
    p.camera.azimuth = 180
    p.camera.elevation = 0
    p.camera.zoom(2.5)

    # top
    p.subplot(0,1)
    p.add_mesh(mesh, color='white', point_size=0, show_scalar_bar=False, smooth_shading=True, ambient=0.2)
    p.add_mesh(glyphs, scalars='data', cmap=cmap, clim=clim, show_scalar_bar=False, lighting=False)

    p.camera_position = 'xy'
    p.camera.azimuth = 0
    p.camera.elevation = 0
    p.camera.zoom(2.5)

    # right
    p.subplot(0,2)
    p.add_mesh(mesh, color='white', point_size=0, show_scalar_bar=False, smooth_shading=True, ambient=0.2)
    p.add_mesh(glyphs, scalars='data', cmap=cmap, clim=clim, show_scalar_bar=False, lighting=False)

    p.camera_position = 'yz'
    p.camera.azimuth = 0
    p.camera.elevation = 0
    p.camera.zoom(2.5)

    # lighting
    light1 = pv.Light(light_type='scene light', intensity=0.6)
    light1.set_direction_angle(0, 0)
    p.add_light(light1)

    light2 = pv.Light(light_type='scene light', intensity=0.6)
    light2.set_direction_angle(0, 180)
    p.add_light(light2)

    light3 = pv.Light(light_type='scene light', intensity=0.6)
    light3.set_direction_angle(80, 90)
    p.add_light(light3)

    # colorbar
    p.subplot(0,3)
    #p.add_scalar_bar(height=0.7, width=0.6, vertical=True, position_x=0.4, position_y=0.15, n_labels=2, label_font_size=32)
    p.add_scalar_bar(height=0.55, width=0.3, vertical=True, position_x=0.15, position_y=0.20, n_labels=2, label_font_size=84, fmt='%.2f')
    
    if return_plotter:
        return p
    else:
        if filename:
            p.screenshot(filename, transparent_background=True, window_size=window_size)
        else:
            p.show(window_size=window_size)
        p.close()
    
    return


def plot_imshow_timeseries(image_data, samples, xlabels, ylabel, cblabel, cbticklabels, figsize=(15,1.5), position=[0.95, 0.11, 0.1, 0.77], aspect=8, pad=0.02, cmap='twilight', vmin=None, vmax=None, cbticks=None, xlabel_fontsize=None):
    """ Plot series of images.
    
    This function can be used to plot snapshots of a timeseries.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Array of images with shape (rows, columns, time).
    samples : numpy.array
        Array of samples that should be plotted.
    xlabels : list of str
        List of labels for each image. Usually timestamp.
    ylabel : str
        String for y label. 
    cblabel : str
        Colorbar label.
    cmap : String
        Colormap.
    vmin : numpy.float
        Minimum value for imshow.
    vmax : numpy.float
        Maximum value for imshow.
    cbticks : list
        List of ticks for colorbar.
    cbticklabels : list
        List of tick labels for colorbar. Must match the size of cbticks.  
    """
    n_samples = len(samples)
    fig, ax = plt.subplots(1,n_samples, figsize=figsize, dpi=300)
    
    for i in range(n_samples):
        im = ax[i].imshow(image_data[:,:,samples[i]], 
                    origin='lower', 
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax
                    )
        ax[i].set_xlabel(f"{xlabels[i]}", fontsize=xlabel_fontsize)
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].set_yticks([])
        ax[i].set_xticks([])

    ax[0].set_ylabel(ylabel)
    
    cbar_ax = fig.add_axes(position)
    cbar = fig.colorbar(im, ax=ax, cax=cbar_ax, ticks=cbticks, aspect=aspect, pad=pad)
    try:
        cbar.ax.set_yticklabels(cbticklabels)
        cbar.ax.set_ylabel(cblabel)
    except:
        cbar.ax.set_ylabel(cblabel)
        
    return


def imshow_movie(image_data, samples, interval, cblabel, cmap='twilight', vmin=None, vmax=None, cbticks=None, cbticklabels=None):
    """ Generate movie of imshows.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Array of images with shape (rows, columns, time).
    samples : numpy.array
        Array of samples used to create movie.
    interval : int
        Interval between frames.
    cblabel : str
        Colorbar label.
    cmap : String
        Colormap.
    vmin : numpy.float
        Minimum value for imshow.
    vmax : numpy.float
        Maximum value for imshow.
    cbticks : list
        List of ticks for colorbar.
    cbticklabels : list
        List of tick labels for colorbar. Must match the size of cbticks.
        
    Returns
    -------
    ani : matplotlib.animation
        Animation.
    
    """
    fig, ax = plt.subplots(1,1, figsize=(2.5,2.5), dpi=300)

    # initialize figure
    im = ax.imshow(image_data[:,:,0], 
                      origin='lower', 
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vmax,
                      animated=True
                      )
    left, right, bottom, top = im.get_extent()
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])

    cbar = plt.colorbar(im, ticks=cbticks, fraction=0.046, pad=0.04)
    try:
        cbar.ax.set_yticklabels(cbticklabels)
        cbar.ax.set_ylabel(cblabel)
    except:
        cbar.ax.set_ylabel(cblabel)
    
    plt.tight_layout()
    
    # update figure
    def update_data(num, im):
        im.set_data(image_data[:,:,num])
        
        return im

    ani = animation.FuncAnimation(fig, update_data, frames=samples, fargs=[im], interval=interval, blit=False)
    
    return ani
        

def make_brain_data_movie(v, f, data, time, start_time, end_time, filename, title=None, clim=None, cmap='cividis', window_size=(2256, 912), quality=4):
    """ Plot data projected on brain locations and make movie.
    
    This function plots data at the specified locations (vertices) of the mesh and writes them to a movie
    
    Parameters
    ----------
    v : numpy.array
        #vertices x 3 array of vertex coordinates in 3D.
    f : numpy.array
        #faces x 3 array of faces indexing the vertices of a triangle.
    data : numpy.array
        #vertices x time array of data to project onto the brain.
    time : numpy.array
        time array in ms.
    start_time : numpy.float
        start time for movie.
    end_time : numpy.float
        end time for movie.
    filename : String
        Filename for saving the output image.
    clim: tuple
        tuple that defines the color limits.
    cmap : matplotlib.cm
        Matplotlib colormap to use for the data.
    clim : tuple
        Tuple of color limits
    window_size : (int, int)
        Window size for plotting and saving an image.
    quality: int
        Quality of movie.
        
    Returns
    -------
    p : pyvista.Plotter
        Pyvista plotter.
    """
    pv.set_plot_theme('document')

    # Build pyvista mesh
    mesh = pv.PolyData(v)

    # convert faces to pyvista
    mesh.faces = np.insert(f.flatten(), np.arange(0,len(f.flatten()),3), 3)

    # assign data
    mesh.point_data['data'] = data[:,0]

    # create point glyphs
    glyphs = mesh.glyph(geom=pv.Sphere(), scale=False, factor=0.01)

    # fill medial wall hole
    mesh_filled = mesh.fill_holes(1.).fill_holes(1.)

    # create plotter
    p = pv.Plotter(shape=(1,4), col_weights=[1,1,1,0.4], off_screen=True, notebook=False, border=False)

    # Open a movie file
    p.open_movie(filename, quality=quality)

    # left
    p.subplot(0,0)
    p.add_mesh(mesh_filled, color='white', point_size=0, show_scalar_bar=False, smooth_shading=True, ambient=0.2)
    p.add_mesh(glyphs, scalars='data', cmap=cmap, clim=clim, show_scalar_bar=False, lighting=False)

    p.camera_position = 'yz'
    p.camera.azimuth = 180
    p.camera.elevation = 0
    p.camera.zoom(2.5)

    # top
    p.subplot(0,1)
    p.add_mesh(mesh_filled, color='white', point_size=0, show_scalar_bar=False, smooth_shading=True, ambient=0.2)
    p.add_mesh(glyphs, scalars='data', cmap=cmap, clim=clim, show_scalar_bar=False, lighting=False)

    p.camera_position = 'xy'
    p.camera.azimuth = 0
    p.camera.elevation = 0
    p.camera.zoom(2.5)

    # right
    p.subplot(0,2)
    p.add_mesh(mesh_filled, color='white', point_size=0, show_scalar_bar=False, smooth_shading=True, ambient=0.2)
    p.add_mesh(glyphs, scalars='data', cmap=cmap, clim=clim, show_scalar_bar=False, lighting=False)

    p.camera_position = 'yz'
    p.camera.azimuth = 0
    p.camera.elevation = 0
    p.camera.zoom(2.5)

    # lighting
    light1 = pv.Light(light_type='scene light', intensity=0.6)
    light1.set_direction_angle(0, 0)
    p.add_light(light1)

    light2 = pv.Light(light_type='scene light', intensity=0.6)
    light2.set_direction_angle(0, 180)
    p.add_light(light2)

    light3 = pv.Light(light_type='scene light', intensity=0.6)
    light3.set_direction_angle(80, 90)
    p.add_light(light3)

    # colorbar
    p.subplot(0,3)
    p.add_scalar_bar(title=title,height=0.7, width=0.6, vertical=True, position_x=0.4, position_y=0.15, n_labels=2, label_font_size=48, fmt='%.3f')

    p.show(auto_close=False, window_size=window_size)  # only necessary for an off-screen movie

    # Run through each frame
    p.write_frame()  # write initial data

    # get start sample
    start_sample = np.where((time>=start_time) & (time<=end_time))[0][0]

    # Update scalars on each frame
    for i, t in enumerate(time[(time>=start_time) & (time<=end_time)]):
        mesh.point_data["data"] = data[:,start_sample+i]
        p.add_text(f'{int(t-start_time)} ms', position='upper_left', font_size=26, name='text')

        # re-create point glyphs
        glyphs_new = mesh.glyph(geom=pv.Sphere(), scale=False, factor=0.01)
        glyphs.overwrite(glyphs_new)

        p.write_frame()  # Write this frame
        p.remove_actor('text')

    # Be sure to close the p when finished
    p.close()
    
    return


def create_brain_icon(v, f, figure_path):
    """ Create legend icon for correlation
    
    The differently colored hemispheres can be used instead of a legend in line plots.
    
    Parameters
    ----------
    v : numpy.array
        #vertices x 3 array of vertex coordinates in 3D.
    f : numpy.array
        #faces x 3 array of faces indexing the vertices of a triangle.
    figure_path : String
        The path where the icon should be saved.
        
    Returns
    -------
    """
    pv.set_plot_theme('document')

    cmap = sns.color_palette()
    cmap = matplotlib.colors.ListedColormap([cmap[1], cmap[0]])

    # Build pyvista mesh
    mesh_lh = pv.PolyData(v[:500])
    mesh_rh = pv.PolyData(v[500:])

    # convert faces to pyvista
    mesh_lh.faces = np.insert(f[:500].flatten(), np.arange(0,len(f[:500].flatten()),3), 3)
    mesh_rh.faces = np.insert(f[500:].flatten(), np.arange(0,len(f[500:].flatten()),3), 3)

    # create plotter
    p = pv.Plotter(off_screen=True, notebook=True, border=False, lighting='none')

    # top
    p.add_mesh(mesh_lh, color='#3baa70', point_size=0, show_scalar_bar=False, smooth_shading=True)
    p.add_mesh(mesh_rh, color='#703baa', point_size=0, show_scalar_bar=False, smooth_shading=True)

    p.camera_position = 'xy'
    p.camera.azimuth = 0
    p.camera.elevation = 0
    p.camera.zoom(1.)

    #p.show()
    p.screenshot(os.path.join(figure_path, 'left_right_brain.png'), transparent_background=True)
    p.close()
    
    return