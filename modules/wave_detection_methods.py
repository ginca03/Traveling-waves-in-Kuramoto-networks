#!/usr/bin/env python
""" Wave Detection Methods

This module contains functions to detect traveling waves.
"""


import numpy as np
import igl
import potpourri3d as pp3d
import scipy as sp
from sksparse.cholmod import cholesky


__author__ = "Dominik Koller"
__date__ = "10. February 2021"
__status__ = "Prototype"



# Patch missing igl.rotate_vectors function
def rotate_vectors_2d(vectors, angles, b1, b2):
    """
    Rotate 2D vectors by given angles using local basis vectors.
    
    Parameters
    ----------
    vectors : numpy.array
        Array of vectors to rotate, shape (n, 3) where 3rd component should be 0
    angles : numpy.array  
        Array of rotation angles in radians (can be single value or array)
    b1 : numpy.array
        First basis vector for each face, shape (n, 3)
    b2 : numpy.array
        Second basis vector for each face, shape (n, 3)
        
    Returns
    -------
    rotated_vectors : numpy.array
        Rotated vectors, same shape as input
    """
    rotated = np.zeros_like(vectors)
    
    # Handle single angle case
    if np.isscalar(angles) or len(angles) == 1:
        angle = angles[0] if hasattr(angles, '__len__') else angles
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        for i in range(len(vectors)):
            # Extract 2D components in local basis
            v_b1 = np.dot(vectors[i], b1[i])
            v_b2 = np.dot(vectors[i], b2[i])
            
            # Rotate in 2D
            rotated_b1 = cos_a * v_b1 - sin_a * v_b2
            rotated_b2 = sin_a * v_b1 + cos_a * v_b2
            
            # Convert back to 3D
            rotated[i] = rotated_b1 * b1[i] + rotated_b2 * b2[i]
    else:
        # Handle multiple angles case
        for i in range(len(vectors)):
            angle = angles[i]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            # Extract 2D components in local basis
            v_b1 = np.dot(vectors[i], b1[i])
            v_b2 = np.dot(vectors[i], b2[i])
            
            # Rotate in 2D
            rotated_b1 = cos_a * v_b1 - sin_a * v_b2
            rotated_b2 = sin_a * v_b1 + cos_a * v_b2
            
            # Convert back to 3D
            rotated[i] = rotated_b1 * b1[i] + rotated_b2 * b2[i]
    
    return rotated

# Monkey patch igl if rotate_vectors doesn't exist
if not hasattr(igl, 'rotate_vectors'):
    igl.rotate_vectors = rotate_vectors_2d



# -------------------------------------------------------------------
# Patch igl.barycentric_coordinates_tri if missing
# -------------------------------------------------------------------
if not hasattr(igl, "barycentric_coordinates_tri"):
    def barycentric_coordinates_tri(P, F, V, *args, **kwargs):
        """
        Temporary Python replacement for igl.barycentric_coordinates_tri.

        Parameters
        ----------
        P : (n_points, 3) array of points
        F : (n_faces, 3) array of triangle indices
        V : (n_vertices, 3) array of vertex positions
        *args, **kwargs : ignored, for compatibility with C++ version

        Returns
        -------
        B : (n_points, 3) array of barycentric coordinates
        """
        B = np.zeros((len(P), 3))
        for i, p in enumerate(P):
            # Get the triangle vertices
            tri = V[F[i].astype(int)]
            v0 = tri[1] - tri[0]
            v1 = tri[2] - tri[0]
            v2 = p - tri[0]

            # Compute dot products
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)
            denom = d00 * d11 - d01 * d01
            if np.isclose(denom, 0.0):
            # Degenerate triangle → assign NaN or put everything on first vertex
                B[i] = [np.nan, np.nan, np.nan]
                continue

            # Compute barycentric coordinates
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            B[i] = [u, v, w]
        return B

    # Patch igl module
    igl.barycentric_coordinates_tri = barycentric_coordinates_tri



def k_ring_neighbours(source_vertex, k, adjacency_list):
    """ Calculate k-ring neighbours.
    
    Parameters
    ----------
    source_vertex : int
        Vertex from which the k-ring neighbours are computed.
    k : int
        Number of rings around source vertex.
    adjacency_list : list of lists
        List of vertices neighouring all other vertices.
    
    
    Returns
    -------
    neighbours : numpy.array
        Vertex indices of source vertex neighbours.
    
    """
    ring = [source_vertex]
    for ik in range(k):
        ring = np.unique(np.concatenate([adjacency_list[i] for i in ring]))
        
    ring = np.setdiff1d(ring, np.array([source_vertex]))
    return ring


def geodesic_neighbours(distance_solver, source_vertex, max_distance):
    """ Calculate approximate geodesic distance from source vertex to all other vertices of the mesh.
    
    Parameters
    ----------
    distance_solver : potpourri3d.mesh.MeshHeatMethodDistanceSolver
        pre-computed potpourri3d geodesic distance solver for meshes.
    source_vertex : int
        Index of the vertex from which the geodesic distance is computed.
    max_distance : numpy.float
        Maximum distance within which neighbouring vertices are selected.
        
    Returns
    -------
    neighbours : numpy.ndarray
        Indices of neighbouring vertices.
    """
    geodesic_distance = distance_solver.compute_distance(source_vertex)
    neighbours = np.setdiff1d(np.where(geodesic_distance <= max_distance), np.array([source_vertex]))
    
    return neighbours


def k_ring_boundary(v, f, k=0):
    """ Get k-ring boundary mask.
    
    Parameters
    ----------
    v : numpy.array
        #vertices x 3 array of vertex coordinates in 3D.
    f : numpy.array
        #faces x 3 array of faces indexing the vertices of a triangle.
    k : int
        Number of k-rings around boundary. k=0 selects the outermost vertices as the boundary.
        
    Returns
    -------
    boundary_mask : numpy.array
        Boolean mask of k-ring boundary vertices (True if vertex is part of boundary).
    """
    
    adj = igl.adjacency_matrix(f)
    
    if k==-1:
        boundary_mask = np.zeros(len(v), dtype=bool)
    else:
        boundary_mask = igl.is_border_vertex(v,f)
        for i in range(k):
            boundary_mask = (adj @ boundary_mask)!=0
        
    return boundary_mask


def compute_barycentric_coords(v, f):
    """
    Compute barycentric coords
    
    The barycentric coordinates are used for computing phase gradients.
    
    Parameters
    ----------
    v : numpy.array
        #vertices x 3 array of vertex coordinates in 3D.
    f : numpy.array
        #faces x 3 array of faces indexing the vertices of a triangle.
        
    Returns
    -------
    barycentric_coords : numpy.array
        #faces x 3 array of barycentric coordinates for each triangle.
    """
    bc = igl.barycenter(v, f)
    bc_coords = igl.barycentric_coordinates_tri(bc, 
                                                v[f][:,0].astype(np.double), 
                                                v[f][:,1].astype(np.double), 
                                                v[f][:,2].astype(np.double))
    
    return bc_coords


def compute_phase_gradient(phase, f, barycentric_coords, gradient_operator):
    """ Compute phase gradient
    
    Parameters
    ----------
    phase : numpy.array
        #vertices x time array of unwrapped instantaneous phases.
    f : numpy.array
        #faces x 3 array of faces indexing the vertices of a triangle.
    barycentric_coords : numpy.array
        #faces x 3 array of barycentric coordinates for each triangle (from libigl).
    gradient_operator : numpy.array
        #faces*3 x #vertices array corresponding to discrete gradient operator (from libigl).
        
    Returns
    -------
    phase_grad : numpy.array
        3 x #faces x time array of phase gradients.
    """  
    # complex phase gradient
    # ----------------------
    # complex exponential phase
    ce_phase = np.exp(1j*phase)

    # interpolate vertex values on triangle barycenters
    ce_phase_faces = np.sum(ce_phase.T[:,f] * barycentric_coords, axis=2).T

    # compute spatial gradient
    phase_grad_tmp = gradient_operator.dot(ce_phase).reshape(3, f.shape[0], phase.shape[1])
    phase_grad = np.real(-1j*ce_phase_faces.conj()*phase_grad_tmp)
    
    return phase_grad


def compute_wave_template(v, f, k=2, return_curl_template=False, use_geodesic_neighbourhood=False):
    """ Compute wave template 
    
    Computes an ideal diverging vector field for the k-ring neighbourhood of each vertex.
    
    Parameters
    ----------
    v : numpy.array
        #vertices x 3 array of vertex coordinates in 3D.
    f : numpy.array
        #faces x 3 array of faces indexing the vertices of a triangle.
    k : numpy.int or numpy.float
        This is either the k-ring distance to be used or the geodesic distance if use_geodesic_neighbourhood=True.
    return_curl_template : bool
        If true the a vector field template for a rotating waves is returned.
    use_geodesic_neighbourhood : bool
        If true the geodesic neighbourhood is used instead of the k-ring neighbourhood.
        
    Returns
    -------
    div_template : list of numpy.arrays
        #vertices x #k-ring-neighbour-triangles x 3 list of ideal diverging vector fields within the k-ring neighbourhood of each vertex.
    neighbours_faces : list of numpy.array
        #vertices x #k-ring-neighbour-triangles list of triangles within the k-ring neighbourhood of each vertex.
    """  
    n = len(v)
    
    # pre-compute gradient operator
    gradient_operator = igl.grad(v, f)
    
    # compute local basis
    b1, b2, _ = igl.local_basis(v, f)
    
    # pre-compute vertex adjacency
    adjacency = igl.adjacency_list(f)

    # pre-compute geodesic distance solver
    solver_distance = pp3d.MeshHeatMethodDistanceSolver(v, f)

    # pre-compute vertex face adjacency
    VF, NI = igl.vertex_triangle_adjacency(f, n)

    # compute wave gradient matching templates
    neighbours_faces = []
    curl_template = []
    div_template = []
    for vi in range(n):
        # compute geodesic distance
        geodesic_distance = solver_distance.compute_distance(vi)
        
        # retrieve neighbouring faces
        if not use_geodesic_neighbourhood:
            neighbours = k_ring_neighbours(vi, k, adjacency)  # to retrieve the faces within the k-ring neighbourhood, choose k to be k-1.
        else:
            neighbours = np.where(geodesic_distance<=k)[0]
        fs = []
        for ni in neighbours:
            fs.append(VF[NI[ni]:NI[ni+1]])
        fs = np.concatenate(fs, dtype=int)
        neighbours_faces.append(fs)
            
        # compute divergent template
        div_template_tmp = gradient_operator.dot(geodesic_distance).reshape(3,f.shape[0])[:,fs]
        div_template_mag = np.linalg.norm(div_template_tmp, axis=0)
        div_template.append((div_template_tmp / div_template_mag).T)
        
        if return_curl_template:
            # compute rotational template (divergent template rotated by 90 deg)
            curl_template.append(igl.rotate_vectors(np.array(div_template[vi], order='F'), np.array([np.pi/2.0]), np.array(b1[fs], order='F'), np.array(b2[fs], order='F')))


    if return_curl_template:
        return div_template, curl_template, neighbours_faces
    else:
        return div_template, neighbours_faces


def compute_angular_similarity(phase_grad, div_template, neighbours_faces, boundary_mask=None):
    """ Compute match between wave template and phase-gradient 
    
    Computes the angular similarity between the estimated phase-gradient and the wave template
    for the k-ring neighbourhood of each vertex.
    
    Parameters
    ----------
    phase_grad : numpy.array
        3 x #faces x time array of phase gradients.
    div_template : list of numpy.arrays
        #vertices x #k-ring-neighbour-triangles x 3 list of ideal diverging vector fields within the k-ring neighbourhood of each vertex.
    neighbours_faces : list of numpy.array
        #vertices x #k-ring-neighbour-triangles list of triangles within the k-ring neighbourhood of each vertex.
    boundary_mask : numpy.array
        Boolean array that indicates boundary vertices (True if vertex is part of boundary). Specifying this will exclude boundary vertices from analysis.
        
    Returns
    -------
    angular_similarity_div : numpy.array
        #vertices array of angular similarity between the estimated phase-gradient and the wave template for the k-ring neighbourhood of each vertex.
    """  
    number_of_timesteps = np.shape(phase_grad)[-1]
    if boundary_mask is not None:
        n = np.invert(boundary_mask).sum()
        vertex_idx = np.where(np.invert(boundary_mask))[0]
    else:
        n = len(div_template)
        vertex_idx = np.arange(n)
    
    # compute angular similarity
    angular_similarity_div = np.zeros((n, number_of_timesteps))

    phase_grad_norm = (phase_grad / np.linalg.norm(phase_grad, axis=0)).T  # compute normalized phase gradients
    for i, vi in enumerate(vertex_idx):
        fs = neighbours_faces[vi]

        # compute cosine similarity between template and phase gradients
        cos_sim_div = np.sum(-phase_grad_norm[:,fs] * div_template[vi], axis=2)
        angular_similarity_div[i] = np.mean(1 - 2*np.arccos(np.clip(cos_sim_div, -1.0, 1.0)) / np.pi, axis=1)
    
    return angular_similarity_div


def compute_helmholtz_hodge_decomposition(vector_field, v, f, return_rotational=False):
    """ Helmholtz-Hodge Decomposition

    Divergence and curl computations are based on Bhatia et al. (The Helmholtz-Hodge Decomposition — A Survey, 2013; page 1397)
    The Helmholtz-Hodge Decomposition is based on Bhatia et al. (2013; page 1392, equations 16)
    
    Parameters
    ----------
    vector_field : numpy.array
        time x #faces x 3 array of face-based vectors.
    v : numpy.array
        #vertices x 3 array of vertex coordinates in 3D.
    f : numpy.array
        #faces x 3 array of faces indexing the vertices of a triangle.
    return_rotational : bool
        If True the divergence-free potential is returned.
        
    Returns
    -------
    U : numpy.array
        #vertices x time curl-free scalar potential of Helmholtz-Hodge Decomposition.
    """
    # pre-compute
    # -----------
    gradient_operator = igl.grad(v, f)
    laplace_operator = igl.cotmatrix(v, f)
    cholesky_factor = cholesky(-laplace_operator, 1e-12)

    # compute divergence operator
    d_area = igl.doublearea(v, f)
    ta = sp.sparse.diags(np.hstack([d_area, d_area, d_area]) * 0.5)
    divergence_operator = -gradient_operator.T.dot(ta)
    
    # compute local basis
    b1, b2, _ = igl.local_basis(v, f)


    # Compute HHD per timepoint
    # -------------------------
    number_of_timesteps = vector_field.shape[0]
    U = np.zeros([number_of_timesteps, v.shape[0]])
    A = np.zeros([number_of_timesteps, v.shape[0]])

    # iterate over time
    for tp in range(number_of_timesteps):
        vector_field_tp = vector_field[tp,:,:]
        
        # Compute vertex-based divergence and curl
        div = divergence_operator.dot(vector_field_tp.flatten('F'))
        
        # solve for helmholtz-hodge decomposition potentials
        # curl-free potential
        U[tp] = cholesky_factor(div)
        
        if return_rotational:
            # rotate vector field
            vector_field_tp_rot = igl.rotate_vectors(np.array(vector_field_tp, order='F'), np.array([np.pi/2.0]), np.array(b1, order='F'), np.array(b2, order='F'))

            # Compute vertex-based divergence and curl
            curl = -divergence_operator.dot(vector_field_tp_rot.flatten('F'))
            
            # divergence-free potential
            A[tp] = cholesky_factor(curl)
        
    if return_rotational:
        return U.T, A.T
    else:
        return U.T
