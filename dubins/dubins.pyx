# Copyright (c) 2008-2014, Andrew Walker
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
cimport cython
cimport core
from libc.stdlib cimport malloc, free
# from libc.math import round

import numpy as np
cimport numpy as np

cdef inline int callback(double q[3], double t, void* f):
    '''Internal c-callback to convert values back to python
    '''
    qn = (q[0], q[1], q[2])
    return (<object>f)(qn, t)


cdef struct map_data:
    void * m # The map 
    int height
    int width
    float origin_x
    float origin_y
    float resolution
    int intersects
    int occupancy_threshold
    int unknown_threshold

cdef inline int map_callback(double q[3], double t, void* data_):
    '''Internal c-callback to convert values back to python
    '''
    # print "shape = ", (<object>data.m).shape
    # print "height = ", height
    # print "width = ", width
    cdef map_data * data = <map_data*> data_
    cdef int x = round((q[0] - data.origin_x)/data.resolution)
    cdef int y = round((q[1] - data.origin_y)/data.resolution)
    if (x < 0) or (y < 0) or (x >= data.width) or (y >= data.height):
        return 0
    m = <object>data.m
    data.intersects = data.intersects or (m[y,x] > data.occupancy_threshold) or (m[y,x] < data.unknown_threshold)
    return 0

LSL = 0
LSR = 1
RSL = 2
RSR = 3
RLR = 4
LRL = 5


# Extension point for pure python classes
cdef class _DubinsPath:
    cdef core.DubinsPath *ppth

    def __cinit__(self):
        self.ppth = <core.DubinsPath*>malloc(sizeof(core.DubinsPath))

    def __dealloc__(self):
        free(self.ppth)

    @staticmethod
    def shortest_path(q0, q1, rho):
        cdef double _q0[3]
        cdef double _q1[3]
        cdef double _rho = rho
        for i in [0, 1, 2]:
            _q0[i] = q0[i]
            _q1[i] = q1[i]

        path = _DubinsPath()
        code = core.dubins_shortest_path(path.ppth, _q0, _q1, _rho)
        if code != 0:
            raise RuntimeError('path did not initialise correctly')
        return path

    @staticmethod
    def path(q0, q1, rho, word):
        cdef double _q0[3]
        cdef double _q1[3]
        cdef double _rho = rho
        for i in [0, 1, 2]:
            _q0[i] = q0[i]
            _q1[i] = q1[i]
        path = _DubinsPath()
        code = core.dubins_path(path.ppth, _q0, _q1, _rho, word)
        if code != 0:
            return None
        return path

    def path_endpoint(self):
        cdef double _q0[3]
        code = core.dubins_path_endpoint(self.ppth, _q0)
        if code != 0:
            raise RuntimeError('endpoint not found')
        return (_q0[0], _q0[1], _q0[2])

    def path_length(self):
        '''Identify the total length of the path
        '''
        return core.dubins_path_length(self.ppth)

    def segment_length(self, i): 
        '''Identify the length of the i-th segment within the path
        '''
        return core.dubins_segment_length(self.ppth, i)

    def segment_length_normalized(self, i): 
        '''Identify the normalized length of the i-th segment within the path
        '''
        return core.dubins_segment_length_normalized(self.ppth, i)

    def path_type(self):
        '''Identify the type of path which applies 
        '''
        return core.dubins_path_type(self.ppth)

    def sample(self, t):
        '''Sample the path
        '''
        cdef double _q0[3]
        code = core.dubins_path_sample(self.ppth, t, _q0)
        if code != 0:
            raise RuntimeError('sample not found')
        return (_q0[0], _q0[1], _q0[2])

    def sample_many(self, step_size):
        '''Sample the entire path
        '''
        qs = []
        ts = []
        def f(q, t):
            qs.append(q)
            ts.append(t)
            return 0
        core.dubins_path_sample_many(self.ppth, step_size, callback, <void*>f)
        return qs, ts

    def sample_intersects(self, step_size, map_msg, occupancy_threshold, unknown_threshold):
        '''Sample in a grid
        '''

        cdef np.ndarray[np.int8_t, ndim=2] m = map_msg.data
        cdef map_data data
        data.m = <void *>m
        data.origin_x = map_msg.info.origin.position.x
        data.origin_y = map_msg.info.origin.position.y
        data.height = map_msg.info.height
        data.width = map_msg.info.width
        data.resolution = map_msg.info.resolution
        data.occupancy_threshold = occupancy_threshold
        data.unknown_threshold = unknown_threshold
        data.intersects = 0

        core.dubins_path_sample_many(self.ppth, step_size, map_callback, <void*>&data)

        # Check the endpoint
        cdef double _q0[3]
        core.dubins_path_endpoint(self.ppth, _q0)
        map_callback(_q0, 1., <void*>&data)

        return data.intersects

    def extract_subpath(self, t):
        '''Extract a subpath
        '''
        newpath = _DubinsPath()
        code = core.dubins_extract_subpath(self.ppth, t, newpath.ppth)
        if code != 0:
            raise RuntimeError('invalid subpath')
        return newpath

def shortest_path(q0, q1, rho):
    '''Shortest path between dubins configurations

    Parameters
    ----------
    q0 : array-like
        the initial configuration
    q1 : array-like
        the final configuration
    rho : float
        the turning radius of the vehicle

    Raises
    ------
    RuntimeError
        If the construction of the path fails

    Returns
    -------
    path : DubinsPath 
        The shortest path
    '''
    return _DubinsPath.shortest_path(q0, q1, rho) 
 

def path(q0, q1, rho, word):
    '''Find the Dubin's path for one specific word

    Parameters
    ----------
    q0 : array-like
        the initial configuration
    q1 : array-like
        the final configuration
    rho : float
        the turning radius of the vehicle
    word : int
        the control word (LSL, LSR, ...)

    Raises
    ------
    RuntimeError
        If the construction of the path fails

    Returns
    -------
    path : _DubinsPath 
        The path with the specified word (if one exists) or None
    '''
    return _DubinsPath.path(q0, q1, rho, word) 

def norm_path(alpha, beta, delta, word):
    '''Find the Dubin's path for one specific word assuming a normalized (alpha, beta, delta) frame

    Parameters
    ----------
    alpha : float
        the initial orientation 
    beta : flaot
        the final orientation
    delta : float
        the distance between configurations
    word : int
        the control word (LSL, LSR, ...)

    Raises
    ------
    RuntimeError
        If the construction of the path fails

    Returns
    -------
    path : DubinsPath 
        The path with the specified word (if one exists) or None
    '''
    q0 = [ 0.0, 0.0, alpha ]
    q1 = [ delta, 0.0, beta ]
    return path(q0, q1, 1.0, word)


