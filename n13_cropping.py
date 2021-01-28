"""
Methods to identify artificial masking of the image.  
  cs = n13hough_lib.XRayStruct

Changelog:
  20201112: first stable version
"""
__version__ = 20201112

import numpy as np

def _max_consecutive_ones(vals):
    """
    calculate max line length
    # https://leetcode.com/problems/max-consecutive-ones/discuss/924225/Python3-faster-than-99.62
    """
    max_global = 0
    max_local = 0
    for num in vals:
        if num:
            max_local += 1
        else:
            max_global = max(max_global, max_local)
            max_local = 0
            
    max_global = max(max_global, max_local)
    return max_global


def _remove_borders(image, bounds, max_pixel_value, max_corner_value):
    """
    Remove borders around the phantom that are (partly) marked artificially.
    """
    [by0,bx0,by1,bx1] = bounds
    dimy,dimx = image.shape
    dmax = 5

    if max_pixel_value in [0, None]:
        max_pixel_value = np.max(image[by0:by1+1,bx0:bx1+1])

    # max corner
    if max_corner_value is None:
        max_corner_value = max([
            image[by0,bx0], 
            image[by0,bx1], 
            image[by1,bx1], 
            image[by1,bx0], 
        ])
    flagged_values = [0, max_pixel_value, max_corner_value]

    # first remove complete lines/rows if total = 0/max
    all_y = dimy*max_pixel_value
    all_x = dimx*max_pixel_value
    while np.sum(image[by0,:]) == 0 or np.sum(image[by0,:]) == all_x:
        if by0<dimy-1:
            by0 += 1
        else:
            break
    #max_pixel_value = np.max(image[by0:by1+1,bx0:bx1+1])

    while np.sum(image[by1,:]) == 0 or np.sum(image[by1,:]) == all_x:
        if by1>0:
            by1 -= 1
        else:
            break
    #max_pixel_value = np.max(image[by0:by1+1,bx0:bx1+1])

    while np.sum(image[:,bx0]) == 0 or np.sum(image[:,bx0]) == all_y:
        if bx0<dimx-1:
            bx0 += 1
        else:
            break
    #max_pixel_value = np.max(image[by0:by1+1,bx0:bx1+1])

    while np.sum(image[:,bx1]) == 0 or np.sum(image[:,bx1]) == all_y:
        if bx1>0:
            bx1 -= 1
        else:
            break
    #max_pixel_value = np.max(image[by0:by1+1,bx0:bx1+1])

    # now remove line or column if first or last value off
    if image[by0,bx0] in flagged_values:
        dx = 0
        dy = 0
        while image[by0,bx0+dx] in flagged_values:
            if bx0+dx<dimx-1:
                dx += 1
            else:
                break 
        while image[by0+dy,bx0] in flagged_values:
            if by0+dy<dimy-1:
                dy += 1
            else:
                break 
        if dx > dmax and dy >dmax:
            # try to remove lines where both x and y give practically all
            bx0 += 1
            by0 += 1
        elif dx < dy:
            bx0 += min(dx,dmax)
        else:
            by0 += min(dy,dmax)

    if image[by1,bx0] in flagged_values:
        dx = 0
        dy = 0
        while image[by1,bx0+dx] in flagged_values:
            if bx0+dx<dimx-1:
                dx += 1
            else:
                break 
        while image[by1-dy,bx0] in flagged_values:
            if by1-dy>0:
                dy += 1
            else:
                break
        if dx > dmax and dy >dmax:
            # try to remove lines where both x and y give practically all
            bx0 += 1
            by1 -= 1
        elif dx < dy:
            bx0 += min(dx,dmax)
        else:
            by1 -= min(dy,dmax)

    if image[by0,bx1] in flagged_values:
        dx = 0
        dy = 0
        while image[by0,bx1-dx] in flagged_values:
            if bx1-dx>0:
                dx += 1
            else:
                break
        while image[by0+dy,bx1] in flagged_values:
            if by0+dy<dimy-1:
                dy += 1
            else:
                break 
        if dx > dmax and dy >dmax:
            # try to remove lines where both x and y give practically all
            bx1 -= 1
            by0 += 1
        elif dx < dy:
            bx1 -= min(dx,dmax)
        else:
            by0 += min(dy,dmax)

    if image[by1,bx1] in flagged_values:
        dx = 0
        dy = 0
        while image[by1,bx1-dx] in flagged_values:
            if bx1-dx>0:
                dx += 1
            else:
                break
        while image[by1-dy,bx1] in flagged_values:
            if by1-dy>0:
                dy += 1
            else:
                break
        if dx > dmax and dy >dmax:
            # try to remove lines where both x and y give practically all
            bx1 -= 1
            by1 -= 1
        elif dx < dy:
            bx1 -= min(dx,dmax)
        else:
            by1 -= min(dy,dmax)

    return [by0, bx0, by1, bx1]
    
def remove_borders(image, max_pixel_value):
    """
    try to remove the borders of the image; if they are 0 or max
    """
    dimy,dimx = image.shape
    bx0=0
    bx1=dimx-1
    by0=0
    by1=dimy-1

    # the values to flag as "man_made_border" are 0, max_possible_pixel_value, current max corner value
    max_corner_value = max([
        min(image[by0,bx0], max_pixel_value), 
        min(image[by0,bx1], max_pixel_value), 
        min(image[by1,bx1], max_pixel_value), 
        min(image[by1,bx0], max_pixel_value), 
    ])

    bounds = [by0,bx0,by1,bx1]
    bounds_new = _remove_borders(image, bounds, max_pixel_value, max_corner_value)

    while bounds_new != bounds:
        bounds = bounds_new
        bounds_new = _remove_borders(image, bounds, max_pixel_value, max_corner_value)
        [by0,bx0,by1,bx1] = bounds

    return bounds

def crop_border(data, threshold_len):
    """
    remove border if too much 1
    """
    dimy,dimx = data.shape
    cropping = False
    bx0=0
    bx1=dimx-1
    by0=0
    by1=dimy-1

    # check if number of whites keeps increasing
    if 1:
        whites = np.sum(data,axis=0) # per x
    else:
        whites = [ _max_consecutive_ones(data[:,i]) for i in range(dimx) ]
    while whites[bx0+1]>whites[bx0]:
        if bx0<dimx-2:
            bx0 += 1
            cropping = True
        else:
            break
    while whites[bx1-1]>whites[bx1]:
        if bx1>1:
            bx1 -= 1
            cropping = True
        else:
            break
                
    if 1:
        whites = np.sum(data,axis=1) # per y
    else:
        whites = [ _max_consecutive_ones(data[i,:]) for i in range(dimy) ]
    while whites[by0+1]>whites[by0]:
        if by0<dimy-2:
            by0 += 1
            cropping = True
        else:
            break
    while whites[by1-1]>whites[by1]:
        if by1>1:
            by1 -= 1
            cropping = True
        else:
            break
        
    # remove border parts with too much stuff
    while np.sum(data[by0,:])> dimx-threshold_len:
        by0 +=1 
        cropping = True
    while np.sum(data[by1,:])> dimx-threshold_len:
        by1 -=1 
        cropping = True
    while np.sum(data[:,bx0])> dimy-threshold_len:
        bx0 +=1 
        cropping = True
    while np.sum(data[:,bx1])> dimy-threshold_len:
        bx1 -=1 
        cropping = True
        
    if cropping:
        bounds = [by0,bx0, by1,bx1]
        return bounds
    
    return None

