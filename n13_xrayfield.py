"""
Abstract:
XRayField
  Start from corner points of inner square, and travel outwards. As soon as value goes above 
  max Cu step or below min Cu step, the edge of the x-ray is defined. Return the max distance
  found for each side.
  
  cs = n13hough_lib.XRayStruct

Changelog:
  20210112: first stable version

"""
__version__ = 20210112

import numpy as np
import scipy.ndimage as scind
import matplotlib.pyplot as plt
try:
    # wad2.0 runs each module stand alone
    import n13_cropping as crop
except ImportError:
    from . import n13_cropping as crop


def XRayField(cs, workim=None, sigma=5):
    """
    Find edges of XRay exposure along the lines of the grid box.
    Use either min across line between box and edge, or use corner values
    """
    mask_max_corner = False
    # remove noise; alternatively sample more pixels and average.
    if workim is None:
        borders_remove = False
        if borders_remove:
            [by0, bx0, by1, bx1] = crop.remove_borders(cs.pixeldataIn, cs.get_max_pixel_value())
            #cs.hough_crop = [by0, bx0, by1, bx1]
            cropim = np.zeros_like(cs.pixeldataIn)
            cropim[by0:by1+1,bx0:bx1+1] = cs.pixeldataIn[by0:by1+1,bx0:bx1+1]
            workim = unif.LocalSNR(cropim, sigma, bksigma = None, uiobject=None)
            workim[  0:by0+1,:] = 0
            workim[by1:,     :] = 0
            workim[:,   0:bx0+1] = 0
            workim[:, bx1:     ] = 0
        else:
            # this works much better and more resembles what you would do visually
            max_pix_val = cs.get_max_pixel_value()
            workim = cs.pixeldataIn
            if mask_max_corner:
                # do not use; breaks stuff
                # the values to flag as "man_made_border" are 0, max_possible_pixel_value, current max corner value
                max_corner_value = max([
                    min(workim[0,0], max_pix_val), 
                    min(workim[0,-1], max_pix_val), 
                    min(workim[-1,-1], max_pix_val), 
                    min(workim[-1,0], max_pix_val), 
                ])
        
                bmask = np.bitwise_or(workim==0, np.bitwise_or(workim==max_pix_val, workim==max_corner_value))
            else:
                bmask = np.bitwise_or(workim==0, workim==max_pix_val)
            cropim = np.zeros_like(workim)
            cropim[:,:] = workim[:,:]
            cropim[bmask] = 0
            workim = scind.gaussian_filter(cropim, sigma=sigma)
                
            workim[bmask] = 0
            
        # use cu wedge to define values for edge of x-ray
        if not len(cs.cuwedge.step_mean) == 7:
            cu_miny = np.min([y for y,x in cs.cuwedge.box_roi])
            cu_maxy = np.max([y for y,x in cs.cuwedge.box_roi])
            cu_minx = np.min([x for y,x in cs.cuwedge.box_roi])
            cu_maxx = np.max([x for y,x in cs.cuwedge.box_roi])
    
            cu_range = np.average(workim[cu_miny:cu_maxy,cu_minx:cu_maxx], axis=1)
            cu_min, cu_max = np.min(cu_range), np.max(cu_range)
            cu_min,cu_max = cu_min+.1*(cu_max-cu_min),cu_min+.9*(cu_max-cu_min)
        else:
            #cu_min, cu_max = cs.cuwedge.step_mean[1], cs.cuwedge.step_mean[-2]
            cu_min_max = []
            for i in [1,-2]:
                steproi = cs.cuwedge.step_rois[i]
                xlo = min([x for x,y in steproi])
                xhi = max([x for x,y in steproi])
                ylo = min([y for x,y in steproi])
                yhi = max([y for x,y in steproi])
                cu_min_max.append(np.mean(workim[xlo:xhi,ylo:yhi]))
            cu_min, cu_max = cu_min_max

    error = False

    xr_NSWEmm = []
    # north- and southside
    xr_NSWEmm.append(FindXRayEdgeMOD(cs, 'N', workim,cu_min, cu_max, rawim=cropim))
    xr_NSWEmm.append(FindXRayEdgeMOD(cs, 'S', workim,cu_min, cu_max, rawim=cropim))
    xr_NSWEmm.append(FindXRayEdgeMOD(cs, 'W', workim,cu_min, cu_max, rawim=cropim))
    xr_NSWEmm.append(FindXRayEdgeMOD(cs, 'E', workim,cu_min, cu_max, rawim=cropim))
    if min(xr_NSWEmm)<1.:
        error = True
    else:
        error = False

    print('Edge [N/S/W/E] cm = %.1f %.1f %.1f %.1f' % (xr_NSWEmm[0]/10., 
                                                       xr_NSWEmm[1]/10., 
                                                       xr_NSWEmm[2]/10., 
                                                       xr_NSWEmm[3]/10. ))
    xr_roi = [] #[ UL, LL, LR, UR]
    xco, yco = cs.geom.phantomposmm2pix(-xr_NSWEmm[2],  xr_NSWEmm[0])
    xr_roi.append([xco, yco])
    xco, yco = cs.geom.phantomposmm2pix( xr_NSWEmm[3],  xr_NSWEmm[0])
    xr_roi.append([xco, yco])
    xco, yco = cs.geom.phantomposmm2pix( xr_NSWEmm[3], -xr_NSWEmm[1])
    xr_roi.append([xco, yco])
    xco, yco = cs.geom.phantomposmm2pix(-xr_NSWEmm[2], -xr_NSWEmm[1])
    xr_roi.append([xco, yco])

    #xr_roi = [ [y+by0,x+bx0] for y,x in xr_roi]

    # copy to geom struct
    cs.geom.xr_roi = xr_roi
    cs.geom.xr_NSWEmm = xr_NSWEmm

    return error

def FindXRayEdgeMOD(cs, side, workim, cu_min,cu_max, rawim=None):
    """
    the ok version about to be modified
    """
    # travel from center to edges
    # workim is a prepared image, getting rid of noise and other stuff
    widthpx, heightpx  = np.shape(workim) ## width/height in pixels
    #rawim=None
    outvalue = cs.forceRoom.outvalue
    # for DiDi, just take the minimal corner value
    if outvalue<0:
        outvalue = min(workim[0][0], workim[-1][0],workim[0][-1],workim[-1][-1])

    """
    0 ll [int(immidx-rad/2+.5),int(immidy-rad/2+.5)],
    1 ul [int(immidx-rad/2+.5),int(immidy+rad/2+.5)],
    2 ur [int(immidx+rad/2+.5),int(immidy+rad/2+.5)],
    3 lr [int(immidx+rad/2+.5),int(immidy-rad/2+.5)] ]
    """
    # north- and southside
    baseidmax = 1
    baseidmin = 0
    horizontal = True
    startmax = False
    if side == 'N':
        baseids = [ [1,0],[2,3] ]
        horizontal = False
        startmax = False
    elif side == 'S':
        baseids = [ [1,0],[2,3] ]
        horizontal = False
        startmax = True
    elif side == 'E':
        baseids = [ [3,0],[2,1] ]
        horizontal = True
        startmax = True
    else: #(side == 'W'):
        baseids = [ [3,0],[2,1] ]
        horizontal = True
        startmax = False

    found = False
    edgemm = []
    for ba in baseids:
        baseidmax = ba[0]
        baseidmin = ba[1]
        if horizontal:
            dypx = 1.*(cs.geom.box_roi[baseidmax][1]-cs.geom.box_roi[baseidmin][1])/(cs.geom.box_roi[baseidmax][0]-cs.geom.box_roi[baseidmin][0]) # diff in x if 1 px to y
            useboxradmm = cs.geom.box_radmm[0]
        else:
            dxpy = 1.*(cs.geom.box_roi[baseidmax][0]-cs.geom.box_roi[baseidmin][0])/(cs.geom.box_roi[baseidmax][1]-cs.geom.box_roi[baseidmin][1]) # diff in x if 1 px to y
            useboxradmm = cs.geom.box_radmm[1]
        posvec = []
        valvec = []
        rawvalvec = [] # on original, allow for shrinkage
        id = 0
        inrange = True
        while inrange:
            if horizontal:
                if startmax:
                    xpos = id
                else:
                    xpos = -id
                ypos = dypx*xpos
            else:
                if startmax:
                    ypos = id
                else:
                    ypos = -id
                xpos = dxpy*ypos

            pos = np.sqrt(xpos*xpos+ypos*ypos)

            # start from maxpoint, and increase with double linear interpolation
            if startmax:
                x0 = int(cs.geom.box_roi[baseidmax][0] + xpos)
                y0 = int(cs.geom.box_roi[baseidmax][1] + ypos)
            else:
                x0 = int(cs.geom.box_roi[baseidmin][0] +xpos)
                y0 = int(cs.geom.box_roi[baseidmin][1] +ypos)
            if xpos<0:
                x0 -= 1
            x1 = x0+1
            if x0<0 or x1>(widthpx-1) or y0<0 or y0>(heightpx-1):
                inrange = False
                if len(valvec)==0:
                    xa = 0 if x0<0 else widthpx-1 
                    ya = 0 if y0<0 else heightpx-1
                    posvec.append(pos)
                    valvec.append(int(workim[xa,ya]))
                    if not rawim is None:
                        rawvalvec.append(int(rawim[xa,ya]))
                break

            if not rawim is None:
                val00 = int(rawim[x0,y0])
                val10 = int(rawim[x1,y0])
                val05 = 1.*val00+(xpos-(int)(xpos))*(val10-val00)
                rawvalvec.append(val05)

            val00 = int(workim[x0,y0])
            val10 = int(workim[x1,y0])
            val05 = 1.*val00+(xpos-(int)(xpos))*(val10-val00)
            posvec.append(pos)
            valvec.append(val05)
                
            if np.abs(val00 -outvalue)<1.:
                break
            if int(val05) < 1:
                break
            if valvec[0]<outvalue and val00>outvalue:
                break
            if valvec[0]>outvalue and val00<outvalue:
                break
            
            id += 1

        #minval = min(valvec)
        minval = min((min(valvec),outvalue))
        maxval = max((max(valvec),outvalue)) # outvalue might not be present in this direction
        meanval = np.mean(valvec)
            
        if meanval < outvalue: # do not include outvalue parts in mean
            npa = np.array(valvec)
            meanval = np.mean( npa[npa<outvalue] )
        elif meanval > outvalue:
            npa = np.array(valvec)
            meanval = np.mean( npa[npa>outvalue] )
            
        if outvalue < meanval: # looking for a low value
            lab = "low"
        else:
            lab = "high"
        threshLow = (9.*minval+meanval)/10.
        threshHigh = (9.*maxval+meanval)/10.

        if not None in [cu_min, cu_max]:
            # define end as below cu_min or above cu_max
            threshLow = cu_min
            threshHigh = cu_max

        # look for the drop or rise
        found = False
        for ix, (p,v) in enumerate(zip(posvec,valvec)):
            if valvec[0]>threshLow and v<threshLow:
                found = True
            elif valvec[0]<threshHigh and v>threshHigh:
                found = True
            if found:
                if not rawim is None: # try to exclude constant region outside which is now blurred.
                    usep = p
                    for _ix in reversed(range(0,ix)):
                        if rawvalvec[_ix] == rawvalvec[ix]:
                            usep = posvec[_ix]
                        else:
                            break
                    p = usep
                edgemm.append( cs.pix2phantommm(p)+useboxradmm )
                if cs.verbose:
                    plt.plot(cs.pix2phantommm(p)+useboxradmm,v,'bo')
                    if not rawim is None:
                        plt.plot(cs.pix2phantommm(p)+useboxradmm,rawvalvec[ix],'bo')
                    cs.hasmadeplots = True
                break
        
        if not found: # add max edge pos
            edgemm.append( cs.pix2phantommm(max(posvec))+useboxradmm )


        if cs.verbose:
            print("out/mean", outvalue, meanval)
            edgepos = edgemm[-1]
            plt.figure()
            pps = [cs.pix2phantommm(p)+useboxradmm for p in posvec]
            plt.plot(pps, valvec, label="val")
            #plt.plot(pps, scind.median_filter(valvec, 7, mode="reflect"), label="mval")
            plt.plot([pps[0],pps[-1]], [cu_min,cu_min], 'r--', label="cu")
            plt.plot([pps[0],pps[-1]], [cu_max,cu_max], 'r--', label="cu")
            plt.plot([pps[0],pps[-1]], [threshLow,threshLow], '--', color="orange", label="thresh")
            plt.plot([pps[0],pps[-1]], [threshHigh,threshHigh], '--', color="orange", label="thresh")
            if 0 and not rawim is None:
                plt.plot(pps, rawvalvec, label="raw")
                plt.plot(pps, scind.median_filter(rawvalvec, 7, mode="reflect"), label="mraw")
                plt.plot(pps, scind.gaussian_filter1d(rawvalvec, sigma=1, order=1, mode="reflect"), label="drval")
            plt.title("{}{} {}/{} {}".format(lab, side, threshLow,threshHigh, edgepos))
            plt.plot( [edgepos,edgepos], [min(valvec), max(valvec)], label="hit" )
            plt.legend()
            cs.hasmadeplots = True
            #plt.show()

    return max(edgemm)

def FindXRayEdgeOK(cs, side, workim, cu_min,cu_max, rawim=None):
    """
    Unmodified version
    """
    # travel from center to edges
    # workim is a prepared image, getting rid of noise and other stuff
    widthpx, heightpx  = np.shape(workim) ## width/height in pixels
    #rawim=None
    outvalue = cs.forceRoom.outvalue
    # for DiDi, just take the minimal corner value
    if outvalue<0:
        outvalue = min(workim[0][0], workim[-1][0],workim[0][-1],workim[-1][-1])

    """
    0 ll [int(immidx-rad/2+.5),int(immidy-rad/2+.5)],
    1 ul [int(immidx-rad/2+.5),int(immidy+rad/2+.5)],
    2 ur [int(immidx+rad/2+.5),int(immidy+rad/2+.5)],
    3 lr [int(immidx+rad/2+.5),int(immidy-rad/2+.5)] ]
    """
    # north- and southside
    baseidmax = 1
    baseidmin = 0
    horizontal = True
    startmax = False
    if side == 'N':
        baseids = [ [1,0],[2,3] ]
        horizontal = False
        startmax = False
    elif side == 'S':
        baseids = [ [1,0],[2,3] ]
        horizontal = False
        startmax = True
    elif side == 'E':
        baseids = [ [3,0],[2,1] ]
        horizontal = True
        startmax = True
    else: #(side == 'W'):
        baseids = [ [3,0],[2,1] ]
        horizontal = True
        startmax = False

    found = False
    edgemm = []
    for ba in baseids:
        baseidmax = ba[0]
        baseidmin = ba[1]
        if horizontal:
            dypx = 1.*(cs.geom.box_roi[baseidmax][1]-cs.geom.box_roi[baseidmin][1])/(cs.geom.box_roi[baseidmax][0]-cs.geom.box_roi[baseidmin][0]) # diff in x if 1 px to y
            useboxradmm = cs.geom.box_radmm[0]
        else:
            dxpy = 1.*(cs.geom.box_roi[baseidmax][0]-cs.geom.box_roi[baseidmin][0])/(cs.geom.box_roi[baseidmax][1]-cs.geom.box_roi[baseidmin][1]) # diff in x if 1 px to y
            useboxradmm = cs.geom.box_radmm[1]
        posvec = []
        valvec = []
        rawvalvec = [] # on original, allow for shrinkage
        id = 0
        inrange = True
        while inrange:
            if horizontal:
                if startmax:
                    xpos = id
                else:
                    xpos = -id
                ypos = dypx*xpos
            else:
                if startmax:
                    ypos = id
                else:
                    ypos = -id
                xpos = dxpy*ypos

            pos = np.sqrt(xpos*xpos+ypos*ypos)

            # start from maxpoint, and increase with double linear interpolation
            if startmax:
                x0 = int(cs.geom.box_roi[baseidmax][0] + xpos)
                y0 = int(cs.geom.box_roi[baseidmax][1] + ypos)
            else:
                x0 = int(cs.geom.box_roi[baseidmin][0] +xpos)
                y0 = int(cs.geom.box_roi[baseidmin][1] +ypos)
            if xpos<0:
                x0 -= 1
            x1 = x0+1
            if x0<0 or x1>(widthpx-1) or y0<0 or y0>(heightpx-1):
                inrange = False
                if len(valvec)==0:
                    xa = 0 if x0<0 else widthpx-1 
                    ya = 0 if y0<0 else heightpx-1
                    posvec.append(pos)
                    valvec.append(int(workim[xa,ya]))
                    if not rawim is None:
                        rawvalvec.append(int(rawim[xa,ya]))
                break

            if not rawim is None:
                val00 = int(rawim[x0,y0])
                val10 = int(rawim[x1,y0])
                val05 = 1.*val00+(xpos-(int)(xpos))*(val10-val00)
                rawvalvec.append(val05)

            val00 = int(workim[x0,y0])
            val10 = int(workim[x1,y0])
            val05 = 1.*val00+(xpos-(int)(xpos))*(val10-val00)
            posvec.append(pos)
            valvec.append(val05)
                
            if val00 == outvalue:
                break
            id += 1

        #minval = min(valvec)
        minval = min((min(valvec),outvalue))
        maxval = max((max(valvec),outvalue)) # outvalue might not be present in this direction
        meanval = np.mean(valvec)
            
        if meanval < outvalue: # do not include outvalue parts in mean
            npa = np.array(valvec)
            meanval = np.mean( npa[npa<outvalue] )
        elif meanval > outvalue:
            npa = np.array(valvec)
            meanval = np.mean( npa[npa>outvalue] )
            
        if outvalue < meanval: # looking for a low value
            lab = "low"
        else:
            lab = "high"
        threshLow = (9.*minval+meanval)/10.
        threshHigh = (9.*maxval+meanval)/10.

        if not None in [cu_min, cu_max]:
            # define end as below cu_min or above cu_max
            threshLow = cu_min
            threshHigh = cu_max

        if cs.verbose:
            plt.figure()
            pps = [cs.pix2phantommm(p)+useboxradmm for p in posvec]
            plt.plot(pps, valvec, label="val")
            plt.plot(pps, scind.median_filter(valvec, 7, mode="reflect"), label="mval")
            plt.plot([pps[0],pps[-1]], [cu_min,cu_min], 'r--')
            plt.plot([pps[0],pps[-1]], [cu_max,cu_max], 'r--')
            if 0 and not rawim is None:
                plt.plot([cs.pix2phantommm(p)+useboxradmm for p in posvec],rawvalvec, label="raw")
                plt.plot(pps, scind.median_filter(rawvalvec, 7, mode="reflect"), label="mraw")
                plt.plot(pps, scind.gaussian_filter1d(rawvalvec, sigma=1, order=1, mode="reflect"), label="drval")
            plt.title(side+" "+str(threshLow)+" "+str(threshHigh)+" "+lab)
            plt.legend()
            cs.hasmadeplots = True
            #plt.show()
            
        found = False
        for ix, (p,v) in enumerate(zip(posvec,valvec)):
            if outvalue < meanval and v<threshLow:
                found = True
            elif outvalue >= meanval and v>threshHigh:#threshLow:
                found = True
            if found:
                if not rawim is None: # try to exclude constant region outside which is now blurred.
                    usep = p
                    for _ix in reversed(range(0,ix)):
                        if rawvalvec[_ix] == rawvalvec[ix]:
                            usep = posvec[_ix]
                        else:
                            break
                    p = usep
                edgemm.append( cs.pix2phantommm(p)+useboxradmm )
                if cs.verbose:
                    plt.plot(cs.pix2phantommm(p)+useboxradmm,v,'bo')
                    if not rawim is None:
                        plt.plot(cs.pix2phantommm(p)+useboxradmm,rawvalvec[ix],'bo')
                    cs.hasmadeplots = True
                break
        
        if 0 and not found:
            # not found. 
            
            for ix, (p,v) in enumerate(zip(posvec,valvec)):
                if v>=threshHigh:
                    found = True
                    if not rawim is None: # try to exclude constant region outside which is now blurred.
                        usep = p
                        for _ix in reversed(range(0,ix)):
                            if rawvalvec[_ix] == rawvalvec[ix]:
                                usep = posvec[_ix]
                            else:
                                break
                        p = usep
                    edgemm.append( cs.pix2phantommm(p)+useboxradmm )
                    if cs.verbose:
                        plt.plot(p,v,'bo')
                        cs.hasmadeplots = True
                    break

        if not found: # add max edge pos
            edgemm.append( cs.pix2phantommm(max(posvec))+useboxradmm )


    return max(edgemm)

