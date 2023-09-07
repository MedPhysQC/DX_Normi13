"""
Abstract:
FindPhantomGridHough
  Use Hough line detector to find phantom horizontal and vertical grid lines. Of those grid line, 
  use the two middle ones to define the inner square.
XRayField
  Start from corner points of inner square, and travel outwards. As soon as value goes above 
  max Cu step or below min Cu step, the edge of the x-ray is defined. Return the max distance
  found for each side.
  
Implementation:
FindPhantomGridHough
  0. Preprocessing. 
    a. Bitshift towards 11 bits; this removes noise but keeps min/max meaning almost 
       the same. This is needed to get the same kind of response (local SNR) for 12 and 15 bit images.
    b. Remove part of image that is blocked (either 0 or maxval)
    c. Use LocalSNR with a small sigma; 0.25mm to get sharp lines.
  1. Define grid lines.
    a. Binarize the image; remove parts with too low response (noise in background) or too high 
       response (objects in phantom), zero noise parts.
    b. Crop the image by removing rows/columns that are completely True (outside) or still increasing
       in number of Trues.
    c. Fancy edge detection. Sobel and Canny and friends find two edges around each line. Not very good.
       Best result with "ridge detectors". Of those meijering keeps best lines (does not create gaps at
       grid crossings).
    d. Binarize ridges. Remove low ridge responses.
  2. Hough line detection of almost vertical lines.
    a. Sort on strength. Keep only a certain number of strong lines (not very important).
    b. Turn lines into coordinate pairs.
    c. Split into lines left from the middle and right from the middle.
    d. Group lines that are very close to each other; take (the median line or) the strongest one.
    e. Remove the copper wedge line if it is detected.
  3. Repeat previous step with transposed image to find horizontal lines.
  4. Take the inner linepairs for both directions to define the crossings as corners of the inner square.

Changelog:
  20230907: remove deprecated np.int, np.float, np.bool
  20210112: first stable version
"""
__version__ = 20230907

import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.filters import sato, frangi, meijering
from skimage import exposure
import scipy.ndimage as scind
import matplotlib.pyplot as plt
    
import json
try:
    # wad2.0 runs each module stand alone
    import n13_cropping as crop
    import unif_lib as unif
except ImportError:
    from . import n13_cropping as crop
    from . import unif_lib as unif


def analyse_lines(lines, image, edges, data, grow):
    """
    Calculate max continous overlap between lines and underlying data.
    If the MTF insert shows up whitish in the data, this can be used to remove
    lines found next to Cu wedge, but keep the lines through MTF.
    Works: use 5mm as allowed gap size (remove by dilations)
    """
    grow = grow//2+1 
    bla = scind.binary_dilation(data, iterations=1)#grow//2+1)
    len1 = []
    for (y0,x0), (y1,x1) in lines:
        num = int(np.sqrt((y1-y0)**2.+(x1-x0)**2.)+.5)
        x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
        # Extract the values along the line, using cubic interpolation
        zi = scind.map_coordinates(bla, np.vstack((y,x)), order=0)
        zi = scind.binary_dilation(zi, iterations=grow)
        len1.append(crop._max_consecutive_ones(zi))
        
    return len1
    
def implot(image, title=None):
    """
    """
    plt.figure()
    if not title is None:
        plt.title(title)
    plt.imshow(image)
    plt.show()
    
def lineplot(yvals, xvals=None, title=None):
    """
    """
    plt.figure()
    if not title is None:
        plt.title(title)
    if not xvals is None:
        plt.plot(xvals,yvals)
    else:
        plt.plot(yvals)
    plt.show()

def hough_options_complete(cs, options={}):
    """
    start from default hough_options. override with provided options. return result
    """
    _options = {
        'sigma_norm_px':1., #cs.phantommm2pix(.5) # want small to make small edges,
        'avg_rad_px': 50, # rad in px of area to avg for phantom value
        'frac_lo': 0.05, # ignore norm values below frac_lo*avg
        'frac_hi': 0.0, # ignore norm values above dip+frac_hi*(avg-dip)
        'pre_blur': 0.66, # reduce noise by gaussian blurring with this sigma (px)
        'def_edges': "meijering", # method to determine edges in norm,
        'edge_mfrac': .5, # 0.5 # keep only >0.5
        'dev_deg': 10., # devation from 0 for hough angle in degrees,
        'num_ang_deg': 200, # number of angles per degree,
        'min_hough_dist_px': int(cs.phantommm2pix(5.)), # min hough distance between 2 lines
        'min_hough_angle_deg': int(200/(2*10.)/2.), # int(num_ang/(2*deg_dev)/2.), # min hough angle distance between 2 lines,
        'min_len_mm': 90., # minimal number of points on hough line,
        'max_candidates': 150, # max number of found line
        'verbose': True, # show lots of info
    }
    for key,val in options.items():
        _options[key] = val
    options = _options

    return options

def hough_preprocessing(cs, options):
    """
    pre-process data before attempting hough line detections. options must be complete.
    """
    image = cs.pixeldataIn

    dimy,dimx = image.shape
    # 0. binarize
    sigma_norm = options['sigma_norm_px']
    avg_rad    = options['avg_rad_px']
    frac_lo    = options['frac_lo']
    frac_hi    = options['frac_hi']
    def_edges  = options['def_edges']

    # generate edges
    # for frangi, experiment with add half sigma; better just use 0.25mm sigma
    # seems logical to go for min value, but max value gives better results

    # 1. remove extreme pixels
    [by0, bx0, by1, bx1] = crop.remove_borders(image, cs.get_max_pixel_value())
    cs.hough_crop = [by0, bx0, by1, bx1]
    image = image[by0:by1+1,bx0:bx1+1]

    if options['verbose']:
        im_avg = image[int(dimy/2)-avg_rad:int(dimy/2)+avg_rad, int(dimx/2)-avg_rad:int(dimx/2)+avg_rad].mean()
        im_sd = image[int(dimy/2)-avg_rad:int(dimy/2)+avg_rad, int(dimx/2)-avg_rad:int(dimx/2)+avg_rad].std()
        im_max = image.max()
        im_min = image.min()
        print("NOISEA", im_avg, im_sd, im_max, im_min, options['pre_blur'], frac_hi)

    # 2. a bit of noise reduction
    image = scind.gaussian_filter(image.astype(float), sigma=options['pre_blur'])
    if options['verbose']:
        im_avg = image[int(dimy/2)-avg_rad:int(dimy/2)+avg_rad, int(dimx/2)-avg_rad:int(dimx/2)+avg_rad].mean()
        im_sd = image[int(dimy/2)-avg_rad:int(dimy/2)+avg_rad, int(dimx/2)-avg_rad:int(dimx/2)+avg_rad].std()
        im_max = image.max()
        im_min = image.min()
        print("NOISEB", im_avg, im_sd, im_max, im_min, options['pre_blur'], frac_hi)

    # rescale to get data in about the same range
    imin = image.min()
    imax = image.max()
    mmax = 1e5
    image = mmax*(image-imin)/(imax-imin)

    # 3. multi-scale local SNR
    sd_offset = 1.
    image = np.max([
        unif.LocalSNR(image, sigma_norm, bksigma = None, uiobject=None, sd_offset=sd_offset),
        unif.LocalSNR(image, 2*sigma_norm, bksigma = None, uiobject=None, sd_offset=sd_offset),
        unif.LocalSNR(image, 4*sigma_norm, bksigma = None, uiobject=None, sd_offset=sd_offset),
        ], axis=0)


    # 4. select range of SNR values to accept
    dimy,dimx = image.shape
    avg = image[int(dimy/2)-avg_rad:int(dimy/2)+avg_rad, int(dimx/2)-avg_rad:int(dimx/2)+avg_rad].mean()

    threshold_lo = avg*frac_lo
    cs.hough_avg_snr = avg


    mdat = image[(image>avg/10.)*(image<avg*2.)] 
    hist, bins_center = exposure.histogram(mdat, nbins=256)
    mdat = scind.gaussian_filter(1.*hist, 5., order=1) # derivative
    hist = scind.gaussian_filter(1.*hist, 5., order=0)
        
    # find max data point <= avg
    i_max = len(mdat)-1
    error = False
    while hist[i_max-1]>hist[i_max] or bins_center[i_max]>avg:
        if i_max>0:
            i_max -= 1
        else:
            error = True
            break
    if i_max>0:
        # find zero-crossing for valley between avg and grindlines
        i_zero = i_max
        while mdat[i_zero-1]>=0:
            if i_zero>0:
                i_zero -= 1
            else:
                #error = True
                break
    
        if i_zero == 0:
            # didn't find a dip; find the least decreasing point
            i_zero = i_max-1
            while mdat[i_zero-1]> mdat[i_zero]:
                if i_zero>0:
                    i_zero -= 1
                else:
                    break
            if i_zero > 0:
                while mdat[i_zero-1]< mdat[i_zero]:
                    if i_zero>0:
                        i_zero -= 1
                    else:
                        error = True
                        break
        
    if options['verbose']:
        plt.figure()
        plt.title("1st mmax={}".format(mmax))
        plt.plot(bins_center, mdat)
        plt.plot([threshold_lo, avg,  0.66*avg],[0,0,0], 'o')
        plt.plot([bins_center[i_zero], bins_center[i_max], bins_center[i_zero]+frac_hi*(avg-bins_center[i_zero])],[mdat[i_zero],mdat[i_max],mdat[i_zero]], 'x')
        plt.grid()
    
        plt.figure()
        plt.title("zero:{}, {}".format(bins_center[i_zero], bins_center[i_zero]/avg))
        plt.plot(bins_center, hist)
        plt.plot([threshold_lo, avg,  0.66*avg],[0,0,0], 'o')
        plt.plot([bins_center[i_zero], bins_center[i_max], bins_center[i_zero]+frac_hi*(avg-bins_center[i_zero])],[hist[i_zero],hist[i_max],hist[i_zero]], 'x')
        plt.grid()
        plt.show()

    if error:
        print("ERROR. Cannot find proper cut-off point for edge-response. Maybe increase pre_blur? For now we will use 0.66*avg value.")
        threshold_hi = 0.66*avg
    else:
        threshold_hi = bins_center[i_zero]+frac_hi*(avg-bins_center[i_zero])

    cs.hough_rthreshold_hi = threshold_hi/avg
    cs.hough_preblur = options['pre_blur']

    ## LOG
    if options['hough_log']:
        with open(options.get("hough_log_fname", "hough_log.log"), "r") as fio:
            hough_log = json.load(fio)
        if options['verbose']:
            hough_log['im_avg'] = float(im_avg)
            hough_log['im_std'] = float(im_sd)
        hough_log['snr_avg'] = float(avg)
        hough_log['frac_hi'] = float(frac_hi)
        hough_log['pre_blur'] = float(options['pre_blur'])
        hough_log['threshold_hi'] = float(threshold_hi)
        hough_log['rthreshold_hi'] = float(threshold_hi/avg)
        try:
            hough_log['kV'] = float(cs.dcmInfile.KVP)
        except:
            hough_log['kV'] = None
        with open(options.get("hough_log_fname", "hough_log.log"), "w") as fio:
            fio.write(json.dumps(hough_log, sort_keys=True, indent=4))



    # 5. binarize image of selected SNR
    data = np.ones_like(image, dtype=bool)
    data[image > threshold_hi] = 0
    data[image < threshold_lo] = 0

    # 6. remove black borders of binarized image
    threshold_len = int(cs.phantommm2pix(100.))   
    bounds = crop.crop_border(data, threshold_len)
    if not bounds is None:
        [by0,bx0, by1,bx1] = bounds
        data  =  data[by0:by1+1,bx0:bx1+1]
        image = image[by0:by1+1,bx0:bx1+1]
        if cs.hough_crop is None:
            cs.hough_crop = [by0, bx0, by1, bx1]
        else:
            _by0,_bx0 =  cs.hough_crop[0:2] 
            cs.hough_crop = [_by0+by0, _bx0+bx0, _by0+by1, _bx0+bx1]
        
    # 7. detect vertical edges
    if def_edges == "sato":
        edges = sato(data, black_ridges=False, mode='constant', sigmas=[sigma_norm, 2*sigma_norm, 4*sigma_norm])
        ddy = int(.1*dimy)
        ddx = int(.1*dimx)
        mv = np.max(edges[ddy:-ddy,ddx:-ddx])
        #mv = np.max(edges)
        edges = np.where(edges<options['edge_mfrac']*mv, 0, 1)
    elif def_edges == "frangi":
        edges = frangi(data, black_ridges=False, mode='constant', sigmas=[sigma_norm, 2*sigma_norm, 4*sigma_norm])
        ddy = int(.1*dimy)
        ddx = int(.1*dimx)
        mv = np.max(edges[ddy:-ddy,ddx:-ddx])
        if options['verbose']: print("max_edges", mv)
        edges = np.where(edges<options['edge_mfrac']*mv, 0, 1)
    elif def_edges == "meijering":
        edges = meijering(data, black_ridges=False, mode='constant', sigmas=[sigma_norm, 2*sigma_norm, 4*sigma_norm])
        ddy = int(.1*dimy)
        ddx = int(.1*dimx)
        mv = np.max(edges[ddy:-ddy,ddx:-ddx])
        if options['verbose']: print("max_edges", mv)
        edges = np.where(edges<options['edge_mfrac']*mv, 0, 1)
    else:
        print("[hough_preprocessing] Unknown edge model {}. Using plain data.".format(def_edges))
        edges = data
        
    
    return edges, data, image, avg

def hough_detection(cs, norm_image, edges_image, data_image, options):
    """
    use hough transform to find all grid lines
    """

    dimy,dimx = edges_image.shape
    # 2. do hough transform 
    # accept deviations of
    deg_dev = options['dev_deg']
    num_ang = options['num_ang_deg']*int(deg_dev+.5)+1
    tested_angles = np.deg2rad(np.linspace(-deg_dev, deg_dev, num_ang))
        
    candidates = []
    # must have at least as many to keep
    threshold_len = int(cs.phantommm2pix(options['min_len_mm']))

    # normal hough
    h, theta, d = hough_line(edges_image, theta=tested_angles)
    #h_min = np.max(h)*.5 # default = 0.5*max(h)
    #h_min = np.max(h[50:-50,50:-50])*.5 # default = 0.5*max(h)
    h_min = None # default None = 0.5*max(h)

    for ac, an, di in zip(*hough_line_peaks(h, theta, d, threshold=h_min, 
                                            min_distance=options['min_hough_dist_px'], 
                                            min_angle=options['min_hough_angle_deg'])):
        if ac > threshold_len: 
            candidates.append(
                [ac, an, di]
            )

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True) # sort on strength
    if options['verbose']:
        print("Candidates", len(candidates))
        print("acc, angle, dist, min_acc")
        for c in candidates:
            print("{}, {}".format(", ".join([str(cc) for cc in c]), threshold_len))
    
    # restrict number of lines found
    if len(candidates)>options['max_candidates']:
        candidates = candidates[:options['max_candidates']]

    # extract lines:
    origin = np.array((0, edges_image.shape[1]))
    ymin,ymax = 0,edges_image.shape[0]-1
    xmin,xmax = 0,edges_image.shape[1]-1
    lines_yx = []
    keep_acc = True # use acc to define best line of groups
    for acc, angle, dist in candidates:
        if np.isclose(np.sin(angle),0.):
            # straight line
            xmin,xmax = dist,dist
        else:
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            dy = y1-y0
            dx = origin[1]-origin[0]
            xmin,xmax = [origin[0]+dx/dy*(y-y0) for y in (ymin, ymax)]
        #print(origin, (y0, y1))
        lines_yx.append(
            [acc, [ymin,xmin], [ymax,xmax]] if keep_acc else [[ymin,xmin], [ymax,xmax]]
        )

    # return lines on left and on right
    xc = int(dimx/2)
    if len(lines_yx)<1:
        left_yx, right_yx = [],[]
    else:
        left_yx, right_yx = _split_yx(lines_yx, xc)

    left_yx =  _choose_vertical_lines("left", left_yx, cs, threshold_len, norm_image, edges_image, data_image)
    right_yx = _choose_vertical_lines("right", right_yx, cs, threshold_len, norm_image, edges_image, data_image)

    if options['verbose']:
        print("mid", xc)
        for l in left_yx:
            print("left", l)
        for l in right_yx:
            print("right", l)


    if options['verbose']:
        if not h is None:
            plt.figure()
            plt.title("hough")
            plt.imshow(#h.T, 
                       np.log(1 + h).T,
                       extent=[d[-1], d[0], np.rad2deg(theta[-1]), np.rad2deg(theta[0])],
                       cmap="gray", aspect=100.)
            angs = [np.rad2deg(angle) for  _, angle, dist in candidates]
            dists = [-dist for  _, angle, dist in candidates]
            plt.plot(dists,angs, 'r.')
            plt.title('Hough transform')
            plt.ylabel('Angles (degrees)')    
            plt.xlabel('Distance (pixels)')
        
    
    if len(left_yx)>0 and len(right_yx)>0:
        print("Hough distance", cs.pix2phantommm(_dist_point_line_pts(left_yx[-1][0], *right_yx[0])))
    
    return left_yx, right_yx

def _debug_show_detections(lines, norm, data, edges, avg):
    """
    show the detected lines on top of the images
    """
    if not norm is None:
        plt.figure()
        plt.title("0. norm")
        plt.imshow(norm, cmap="gray", vmax=avg, origin="lower")
        for lines_yx in lines:
            for (y1,x1),(y2,x2) in lines_yx:
                plt.plot([x1,x2], [y1,y2], '-r')

        plt.xlim((0, edges.shape[1]))
        plt.ylim((0, edges.shape[0]))

    if not data is None:
        plt.figure()
        plt.title("1. data")
        plt.imshow(data.astype(np.uint8), origin="lower")

    if not edges is None:
        plt.figure()
        plt.title("2. edges")
        plt.imshow(edges, origin="lower")
        for lines_yx in lines:
            for (y1,x1),(y2,x2) in lines_yx:
                plt.plot([x1,x2], [y1,y2], '-r')
        plt.xlim((0, edges.shape[1]))
        plt.ylim((0, edges.shape[0]))

    
def _dist_line_pts_line_pts(p_yx0, p_yx1, l_yx0, l_yx1):
    """
    return max distance between two lines indicated by two points each
    """
    return max(_dist_point_line_pts(p_yx0, l_yx0, l_yx1), _dist_point_line_pts(p_yx1, l_yx0, l_yx1))

def _dist_point_line_pts( p_yx, l_yx0, l_yx1):
    """
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    If the line passes through two points P1=(x1,y1) and P2=(x2,y2) then the distance of (x0,y0) from the line is:
    """
    y0,x0 = p_yx
    y1,x1 = l_yx0
    y2,x2 = l_yx1
    
    d = np.abs( (y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1 )/np.sqrt( (y2-y1)**2.+(x2-x1)**2. )
    return d

def _split_yx( lines_yx, xc):
    """
    helper to split found lines in left/right
    """
    left_yx = []
    right_yx = []
    if len (lines_yx[0]) ==2:
        for (y0,x0),(y1,x1) in lines_yx:
            if max(x0,x1)< xc:
                left_yx.append( [[y0,x0],[y1,x1]] )
            else:
                right_yx.append( [[y0,x0],[y1,x1]] )
                
        left_yx  = sorted( left_yx, key=lambda x: x[0][1])
        right_yx = sorted(right_yx, key=lambda x: x[0][1])
    else:
        for acc, (y0,x0),(y1,x1) in lines_yx:
            if max(x0,x1)< xc:
                left_yx.append( [acc, [y0,x0],[y1,x1] ])
            else:
                right_yx.append( [acc, [y0,x0],[y1,x1] ])
                
        left_yx  = sorted( left_yx, key=lambda x: x[1][1])
        right_yx = sorted(right_yx, key=lambda x: x[1][1])

    return left_yx, right_yx

def _group_yx( left_yx, dd):
    """
    average (median) neighboring lines if the distance to the previous line is smaller than dd (x-wise)
    """
    _left = []
    ll = [  left_yx[0] ]
    for (y0,x0),(y1,x1) in left_yx[1:]:
        if np.abs(x0-ll[0][0][1])<dd and np.abs(x1-ll[0][1][1])<dd:
            ll.append([  (y0,x0),(y1,x1) ])
        else:
            #for (y00,x00),(y10,x10) in ll:
            #    print("dx", len(_left), x00, x10, x10-x00)
            _y0 = np.median([y00 for (y00,x00),(y10,x10) in ll])
            _y1 = np.median([y10 for (y00,x00),(y10,x10) in ll])
            _x0 = np.median([x00 for (y00,x00),(y10,x10) in ll])
            _x1 = np.median([x10 for (y00,x00),(y10,x10) in ll])
            _left.append([  (_y0,_x0),(_y1,_x1) ])
            ll = [  [(y0,x0),(y1,x1)] ]

    # add last one
    #for (y00,x00),(y10,x10) in ll:
    #    print("dx", len(_left), x00, x10, x10-x00)
    _y0 = np.median([y00 for (y00,x00),(y10,x10) in ll])
    _y1 = np.median([y10 for (y00,x00),(y10,x10) in ll])
    _x0 = np.median([x00 for (y00,x00),(y10,x10) in ll])
    _x1 = np.median([x10 for (y00,x00),(y10,x10) in ll])
    _left.append([  (_y0,_x0),(_y1,_x1) ])

    return _left
    
def _best_yx( left_yx, dd):
    """
    per group on lines that are less than dd from eachother (x-wise), take the strongest line
    """
    _left = []
    ll = [  left_yx[0] ]
    for a, (y0,x0),(y1,x1) in left_yx[1:]:
        if np.abs(x0-ll[0][1][1])<dd and np.abs(x1-ll[0][2][1])<dd:
            ll.append([a, (y0,x0),(y1,x1)])
        else:
            #for (y00,x00),(y10,x10) in ll:
            #    print("dx", len(_left), x00, x10, x10-x00)
            ll = sorted(ll, key=lambda x:x[0], reverse=True)
            _left.append(ll[0])
            ll = [[ a, (y0,x0),(y1,x1)] ]

    # add last one
    #for (y00,x00),(y10,x10) in ll:
    #    print("dx", len(_left), x00, x10, x10-x00)
    ll = sorted(ll, key=lambda x:x[0], reverse=True)
    _left.append(ll[0])

    return _left
    
def _choose_vertical_lines(side, lines_yx, cs, threshold_len, image, edges, data):
    """
    group all detected lines from one side of the image and return the best ones
    """
    if side == "right":
        # analysis for right side is same as for left side, if order of lines is reversed
        lines_yx = list(reversed(lines_yx))

    # average neighboring lines
    dd = cs.phantommm2pix(5.)
    if len(lines_yx)>0:
        if len(lines_yx[0])==2:
            # if there are only 2 lines, take the average one
            lines_yx = _group_yx(lines_yx, dd)
        else:
            # else take the best one
            lines_yx = _best_yx(lines_yx, dd)

    # strip acc (strength indicator) if present in line descriptions
    if len(lines_yx)>0:
        if len(lines_yx[0])==3:
            lines_yx = [[b,c] for a,b,c in lines_yx]
            
    if side == "right":
        lines_yx = list(reversed(lines_yx))
    
    return lines_yx
        

def _line_crossing( A, B):
    """
    return crossing between lines
    """
    A = np.array(A)
    B = np.array(B)
    #A = np.array([[4, 0], [4, -3]])
    #B = np.array([[6, 2], [10, 2]])
    t, s = np.linalg.solve(np.array([A[1]-A[0], B[0]-B[1]]).T, B[0]-A[0])
    return (1-t)*A[0] + t*A[1]
    

def _line_extension(pix, src_yx, dimyx):
    """
    return line from end-pix to end+pix (if pix>0) of from (start-pix) to start+pix if pix<0
    """
    dimy, dimx = dimyx
    y1,x1 = src_yx[1]
    y0,x0 = src_yx[0]

    _pix = np.abs(pix)

    _dy,_dx = (y1-y0),(x1-x0)
    f = _pix/np.sqrt(_dx*_dx+_dy*_dy)
    dy = f*_dy
    dx = f*_dx

    if pix<0:
        return [ [max(0,y0-dy), max(0,x0-dx)], [min(dimy-1,y0+dy), min(dimx-1,x0+dx)] ]
    else:
        return [ [max(0,y1-dy), max(0,x1-dx)], [min(dimy-1,y1+dy), min(dimx-1,x1+dx)] ]

def _line_copy(pix, src_yx, dir_y=None, dir_x=None):
    """
    translate a line by 18 cm
    """
    y1,x1 = src_yx[1]
    y0,x0 = src_yx[0]
    if not dir_x is None:
        # move towards x
        if x1 == x0:
            dx = dir_x*pix
            dy = 0
        else:
            _dy,_dx = (y1-y0),(x1-x0)
            f = pix/np.sqrt(_dx*_dx+_dy*_dy)
            dx = -f*_dy
            dy =  f*_dx
            if (dir_x>0 and dx<0) or (dir_x<0 and dx>0): 
                dx = -dx
                dy = -dy
    else:        
        # move towards y
        if y1 == y0:
            dy = dir_y*pix
            dx = 0
        else:
            _dy,_dx = (y1-y0),(x1-x0)
            f = pix/np.sqrt(_dx*_dx+_dy*_dy)
            dx = -f*_dy
            dy =  f*_dx
            if (dir_y>0 and dy<0) or (dir_y<0 and dy>0): 
                dx = -dx
                dy = -dy

    return [ [y0+dy, x0+dx], [y1+dy, x1+dx]]
    
def _line_best_overlap(edges_image, left_yx, right_yx, tops_yx, side, diffpx, hough_options):
    """
    Find line piece of each top_yx between crossing points with left_yx and right_yx.
    Make line piece of x cm before and after the left crossing point and same for right crossing point.
    Calculate the length of that line piece overlapping with an edge in the edge image.
    Select that line that has the largest overlap, but is situated most to the left or right.
    """
    show_extensions = False
    dimyx = edges_image.shape
    
    if show_extensions:
        edges2 = 1.*edges_image.astype(float)
        
    hits = []
    for i,top in enumerate(tops_yx):
        # the crossing points
        corner_left = _line_crossing(left_yx[-1], top)
        corner_right = _line_crossing(right_yx[0], top)
    
        pos_y = (corner_left[0]+corner_right[0])/2.
        pos_x = (corner_left[1]+corner_right[1])/2.
        # the line extensions
        ext_left  = _line_extension(-diffpx, [corner_left, corner_right], dimyx)
        ext_right = _line_extension( diffpx, [corner_left, corner_right], dimyx)
    
        # now calculate overlap with edges
        _hit = 0
        _dimy, _dimx = np.shape(edges_image)
        for (y0, x0), (y1, x1) in [ext_left, ext_right]:
            dy = y1-y0
            dx = x1-x0
            le = np.sqrt(dy*dy+dx*dx)
            for ll in range(int(le)):
                y = min(max(0, int(y0+ll/le*dy)), _dimy-1)
                x = min(max(0, int(x0+ll/le*dx)), _dimx-1)
                _hit += edges_image[y,x]
                if show_extensions and side == "top":
                    edges2[y,x] = -1

        hits.append((i, _hit, pos_y, pos_x))

    pos_ix = 2 if side in ["top", "bot"] else 3
    # sort on relevant position
    hits = sorted(hits, key=lambda x:x[pos_ix], reverse= (side in ["left", "top"]) )
    hit_id = 0
    max_hit = max([h[1] for h in hits])
    hit_pos = hits[hit_id][pos_ix]

    for i, hit in enumerate(hits):
        #if side == "left":
        #    print("HI", i, hit_id, hit[1]/max_hit, hits[hit_id][1]/hit[1])
        if hit[1]/max_hit>.9 and hits[hit_id][1]/hit[1]<.75:
            # only consider this point if its score is rather large and much larger than the present one
            hit_id = i
            hit_pos = hits[hit_id][pos_ix]

    if hough_options['verbose']:
        print("Selected ({}) line {} of {}: {}".format(side, hits[hit_id][0], len(hits), hits[hit_id]))
        for h in hits:
            print(h[0], h[1], 2*diffpx, h[1]/max_hit, h[2], h[3])

    if show_extensions and side == "top":
        plt.figure()
        plt.title(side)
        plt.imshow(edges2, origin="lower")

    return hits[hit_id][0]
    

def construct_inner_square(inspect_px, edges_image, left_yx, right_yx, top_yx, bot_yx, hough_options):
    """
    constuct inner square of N13 phantom, by picking the correct detected lines for each side of grids
    """
    idx = {}
    for sel in ["left", "right", "top", "bot"]:
        if sel == "top":
            # inspect top line
            _left_yx = left_yx
            _right_yx = right_yx
            _top_yx = top_yx
    
        elif sel == "bot":
            # inspect bot line
            _left_yx = left_yx
            _right_yx = right_yx
            _top_yx = bot_yx
    
        elif sel == "left":
            # inspect left line
            _left_yx = top_yx
            _right_yx = bot_yx
            _top_yx = left_yx
    
        elif sel == "right":
            # inspect right line
            _left_yx = top_yx
            _right_yx = bot_yx
            _top_yx = right_yx

        if hough_options['verbose']:
            print("SEL", sel)    
        ix = _line_best_overlap(edges_image, _left_yx, _right_yx, _top_yx, sel, inspect_px, hough_options)
        idx[sel] = ix
    
    return idx

def FindPhantomGridHough(cs, hough_options):
    """
    Use (laplace) edge/ridge detector and Hough to find grid lines.
    Return inner crossing points.
    """
    cs.hough_crop = None
    corners = []

    hough_options = hough_options_complete(cs, hough_options)
    # Preprocessing for Hough
    edges_image, data_image, norm_image, avg_norm = hough_preprocessing(cs, hough_options)
    left_yx, right_yx = hough_detection(cs, 
                                        norm_image=norm_image, edges_image=edges_image, data_image=data_image,
                                        options=hough_options)

    # length of inspection line beyond crossing point
    inspect_px = int(cs.phantommm2pix(30.))
    #inspect_px = int(cs.phantommm2pix(70.))

    # copy if we have only one line
    pix = cs.phantommm2pix(180.) # inner distance
    if len(left_yx) ==0 and not len(right_yx)==0:
        print("Copying right line to left")
        left_yx.append(_line_copy(pix, right_yx[0], dir_x=-1))
    elif not len(left_yx) ==0 and len(right_yx)==0:
        print("Copying left line to right")
        right_yx.append(_line_copy(pix, left_yx[-1], dir_x=+1))

    if len(left_yx)>0 and len(right_yx)>0:
        top_yx, bot_yx = hough_detection(cs, 
                                         norm_image=norm_image.T, edges_image=edges_image.T, data_image=data_image.T,
                                         options=hough_options)
        
        if len(top_yx) ==0 and not len(bot_yx)==0:
            print("Copying bottom line to top")
            top_yx.append(_line_copy(pix, bot_yx[0], dir_x=-1))
        elif not len(top_yx) ==0 and len(bot_yx)==0:
            print("Copying top line to bottom")
            bot_yx.append(_line_copy(pix, top_yx[-1], dir_x=+1))

        if len(top_yx)>0 and len(bot_yx)>0:
            top_yx = [ [(x0,y0),(x1,y1)] for (y0,x0),(y1,x1) in top_yx]
            bot_yx = [ [(x0,y0),(x1,y1)] for (y0,x0),(y1,x1) in bot_yx]
            idx = construct_inner_square(inspect_px, edges_image, left_yx, right_yx, top_yx, bot_yx, hough_options)
            if 1:
                for ii in range(9): # max 9 times
                    # use conclusions of idx, and repeat to make sure we do it right. If we start with a wrong line, we might 
                    # select a wrong line. the correct line should be a stable solution; repeat at most 9 times to avoid loop
                    repeat = False
                    if not idx['left'] == len(left_yx)-1:
                        repeat = True
                        _tmp =  left_yx[-1]
                        left_yx[-1] = left_yx[idx['left']]
                        left_yx[idx['left']] = _tmp
                    if not idx['right'] == 0:
                        repeat = True
                        _tmp =  right_yx[0]
                        right_yx[0] = right_yx[idx['right']]
                        right_yx[idx['right']] = _tmp
                    if not idx['top'] == len(top_yx)-1:
                        repeat = True
                        _tmp = top_yx[-1]
                        top_yx[-1] = top_yx[idx['top']]
                        top_yx[idx['top']] = _tmp
                    if not idx['bot'] == 0:
                        repeat = True
                        _tmp = bot_yx[0]
                        bot_yx[0] = bot_yx[idx['bot']]
                        bot_yx[idx['bot']] = _tmp
                    if not repeat:
                        break
                    idx = construct_inner_square(inspect_px, edges_image, left_yx, right_yx, top_yx, bot_yx, hough_options)
                
            print("Repeated {} times".format(ii))

            if 1:
                left_yx  = [left_yx[idx['left']]]
                right_yx = [right_yx[idx['right']]]
                top_yx   = [top_yx[idx['top']]]
                bot_yx   = [bot_yx[idx['bot']]]

            corners.append( _line_crossing(left_yx[-1], top_yx[-1]) )
            corners.append( _line_crossing(right_yx[0], top_yx[-1]) )
            corners.append( _line_crossing(right_yx[0], bot_yx[0]) )
            corners.append( _line_crossing(left_yx[-1], bot_yx[0]) )
    else:
        top_yx = []
        bot_yx = []
        

    # show lines
    if hough_options['verbose']:
        _debug_show_detections([left_yx, right_yx, top_yx, bot_yx], norm_image, data_image, edges_image, avg_norm)
        
        
    if not cs.hough_crop is None:
        corners = [ [x+cs.hough_crop[0], y+cs.hough_crop[1]] for x,y in corners ]

    if hough_options['verbose']:
        plt.figure()
        plt.title("Image")
        plt.imshow(cs.pixeldataIn, origin="lower")
        plt.plot([x for y,x in corners], [y for y,x in corners], 'o')
        plt.show()

    ## LOG
    if hough_options['hough_log']:
        with open(hough_options.get("hough_log_fname", "hough_log.log"), "r") as fio:
            hough_log = json.load(fio)
        if len(left_yx)>0 and len(right_yx)>0:
            hough_log['dist_lr0'] = cs.pix2phantommm(_dist_point_line_pts(left_yx[-1][0], *right_yx[0]))
            hough_log['dist_lr1'] = cs.pix2phantommm(_dist_point_line_pts(left_yx[-1][1], *right_yx[0]))
        else:
            hough_log['dist_lr0'] = 0
            hough_log['dist_lr1'] = 0
        if len(top_yx)>0 and len(bot_yx)>0:
            hough_log['dist_td0'] = cs.pix2phantommm(_dist_point_line_pts(top_yx[-1][0], *bot_yx[0]))
            hough_log['dist_td1'] = cs.pix2phantommm(_dist_point_line_pts(top_yx[-1][1], *bot_yx[0]))
        else:
            hough_log['dist_td0'] = 0
            hough_log['dist_td1'] = 0
        for i,(y,x) in enumerate(corners):
            hough_log['square_px_x_{}'.format(i)] = x
            hough_log['square_px_y_{}'.format(i)] = y
        hough_log['mm_px'] = cs.phantommm2pix(1.)
        with open(hough_options.get("hough_log_fname", "hough_log.log"), "w") as fio:
            fio.write(json.dumps(hough_log, sort_keys=True, indent=4))
    
    return corners

