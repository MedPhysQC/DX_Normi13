# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Warning: THIS MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED! And make sure rescaling is corrected!

Note: comparison will be against lit.stTable, if not matched (eg. overwritten by config) assume wall

TODO:
Changelog:
    20210127: merged into n13_lib.py; keep only changes
    20200729: attempt to fix phantom_orientation for small detectors
    20200508: dropping support for python2; dropping support for WAD-QC 1; toimage no longer exists in scipy.misc
    20190705: Remove double RelativeXRayExposure entry
    20190611: Added use_phantomrotation to skip autodetect phantom rotation
    20180501: Detect infinite loop in CuWedge
    20180205: fix in n13_geometry to allow finding droplines at two heights; added extra param mustbeprecropped to room  
    20180124: fix in uniformity where border px was ignored if cropping detected
    20171116: fix scipy version 1.0
    20170825: added optional dicom header fields (should at some point replace the kludge of checking for modality)
    20170731: shrink xrayfield to exclude constant outside region; add param for mirroring of images; 
              if crop_frac>0.98, likely invert ratio (swapped fore and background);
              fix cropping of images where original seach area is empty (XA all 0.)
    20170629: decreased threshold for orientation for 90% to 66%
    20170623: fix for finding wrong grid line in 1 direction; bot more robust MTF
    20170619: fix for missing step in Cu Wedge due to noise; fix for artefacts if treshold close to 0; 
              fix for low crontrast out-of-image; increase box search range; fix xray edge found too early;
              fix for wrong orientation if outside phantom included
    20170518: _findDropLine now uses median which is more robust for noise
    20170324: made Geometry._FineTunePhantomBox bit more robust (prefer small shift); also do not quit if phantomGrid 
              found with too little confidence
    20170310: add override params for inversion and pixmm; geometry changed xray-edge finding logic; draw uniformity_crop on image; 
    20161220: Removed class variables; removed testing stuff
    20160816: split in separate files for each block of analysis
    20160812: another attempt at consistent box finding
    20160811: fix bad align for too low contrast; trendremoval; hessian; bugfix rotated phantom; fixes for small detector
    20160802: sync with wad2.0
    20160701: Fix gridscale for none DX; fixed TableWall none DX; laxer confidence; fix for invert; fix ints
    20160205: Distinguish between linepairs insert typ38 and RXT02
    20160202: added uniformity
    20151109: start of new module, based on QCXRay_lib of Bucky_PEHAMED_Wellhofer of 20151029
"""
__version__ = '20210127'
__author__ = 'aschilham'

import numpy as np
import copy

# sanity check: we need at least scipy 0.10.1 to avoid problems mixing PIL and Pillow
import scipy
scipy_version = [int(v) for v in scipy.__version__ .split('.')]
if scipy_version[0] == 0:
    if scipy_version[1]<10 or (scipy_version[1] == 10 and scipy_version[1]<1):
        raise RuntimeError("scipy version too old. Upgrade scipy to at least 0.10.1")

try:
    # wad2.0 runs each module stand alone
    import n13hough_stuff as Hough
    import n13_xrayfield as Field
    import n13_lib as Lib
except ImportError:
    from . import n13hough_stuff as Hough
    from . import n13_xrayfield as Field
    from . import n13_lib as Lib
    
class XRayQC(Lib.XRayQC):
    def __init__(self):
        self.qcversion = __version__
        self.hough_options = {}

    #------------------- override Lib.XRayQC
    def XRayField(self, cs, sigma):
        # Find edges of XRay exposure
        # note that this should be done on uncropped image!
        return Field.XRayField(cs, workim=None, sigma=sigma)

    def FindPhantomGridHough(self, cs, hough_options):
        return Hough.FindPhantomGridHough(cs, hough_options)

    def QC(self, cs):
        """
        Normi13 analysis all in one routine.

        Outline:
        (0. invert image or not; already in creation of qc struct!)
        1. geometry:
          1.3 PhantomCoordinates
          1.2 Phantom90Degrees
        2. Cu Wedge analysis
        1.4 XRayEdges (check full field)
        3. Low Contrast analysis
        4. MTF analysis
        """
        error = True
        msg = ''

        #print('[QCNormi13]',cs.dcmInfile.SeriesDescription)
        """
        two strategies:
        1. take small sigma_norm ~1px, find all grid lines.
        2. take big sigma_norm ~ 1mm, find only inner most (and edge) grid lines.
        ad1. seems more stable
        
        Status:
        o find min/max (left/right, up/down);
        o use to find "initial" location of inner crossings
        o zoom in and use harris corner detection in small<1cm neighborhood, take most central peak?
        o rotate and crop
        o make thumbnail
        
        opt: find angle, rotate, find again for smaller dev_angles.
        """
        #print(1./cs.phantommm2pix(1), cs.phantommm2pix(1.))
        
        # default hough options
        hough_options = {
            'max_bits': 11, # 11 # max bits depth
            'hough_log': True,
            #'hough_log': False,
            'verbose': False,
        }

        hough_sato = {
            'sigma_norm_px': cs.phantommm2pix(.25), #1. cs.phantommm2pix(.5) # want small to make small edges,
            'avg_rad_px': 50, # rad in px of area to avg for phantom value
            'frac_lo': 0.1, # 0.2 #ignore norm values below frac_lo*avg
            'frac_hi': 0.75, # 0.5 #ignore norm values above frac_hi*avg
            'def_edges': "sato", #"canny", # method to determine edges in norm,
            'edge_mfrac': 0.5, # 0.5 # keep only >0.5
            'dev_deg': 10., # devation from 0 for hough angle in degrees,
            'num_ang_deg': 200, # number of angles per degree,
            'min_hough_dist_px': int(cs.phantommm2pix(5.)), # min hough distance between 2 lines
            'min_hough_angle_deg': int(200/(2*10.)/2.), # int(num_ang/(2*deg_dev)/2.), # min hough angle distance between 2 lines,
            'min_len_mm': 45., # minimal number of points on hough line,
            'max_candidates': 150, # max number of found lines
        }
        hough_frangi = {
            # last attempt: 'sigma_norm_px': cs.phantommm2pix(.25) and no extra LocalSNR
            'sigma_norm_px': cs.phantommm2pix(.25), #cs.phantommm2pix(.25), #1. cs.phantommm2pix(.5) # want small to make small edges,
            'avg_rad_px': 50, # rad in px of area to avg for phantom value
            'frac_lo': 0.05, #sato:0.1 # 0.2 #ignore norm values below frac_lo*avg
            'frac_hi': 0.5, #sato:0.75 # 0.5 #ignore norm values above frac_hi*avg
            'def_edges': "frangi", #"sato", #"canny", # method to determine edges in norm,
            'edge_mfrac': .25, # 0.5 # keep only >0.5
            'dev_deg': 10., # devation from 0 for hough angle in degrees,
            'num_ang_deg': 200, # number of angles per degree,
            'min_hough_dist_px': int(cs.phantommm2pix(5.)), # min hough distance between 2 lines
            'min_hough_angle_deg': int(200/(2*10.)/2.), # int(num_ang/(2*deg_dev)/2.), # min hough angle distance between 2 lines,
            'min_len_mm': 45., # minimal number of points on hough line,
            'max_candidates': 150, # max number of found lines
        }
        hough_meijering = {
            # last attempt: 'sigma_norm_px': cs.phantommm2pix(.25) and no extra LocalSNR
            'sigma_norm_px': cs.phantommm2pix(.25), #cs.phantommm2pix(.25), #1. cs.phantommm2pix(.5) # want small to make small edges,
            'avg_rad_px': 50, # rad in px of area to avg for phantom value
            'frac_lo': 0.05, #sato:0.1 # 0.2 #ignore norm values below frac_lo*avg
            'frac_hi': 0.0, # ignore norm values above dip+frac_hi*(avg-dip)
            'pre_blur': 0.66, # reduce noise by gaussian blurring with this sigma (px)
            'def_edges': "meijering", #"sato", #"canny", # method to determine edges in norm,
            'edge_mfrac': .5, # 0.5 # keep only >0.5
            'dev_deg': 10., # devation from 0 for hough angle in degrees,
            'num_ang_deg': 200, # number of angles per degree,
            'min_hough_dist_px': int(cs.phantommm2pix(5.)), # min hough distance between 2 lines
            'min_hough_angle_deg': int(200/(2*10.)/2.), # int(num_ang/(2*deg_dev)/2.), # min hough angle distance between 2 lines,
            'min_len_mm': 90., # minimal number of points on hough line,
            'max_candidates': 150, # max number of found lines
        }

        # set default hough_options based on supplied hough_mode
        hough_mode = getattr(self.hough_options, 'hough_mode', "meijering")
        if hough_mode == "frangi":
            for k, v in hough_frangi.items():
                hough_options[k] = v
        elif hough_mode == "sato":
            hough_options = hough_sato
            for k, v in hough_sato.items():
                hough_options[k] = v
        elif hough_mode == "meijering":
            for k, v in hough_meijering.items():
                hough_options[k] = v
            hough_options['frac_hi'] = 0.66# Good for all but f11b_wand and t2_wand
            hough_options['frac_hi'] = 0.75# f11b_wand, t2_wand
        else:
            raise ValueError("Unknown Hough_Mode {}".format(hough_mode))
        
        # override with user supplied options
        for k, v in self.hough_options.items():
            hough_options[k] = v
 
        ## LOG
        if hough_options['hough_log']:
            import json
            hough_log = {'finished': False, 'hough_mode': hough_mode, 'frac_hi': hough_options.get('frac_hi', None)}
            with open(hough_options.get("hough_log_fname", "hough_log.log"), "w") as fio:
                fio.write(json.dumps(hough_log, sort_keys=True, indent=4))
                
        # 0. preprocessing data: first make range 0-4095
        num_bits = cs.getBitsStored()
        max_bits = hough_options.get('max_bits', 12)
        if num_bits>max_bits:
            print("Downscaling from {} to {} bit for Hough only".format(num_bits, max_bits))
            original_data = copy.deepcopy(cs.pixeldataIn)
            original_max = cs.get_max_pixel_value()
            cs.pixeldataIn = cs.pixeldataIn >> (num_bits-max_bits)
            cs.max_pixel_value = cs.get_max_pixel_value() >> (num_bits-max_bits)
        elif num_bits<max_bits:
            print("Upscaling from {} to {} bit for Hough only".format(num_bits, max_bits))
            original_data = copy.deepcopy(cs.pixeldataIn)
            original_max = cs.get_max_pixel_value()
            cs.pixeldataIn = cs.pixeldataIn.astype(np.int) << (max_bits-num_bits)
            cs.max_pixel_value = int(cs.get_max_pixel_value())<< (max_bits-num_bits)


        #####
        # start it!
        error = False
        # 1.1 geometry: find gridlines and construct inner box
        roipts = self.FindPhantomGridHough(cs, hough_options)
        if len(roipts)<4:
            error = True
            msg += 'Grid '
            raise ValueError("Not enough gridlines")
        
        roipts = [ (int(y+.5), int(x+.5)) for y,x in roipts ]


        if not error:
            # 1.2 geometry: find the phantom orientation wrt the found box 
            xmin = np.min([x for y,x in roipts])
            xmax = np.max([x for y,x in roipts])
            ymin = np.min([y for y,x in roipts])
            ymax = np.max([y for y,x in roipts])

            # need original dimensions to rotate found box
            odimy, odimx = cs.pixeldataIn.shape
            error = self.FixPhantomOrientation(cs, limits=[xmin,ymin,xmax,ymax])
            if error:
                msg += 'Orientation '

            # if the orientation has changed, else change coords of box!
            if not cs.geom.crop_ranges is None:
                [xmin_px,ymin_px, xmax_px,ymax_px] = cs.geom.crop_ranges
                croppts = [
                    [ymin_px, xmin_px],
                    [ymin_px, xmax_px],
                    [ymax_px, xmax_px],
                    [ymax_px, xmin_px]
                ]

            # rotate box and crop with same number of rotations
            rots = cs.geom.box_orientation//90
            rdimy,rdimx = odimy, odimx
            while rots>0:
                roipts = [ [x, rdimy-1-y] for y,x in roipts ]
                roipts = np.roll(roipts,1,axis=0)
                if not cs.geom.crop_ranges is None:
                    croppts = [ [x, rdimy-1-y] for y,x in croppts ]
                    croppts = np.roll(roipts,1,axis=0)

                rdimy,rdimx = rdimx, rdimy
                rots -= 1

            cs.geom.box_roi = roipts
            if not cs.geom.crop_ranges is None:
                [xmin_px,ymin_px, xmax_px,ymax_px] = cs.geom.crop_ranges
                cs.geom.crop_ranges = [
                    min([x for y,x in croppts]),
                    min([y for y,x in croppts]),
                    max([x for y,x in croppts]),
                    max([y for y,x in croppts]),
                ]


            # now calculate center shift
            cs.geom.box_radmm = [90., 90.]
            cs.geom.box_roi = roipts

            if cs.geom.crop_ranges is None:
                xmin_px = ymin_px = 0
            else:
                [xmin_px,ymin_px, xmax_px,ymax_px] = cs.geom.crop_ranges

            orig_midx = .5*(cs.pixeldataIn.shape[0]-1)
            orig_midy = .5*(cs.pixeldataIn.shape[1]-1)
            roi_midx = np.mean([x for x,y in roipts])
            roi_midy = np.mean([y for x,y in roipts])
            cs.geom.center_shiftxy = [roi_midx+xmin_px-orig_midx, roi_midy+ymin_px-orig_midy]

        
        #### restore data
        if not error:
            if num_bits>max_bits or max_bits>num_bits:
                cs.pixeldataIn = original_data
                rots = cs.geom.box_orientation//90
                cs.pixeldataIn = np.rot90(cs.pixeldataIn, -rots)
                cs.max_pixel_value = original_max
                

        # 2: find Cu wedge stuff
        if not error:
            error = self.CuWedge(cs)
            if error:
                msg += 'CuWedge '

        # 1.4: travel straight along NS and find edge of x-ray; similar for EW
        if not error:
            error = self.XRayField(cs, sigma=hough_options['sigma_norm_px'])
            if error:
                msg += 'XRayField '
            
        # 3: low contrast stuff
        if not error:
            error = self.LowContrast(cs)
            if error:
                msg += 'LowContrast '

        # 4: find resolution stuff
        if not error:
            error = self.MTF(cs)
            if error:
                msg += 'MTF '


        ## LOG
        if hough_options['hough_log']:
            import json
            with open(hough_options.get("hough_log_fname", "hough_log.log"), "r") as fio:
                hough_log = json.load(fio)
            hough_log['finished'] = True
            try:
                hough_log['AlignConfidence'] = 100.*cs.geom.box_confidence
                hough_log['xray[N]cm'] = cs.geom.xr_NSWEmm[0]/10.
                hough_log['xray[E]cm'] = cs.geom.xr_NSWEmm[3]/10.
                hough_log['xray[S]cm'] = cs.geom.xr_NSWEmm[1]/10.
                hough_log['xray[W]cm'] = cs.geom.xr_NSWEmm[2]/10.
                hough_log['CuConfidence'] = 100.*cs.cuwedge.wedge_confidence
                for mm,snr in zip(cs.cuwedge.step_mmcu, cs.cuwedge.step_snr):
                    hough_log['CuSNR_{}'.format(mm)] = float(snr)
    
                hough_log['AlignMTFPosConfidenceConfidence'] = 100.*cs.mtf.pos_confidence
                hough_log['MTFFreqConfidence'] = 100.*cs.mtf.freq_confidence
            except Exception as e:
                hough_log['exception'] = str(e)

            with open(hough_options.get("hough_log_fname", "hough_log.log"), "w") as fio:
                fio.write(json.dumps(hough_log, sort_keys=True, indent=4))

        return error,msg

    # -------------------
