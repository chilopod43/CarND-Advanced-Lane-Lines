import os
import glob
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Line:
    """ a Line 
    
    """
    def __init__(self, ym_per_pix, xm_per_pix, parallel_thresh=0.1, radius_thresh=5000.0, width_thresh=3.0, n=10, right=False):
        #was the line detected in the last iteration?
        self.detected = False
        self.n = n
        #average x values of the fitted line over the last n iterations
        self.recent_xfitted = []
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.recent_fit = []
        self.bestfit = None  
        
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        # thresholds
        self.radius_threshold = radius_thresh
        self.width_threshold = width_thresh
        self.parallel_threshold = parallel_thresh
        # define conversions in x and y from pixels space to meters
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix
        # right image flag
        self.right = right
        
    def detect(self, binary_warped):
        if self.detected:
            self.ally, self.allx, next_fit, next_x = \
                self.search_around_poly(binary_warped, self.current_fit)
        else:
            self.ally, self.allx, next_fit, next_x = \
                self.fit_polynomial(binary_warped, margin=200)
            self.detected = True
            
        # calculate best x in n frame.
        self.recent_xfitted.append(next_x)
        self.recent_xfitted = self.recent_xfitted[-self.n:]
        self.bestx = np.mean(self.recent_xfitted, axis=0)
            
        # calculate best fit in n frame.
        self.recent_fit.append(next_fit)
        self.recent_fit = self.recent_fit[-self.n:]
        self.bestfit = np.mean(self.recent_fit, axis=0)
        
        self.diffs = np.abs(next_fit - self.current_fit)        
        self.current_fit = next_fit
        self.line_base_pos = next_x
        self.radius_of_curvature = self.calc_curvature(binary_warped.shape[0]-1)
            
    def check(self, line):
        radius_diff = abs(self.radius_of_curvature - line.radius_of_curvature)
        basepos_diff = abs(self.line_base_pos - line.line_base_pos) * self.xm_per_pix
        parallel_diff = abs((self.current_fit - line.current_fit)[1])

        if (radius_diff > self.radius_threshold) or \
               (basepos_diff < self.width_threshold) or \
               (parallel_diff > self.parallel_threshold):
            self.detected = False
            
            # delete current xs,fit
            self.recent_xfitted = self.recent_xfitted[:-1]
            self.recent_fit = self.recent_fit[:-1]
            
            self.line_base_pos = self.bestx
            self.current_fit = self.bestfit
            return False
        else:
            return True
            
    def fit_polynomial(self, binary_warped, nwindows=9, margin=200, minpix=50):
        """
        nwindows: the number of sliding windows 
        margin: the width of the windows +/- margin
        minpix: minimum number of pixels found to recenter window
        """
        # take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        # find the peak of the left and right halves of the histogram.
        # these will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        if not self.right:
            x_base = np.argmax(histogram[:midpoint])
        else:
            x_base = np.argmax(histogram[midpoint:]) + midpoint

        # set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)

        # identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # current positions to be updated later for each window in nwindows
        x_current = x_base

        # create empty lists to receive left and right lane pixel indices
        lane_inds = []
        for window in range(nwindows):
            # identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height

            # define the four below boundaries of the window
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            # identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

            # append these indices to the lists
            lane_inds.append(good_inds)

            # if you found > minpix pixels, recenter next window.
            # (`right` or `leftx_current`) on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # avoids an error if the above is not implemented fully
            pass

        # extract left and right line pixel positions
        xs = nonzerox[lane_inds]
        ys = nonzeroy[lane_inds] 

        fit = np.polyfit(ys*self.ym_per_pix, xs*self.xm_per_pix, 2)
        return ys, xs, fit, x_current

    def search_around_poly(self, binary_warped, fit, margin=100):
        """
        margin: the width of the margin around the previous polynomial to search
        """
        # grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # set the area of search based on activated x-values
        lane_points = fit[0]*((nonzeroy*self.ym_per_pix)**2) + fit[1]*(nonzeroy*self.ym_per_pix) + fit[2]
        lane_inds = (
            (nonzerox*self.xm_per_pix > (lane_points - margin*self.xm_per_pix)) & 
            (nonzerox*self.xm_per_pix < (lane_points + margin*self.xm_per_pix)) )

        # extract left and right line pixel positions
        xs = nonzerox[lane_inds]
        ys = nonzeroy[lane_inds] 

        # fit new polynomials
        fit = np.polyfit(ys*self.ym_per_pix, xs*self.xm_per_pix, 2)
        return ys, xs, fit, self.line_base_pos

    def calc_curvature(self, y_eval):
        curverad = (1+(2*self.current_fit[0]*y_eval*self.ym_per_pix+self.current_fit[1])**2)**1.5/np.abs(2*self.current_fit[0])
        return curverad

    
class Camera:
    def __init__(self, img_h, img_w, src_corners, side_margin=250):
        self.img_h = img_h
        self.img_w = img_w
        self.side_margin = side_margin
        
        self.K = None
        self.D = None
        
        dst_corners = np.float32([
              [img_w-side_margin, 0],
              [img_w-side_margin, img_h-1],
              [side_margin, img_h-1],
              [side_margin, 0]])
        self.M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    
    def calibrate(self, image_glob, grid_x=9, grid_y=6):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,9,0)
        objp = np.zeros((grid_x*grid_y,3), np.float32)
        objp[:,:2] = np.mgrid[0:grid_y,0:grid_x].T.reshape(-1,2)

        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # search for chessboard corners
        for fname in glob.glob(image_glob):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (grid_y, grid_x), None)

            # if found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, k_mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        self.K = k_mtx
        self.D = dist
    
    def undistort(self, img):
        if self.K.any() and self.D.any():
            return cv2.undistort(img, self.K, self.D, None, self.K)
        else:
            raise RuntimeError("Need to calibrate camera.")
            
    def perspective(self, img):
        if self.M.any():
            return cv2.warpPerspective(img, self.M, (self.img_w, self.img_h) )
        else:
            raise RuntimeError("M matrix is invalid.")

    def binarize(self, img, s_thresh=(170, 255), sx_thresh=(30, 100)):
        # convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        # sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)

        # absolute x derivative to accentuate lines away from horizontal
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # threshold x gradient and color channel
        binary_image = np.zeros_like(scaled_sobel)
        binary_image[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]) | 
                 (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        return binary_image
    
def lane_boundaries_image(image, camera, l_line, r_line):
    # create an image to draw the lanes
    warp_zero = np.zeros(image.shape[:2]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # recast the x and y points into usable format for cv2.fillPoly().
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    
    # generate x and y values for plotting
    lfit = l_line.current_fit
    left_fitx = (lfit[0]*(ploty*l_line.ym_per_pix)**2 + lfit[1]*(ploty*l_line.ym_per_pix) + lfit[2]) / l_line.xm_per_pix
    
    rfit = r_line.current_fit
    right_fitx = (rfit[0]*(ploty*r_line.ym_per_pix)**2 + rfit[1]*(ploty*r_line.ym_per_pix) + rfit[2]) / r_line.xm_per_pix

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))], dtype=np.int32)
    pts = np.hstack((pts_left, pts_right))

    # draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, pts_left, False, color=(255, 0, 0), thickness=50)
    cv2.polylines(color_warp, pts_right, False, color=(0, 0, 255), thickness=50)

    # warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = np.linalg.inv(camera.M)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    
    # combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    # put text about curve radius.
    left_str = "left radius: " + str(int(l_line.radius_of_curvature)/1000) + " km"
    right_str = "right radius: " + str(int(r_line.radius_of_curvature)/1000) + " km"
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(result, left_str, (50,100), font, 2, (255,255,255), 5, cv2.LINE_AA)
    cv2.putText(result, right_str, (50,200), font, 2, (255,255,255), 5, cv2.LINE_AA)
    return result

def lane_finding_pipeline(image, camera, l_line, r_line):
    undist = camera.undistort(image)
    binary = camera.binarize(undist)
    warped = camera.perspective(binary)
    
    l_line.detect(warped)
    l_curve = l_line.calc_curvature(image.shape[0]-1)
    
    r_line.detect(warped)
    r_curve = r_line.calc_curvature(image.shape[0]-1)

    l_ok = l_line.check(r_line)
    r_ok = r_line.check(l_line)
    #if not(l_ok) or not(r_ok):
    #    print("recalculate line.")
    
    result = lane_boundaries_image(image, camera, l_line, r_line)
    return result

if __name__=="__main__":
    import sys
    image = mpimg.imread(sys.argv[1])

    # Camera Object
    img_h, img_w = image.shape[:2]
    side_margin=250
    src_corners = np.float32([
                  [710, 467],
                  [1108,719],
                  [207, 719],
                  [570, 467]])

    camera = Camera(img_h, img_w, src_corners, side_margin=side_margin)
    camera.calibrate('./camera_cal/calibration*.jpg')

    # Line Object
    ym_per_pix = 18.0/img_h
    xm_per_pix = 3.7/(img_w-2*side_margin)

    l_line = Line(ym_per_pix, xm_per_pix)
    r_line = Line(ym_per_pix, xm_per_pix, right=True)
    
    # execute pipeline
    result = lane_finding_pipeline(image, camera, l_line, r_line)
    
    # write image.
    mpimg.imsave(sys.argv[2], result)