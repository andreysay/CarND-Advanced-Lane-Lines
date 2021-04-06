import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import logging

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
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

class LaneLines():
    def __init__(self):
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        self.nwindows = 9
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 50
        # with U.S. regulations that require a minimum lane width of 12 feet or 3.7 meters,
        # and the dashed lane lines are 10 feet or 3 meters long each.
        # define road width as 700px
        self.roadWidth = 700
        self.ploty = None

        self.leftLine = Line()
        self.rightLine = Line()

    def find_lane_pixels(self, binary_warped):
        assert(len(binary_warped.shape) == 2)
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
        #print(f'Base position for left window: {leftx_current}, for right window: {rightx_base}')

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            # Find the four below boundaries of the window
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin  
            win_xright_low = rightx_current - self.margin  
            win_xright_high = rightx_current + self.margin
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            ### If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)


        # Extract left and right line pixel positions
        self.leftLine.allx = nonzerox[left_lane_inds]
        self.leftLine.ally = nonzeroy[left_lane_inds] 
        self.rightLine.allx = nonzerox[right_lane_inds]
        self.rightLine.ally = nonzeroy[right_lane_inds]

        logging.debug("left allx: %d", len(self.leftLine.allx))
        logging.debug("left ally: %d", len(self.leftLine.ally))
        logging.debug("right allx: %d", len(self.rightLine.allx))
        logging.debug("right ally: %d", len(self.rightLine.ally))
        logging.debug("----------------")

        ### Fit a second order polynomial to each using `np.polyfit` ###
        if len(self.leftLine.allx) != 0 and len(self.leftLine.ally) !=0: 
            self.leftLine.current_fit = np.polyfit(self.leftLine.ally, self.leftLine.allx, 2)
            self.leftLine.detected = True
        if len(self.rightLine.allx) != 0 and len(self.rightLine.ally) !=0: 
            self.rightLine.current_fit = np.polyfit(self.rightLine.ally, self.rightLine.allx, 2)
            self.rightLine.detected = True    

    def fit_polynomial(self, binary_warped):
        # Find our lane pixels first
        
        self.find_lane_pixels(binary_warped)
        if self.leftLine.detected and self.rightLine.detected:        
            self.search_around_poly(binary_warped)


    # Maybe we don't need it
    def search_around_poly(self, binary_warped):
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        left_lane_inds = ((nonzerox > (self.leftLine.current_fit[0]*(nonzeroy**2) + self.leftLine.current_fit[1]*nonzeroy + 
                        self.leftLine.current_fit[2] - self.margin)) & (nonzerox < (self.leftLine.current_fit[0]*(nonzeroy**2) + 
                        self.leftLine.current_fit[1]*nonzeroy + self.leftLine.current_fit[2] + self.margin)))
        right_lane_inds = ((nonzerox > (self.rightLine.current_fit[0]*(nonzeroy**2) + self.rightLine.current_fit[1]*nonzeroy + 
                        self.rightLine.current_fit[2] - self.margin)) & (nonzerox < (self.rightLine.current_fit[0]*(nonzeroy**2) + 
                        self.rightLine.current_fit[1]*nonzeroy + self.rightLine.current_fit[2] + self.margin)))
        
        # Again, extract left and right line pixel positions
        self.leftLine.allx = nonzerox[left_lane_inds]
        self.leftLine.ally = nonzeroy[left_lane_inds] 
        self.rightLine.allx = nonzerox[right_lane_inds]
        self.rightLine.ally = nonzeroy[right_lane_inds]

        ### Fit a second order polynomial to each with np.polyfit() ###
        if len(self.leftLine.ally) > 1500:
            self.leftLine.best_fit = np.polyfit(self.leftLine.ally, self.leftLine.allx, 2)
        # else:
        #     self.leftLine.best_fit = []

        if len(self.rightLine.ally) > 1500:
            self.rightLine.best_fit = np.polyfit(self.rightLine.ally, self.rightLine.allx, 2)
        # else:
        #     self.rightLine.best_fit = []

        # Generate x and y values for plotting
        maxy = binary_warped.shape[0] - 1
        miny = binary_warped.shape[0] // 3    
        if len(self.leftLine.ally):
            maxy = max(maxy, np.max(self.leftLine.ally))
            miny = min(miny, np.min(self.leftLine.ally))

        if len(self.rightLine.ally):
            maxy = max(maxy, np.max(self.rightLine.ally))
            miny = min(miny, np.min(self.rightLine.ally))   

        self.ploty = np.linspace(miny, maxy, binary_warped.shape[0])       

        ### Calc both polynomials using ploty, left_fit and right_fit ###
        if self.leftLine.best_fit is not None and self.rightLine.best_fit is not None:
            self.leftLine.bestx = self.leftLine.best_fit[0]*self.ploty**2 + self.leftLine.best_fit[1]*self.ploty + self.leftLine.best_fit[2]
            self.rightLine.bestx = self.rightLine.best_fit[0]*self.ploty**2 + self.rightLine.best_fit[1]*self.ploty + self.rightLine.best_fit[2]

            self.leftLine.recent_xfitted.append(self.rightLine.bestx)
            if len(self.leftLine.recent_xfitted) > 5:
                self.leftLine.recent_xfitted.pop(0)
            
            self.rightLine.recent_xfitted.append(self.rightLine.bestx)
            if len(self.rightLine.recent_xfitted) > 5:
                self.rightLine.recent_xfitted.pop(0)
            
            self.leftLine.detected = True
            self.rightLine.detected = True
            # self.leftLine.diffs = abs(self.leftLine.best_fit - self.leftLine.current_fit)
            # l_fit_x_int = abs(self.leftLine.diffs[0]*binary_warped.shape[0]**2 + self.leftLine.diffs[1]*binary_warped.shape[0] + self.leftLine.diffs[2])
            # if (self.leftLine.diffs[0] > 0.001 or \
            #    self.leftLine.diffs[1] > 1.0 or \
            #    self.leftLine.diffs[2] > 100.):
            #    self.leftLine.detected = False
            #    self.leftLine.bestx = self.leftLine.recent_xfitted[-1]
            # else:
            #     self.leftLine.detected = True
            #     self.leftLine.recent_xfitted.append(self.rightLine.bestx)
            #     if len(self.leftLine.recent_xfitted) > 5:
            #         self.leftLine.recent_xfitted.pop(0)
            # if l_fit_x_int > 100:
            #     self.leftLine.bestx = self.leftLine.recent_xfitted[-1]
            #     self.leftLine.detected = False   
            # else:
            #     self.leftLine.recent_xfitted.append(self.leftLine.bestx)
            #     if len(self.leftLine.recent_xfitted) > 5:
            #         self.leftLine.recent_xfitted.pop(0)
            #logging.debug("left pixels diff = %d", l_fit_x_int)
            #logging.debug("leftLine.recent_xfitted len = %d", len(self.leftLine.recent_xfitted))
        else:
            if len(self.leftLine.recent_xfitted) > 0 and len(self.rightLine.recent_xfitted) > 0:
                self.leftLine.bestx = self.leftLine.recent_xfitted[0]
                self.rightLine.bestx = self.rightLine.recent_xfitted[0]
                self.leftLine.detected = True
                self.rightLine.detected = True
            else:
                self.leftLine.detected = False
                self.rightLine.detected = False
        #if :      
            
            # self.rightLine.diffs = abs(self.rightLine.best_fit - self.rightLine.current_fit)
            # r_fit_x_int = abs(self.rightLine.diffs[0]*binary_warped.shape[0]**2 + self.rightLine.diffs[1]*binary_warped.shape[0] + self.rightLine.diffs[2])
            # if (self.rightLine.diffs[0] > 0.001 or \
            #    self.rightLine.diffs[1] > 1.0 or \
            #    self.rightLine.diffs[2] > 100.):
            #    self.rightLine.detected = False
            #    self.rightLine.bestx = self.rightLine.recent_xfitted[-1]
            # else:
            #     self.rightLine.detected = True
            #     self.rightLine.recent_xfitted.append(self.rightLine.bestx)
            #     if len(self.rightLine.recent_xfitted) > 5:
            #         self.rightLine.recent_xfitted.pop(0)                 
            # if r_fit_x_int > 100:
            #     self.rightLine.bestx = self.rightLine.recent_xfitted[-1]
            #     self.rightLine.detected = False
            # else:
            #     self.rightLine.recent_xfitted.append(self.rightLine.bestx)
            #     if len(self.rightLine.recent_xfitted) > 5:
            #         self.rightLine.recent_xfitted.pop(0) 
            #logging.debug("right pixels diff = %d", r_fit_x_int)                  
            #logging.debug("leftLine.recent_xfitted len = %d", len(self.leftLine.recent_xfitted))
        #else:
            

        # if self.rightLine.best_fit is not None and self.leftLine.best_fit is not None:
        #     ploty = np.linspace(0, 20, num=10 )
        #     left_fitx = self.leftLine.best_fit[0]*ploty**2 + self.leftLine.best_fit[1]*ploty + self.leftLine.best_fit[2]
        #     right_fitx = self.rightLine.best_fit[0]*ploty**2 + self.rightLine.best_fit[1]*ploty + self.rightLine.best_fit[2]
        #     delta_lines = np.mean(right_fitx - left_fitx)
        #     if delta_lines >= 150 and delta_lines <=430: 
        #         self.rightLine.detected = True
        #         self.leftLine.detected = True
        #     else:
        #         logging.debug("delta_lines = %d", delta_lines)
        #         self.rightLine.detected = False
        #         self.leftLine.detected = False

        

    def measure_curvature_real(self, img):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        height = img.shape[0]
        width = img.shape[1]
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30./height # meters per pixel in y dimension
        xm_per_pix = 3.7/width # meters per pixel in x dimension
        
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        ploty = np.linspace(0, height-1, num=height)
        y_eval = np.max(ploty)
        
        
        left_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.leftLine.bestx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.rightLine.bestx*xm_per_pix, 2)

        
        ##### Implement the calculation of R_curve (radius of curvature) #####
        self.leftLine.radius_of_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.rightLine.radius_of_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        # Calculate x bottom position for y for left lane
        left_lane_bottom = (left_fit_cr[0] * (y_eval * ym_per_pix) ** 2 + 
                            left_fit_cr[1] * (y_eval * ym_per_pix) + left_fit_cr[2])
        # Calculate x bottom position for y for right lane
        right_lane_bottom = (right_fit_cr[0] * (y_eval * ym_per_pix) ** 2 + 
                            right_fit_cr[1] * (y_eval * ym_per_pix) + right_fit_cr[2])
        # Calculate the mid point of the lane
        lane_midpoint = float(right_lane_bottom + left_lane_bottom) / 2
        # Calculate the image center in meters from left edge of the image
        image_mid_point_in_meter = width/2 * xm_per_pix
        # Positive value indicates vehicle on the right side of lane center, else on the left.
        self.leftLine.line_base_pos = (image_mid_point_in_meter - lane_midpoint)
        self.rightLine.line_base_pos = (image_mid_point_in_meter - lane_midpoint)
    
    def drawLane(self, img, binary, invM):
        #new_img = np.copy(img)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        points_left = np.array([np.transpose(np.vstack([self.leftLine.bestx, self.ploty]))])
        points_right = np.array([np.flipud(np.transpose(np.vstack([self.rightLine.bestx, self.ploty])))])
        points = np.hstack((points_left, points_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([points]), (0, 0, 255))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, invM, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.5, 0)

        return result

    def drawData(self, img):
        #result = np.copy(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        curv_l_label = 'Left line Curvature: {:.0f} m.'.format(self.leftLine.radius_of_curvature)
        curv_r_label = 'Right line Curvature: {:.0f} m.'.format(self.rightLine.radius_of_curvature)
        deviation_label = 'Vehicle Deviation: {:.3f} m.'.format(self.rightLine.line_base_pos)

        cv2.putText(img, curv_l_label, (30, 50), font, 1, (255,255,255), 2)
        cv2.putText(img, curv_r_label, (30, 100), font, 1, (255,255,255), 2)
        cv2.putText(img, deviation_label, (30, 150), font, 1, (255,255,255), 2)

        return img
