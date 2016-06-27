# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:46:41 2015

@author: Wolf
"""

# This code is to construct a GUI for the particle counting

from Tkinter import *
#from Tkinter import _setit

import numpy as np
import cv2

import tkFileDialog, os

#import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg  
from matplotlib.figure import Figure   # Tkinter has no figure widget

from skimage.morphology import reconstruction
from skimage.filters import threshold_adaptive

import FileDialog # For the py2applet

#---------------------------------------------------------------------
class data:   # Use the 'dictionary' and 'class' to collect the data
      def __init__(self, center, radius, boundary):
          self.center = center            
          self.radius = radius
          self.boundary = boundary
          
#----------------------------------------------------------------------
def open_file():
    
    global img, filepath    
    
    # Load the file
    returned_values['filename'] = tkFileDialog.askopenfilename()
    filepath = returned_values.get('filename')
    
    # Load the image (gray scale: 0)
    img = cv2.imread(filepath)
    (r, g, b) = cv2.split(img) # cv2 use BGR from rather than RGB form
    img = cv2.merge([b, g, r])    
    
    # Take the middle part
#    w, l, s = img.shape
#    img = img[int(w/4):w-int(w/4),int(l/4):l-int(l/4),:]
        
    
    # Show the square wave 
    title = filepath
    refreshFigure(img, title) 
    
def analyze_image():  
    
    global sin_par, dipole, multiple, contours
    
    # Change to the gray image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # Adaptive Thresholding
    block_size = 200
    img_binary = threshold_adaptive(img_gray, block_size, offset=7)    
    img_binary = img_binary.astype(dtype='uint8')*255
    
    # Remove the salt and pepper noise
    img_binary = cv2.medianBlur(img_binary,7)    
    
    # Fill the hole
    img_binary = 255-img_binary # Change the white/black for filling the hole     
    seed = np.copy(img_binary)
    seed[1:-1, 1:-1] = img_binary.max()
    mask = img_binary
    filled = reconstruction(seed, mask, method='erosion')
    filled = filled.astype(dtype = 'uint8')

    # Find the contours
    # For terminal use    
#    image, contours, hierarchy = cv2.findContours(filled,
#                                 cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)         
    # For Anaconda use    
    contours, hierarchy = cv2.findContours(filled,
                                  cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)        
    
    # Calculate the area (= len) of each contour
    area = np.zeros(len(contours))
    area_contour = np.zeros(len(contours))

    for kk in range(0,len(contours)):
        area[kk] = len(contours[kk])
        area_contour[kk] = cv2.contourArea(contours[kk])
  
    single = np.where((area_contour>200)&(area_contour<=624))[0]
    dipole = np.where((area_contour>624)&(area_contour<=936))[0]
    multiple = np.where(area_contour>936)[0]    
    
    # Analyze the single particle     
    sin_par = analyze_single(single,filled)
    
    # Draw the contours
    refreshFigure_contours(img,sin_par,dipole,multiple,contours)    
 
def refreshFigure(img_temp, title = None):
     
    # Give the initial setting     
    if title == None:
       title = 'Image'
     
    # Plot the figure
    fig.clear()
    fig.imshow(img_temp)   
    fig.set_xlabel('x (pixel)')    
    fig.set_ylabel('y (pixel)') 
    fig.set_title(title)
    canvas_frame.show()
    
def refreshFigure_contours(img_temp,sin_par_temp,dipole_temp,multiple_temp,contours_temp):

    # Keep the original image
    img1 = img_temp.copy()
     
    # Plot the figure
    fig.clear()
    fig.imshow(img1)   
    
    # Draw the contours
    # draw the outer circle 
    for ii in range(0,len(sin_par_temp.radius)):
        cv2.circle(img1,(sin_par_temp.center[ii,0],sin_par_temp.center[ii,1]),
                   sin_par_temp.radius[ii],(0,0,0),2)
               
    # Tag the dipole particle
    for jj in range(0,len(dipole_temp)):
        cv2.drawContours(img1, contours_temp, dipole_temp[jj], (0,255,0), 3)

    # Tag the multiple particle
    for kk in range(0,len(multiple_temp)):
        cv2.drawContours(img1, contours_temp, multiple_temp[kk], (255,255,255), 3)
    
    fig.set_xlabel('x (pixel)')    
    fig.set_ylabel('y (pixel)') 
    fig.set_title('Analysis')
    canvas_frame.show()    
   
def export_data(sin_par_temp, filepath_temp):
    
    # Export data 
    data = np.column_stack((sin_par_temp.center,sin_par_temp.radius)) 
    
    # Give the file name
    filepath = os.path.splitext(filepath_temp)[0]
    filename = os.path.basename(filepath)    
    dirname = os.path.dirname(filepath)
    
    # Output the file
    np.savetxt(dirname +'/' + filename + '_single', data) 
    
    # Export the figure   
    eps_name = dirname + '/' + filename + '_circle.eps' 
    canvas_frame.print_figure(eps_name, format = 'eps', dpi = 300)
        
def show_image():
    
    global num_single, num_double, num_multiple, temp_boundary
    
    # Check the choise
    var_value = var.get()
    
    if (var_value == 'Single'):
        
        if (len(sin_par.radius)!=0 and num_single == 0):
            num_single = 1
        
        # Show the number of frame of condition
        condition = str(num_single) + '/' + str(len(sin_par.radius))
        var_label.set(condition)
        
        # Get the single image and related parameters        
        temp_image, temp_par = show_single_image()
        
        # Show the image        
        refreshFigure_contours(temp_image,temp_par,[],[],[])
        
    elif (var_value == 'Double'):    
        
        
        if (len(dipole)!=0 and num_double == 0):
            num_double = 1
        
        # Show the number of frame of condition
        condition = str(num_double) + '/' + str(len(dipole))
        var_label.set(condition)
        
        # Get the 'double' image and related parameters 
        rgb_color = (0,255,0)
        temp_image, temp_boundary = show_contour_image(dipole,num_double,
                                                       rgb_color)        
        refreshFigure(temp_image, title = None)
        
    elif (var_value == 'Multiple'):    
        
        if (len(multiple)!=0 and num_multiple == 0):
            num_multiple = 1
        
        # Show the number of frame of condition
        condition = str(num_multiple) + '/' + str(len(multiple))
        var_label.set(condition)     
        
        # Get the 'multiple' image and related parameters 
        rgb_color = (255,255,255)
        temp_image, temp_boundary = show_contour_image(multiple,num_multiple,
                                                       rgb_color)        
        refreshFigure(temp_image, title = None)
        
def next_image():

    global num_single, num_double, num_multiple, temp_boundary

    # Check the choise
    var_value = var.get()
        
    if (var_value == 'Single'):
        
        # Do a circularized check
        if (num_single < len(sin_par.radius)):
            num_single = num_single + 1
        else:
            num_single = 1
        
        # Show the number of frame of condition
        condition = str(num_single) + '/' + str(len(sin_par.radius))
        var_label.set(condition)
        
        # Get the single image and related parameters        
        temp_image, temp_par = show_single_image()
        
        # Show the image        
        refreshFigure_contours(temp_image,temp_par,[],[],[])
        
    elif (var_value == 'Double'):    
        
        # Do a circularized check
        if (num_double < len(dipole)):
            num_double = num_double + 1
        else:
            num_double = 1
        
        # Show the number of frame of condition
        condition = str(num_double) + '/' + str(len(dipole))
        var_label.set(condition)
        
        # Get the 'double' image and related parameters 
        rgb_color = (0,255,0)
        temp_image, temp_boundary = show_contour_image(dipole,num_double,
                                                       rgb_color)        
        refreshFigure(temp_image, title = None)
        
    elif (var_value == 'Multiple'):    
        
         # Do a circularized check
         if (num_multiple < len(multiple)):
             num_multiple = num_multiple + 1
         else:
             num_multiple = 1       
        
         # Show the number of frame of condition
         condition = str(num_multiple) + '/' + str(len(multiple))
         var_label.set(condition)
        
         # Get the 'multiple' image and related parameters 
         rgb_color = (255,255,255)
         temp_image, temp_boundary = show_contour_image(multiple,num_multiple,
                                                        rgb_color)         
         refreshFigure(temp_image, title = None)
        
def previous_image():

    global num_single, num_double, num_multiple, temp_boundary

    # Check the choise
    var_value = var.get()
        
    if (var_value == 'Single'):
        
        # Do a circularized check
        if (num_single > 1):
            num_single = num_single - 1
        else:
            num_single = len(sin_par.radius)
        
        # Show the number of frame of condition
        condition = str(num_single) + '/' + str(len(sin_par.radius))
        var_label.set(condition)
        
        # Get the single image and related parameters        
        temp_image, temp_par = show_single_image()
        
        # Show the image        
        refreshFigure_contours(temp_image,temp_par,[],[],[])
        
    elif (var_value == 'Double'):    
        
        # Do a circularized check
        if (num_double > 1):
            num_double = num_double - 1
        else:
            num_double = len(dipole)
        
        # Show the number of frame of condition
        condition = str(num_double) + '/' + str(len(dipole))
        var_label.set(condition)
        
        # Get the 'double' image and related parameters 
        rgb_color = (0,255,0)
        temp_image, temp_boundary = show_contour_image(dipole,num_double,
                                                       rgb_color)        
        refreshFigure(temp_image, title = None)
        
    elif (var_value == 'Multiple'):    
        
         # Do a circularized check
        if (num_multiple > 1):
            num_multiple = num_multiple - 1
        else:
            num_multiple = len(multiple)       
        
        # Show the number of frame of condition
        condition = str(num_multiple) + '/' + str(len(multiple))
        var_label.set(condition)
        
        # Get the 'multiple' image and related parameters 
        rgb_color = (255,255,255)
        temp_image, temp_boundary = show_contour_image(multiple,num_multiple,
                                                       rgb_color)         
        refreshFigure(temp_image, title = None)      

def add_new_single_particle(event):
    
    global sin_par
    
    # Get the center position and add the circle on the figure
    if var_add.get() == 1: # To check the value of check_button
       
       temp_center = np.array([event.xdata, event.ydata],dtype=int)
       temp_radius = np.mean(sin_par.radius,dtype=int)
       
       circle = plt.Circle(temp_center,temp_radius,color='black',fill=False)
       fig.add_artist(circle)

       # Update the data of single particle
       temp_center = temp_center + (temp_boundary[2],temp_boundary[0])

       sin_par.radius = np.append(sin_par.radius,temp_radius)
       sin_par.center = np.append(sin_par.center,[temp_center],axis=0)
       sin_par.boundary = np.append(sin_par.boundary,[temp_boundary],axis=0)
    
       # Refresh the figure on canvas
       canvas_frame.show()

def delete_single():
    
    global sin_par, num_single
    
    # Check the choise
    var_value = var.get()
        
    if (var_value == 'Single'):  
        
        # Delete the data 
        sin_par.center = np.delete(sin_par.center, num_single-1, 0)
        sin_par.radius = np.delete(sin_par.radius, num_single-1, 0)
        sin_par.boundary = np.delete(sin_par.boundary, num_single-1, 0)
        
        # Get the corrected index
        if (num_single > len(sin_par.radius)):     
            num_single = len(sin_par.radius)
        
        # Reshow the image
        # Show the number of frame of condition
        condition = str(num_single) + '/' + str(len(sin_par.radius))
        var_label.set(condition)
        
        if (num_single==0):        
            fig.clear()
            canvas_frame.show()
        else:
            # Get the single image and related parameters        
            temp_image, temp_par = show_single_image()
        
            # Show the image        
            refreshFigure_contours(temp_image,temp_par,[],[],[])

#----------------------------------------------------------------------
def analyze_single(single,filled):
    
    # Determine the bounadry for distinguishing the single particle
    sin_par = data(np.zeros([len(single), 2],dtype=int),
               np.zeros(len(single),dtype=int),
               np.zeros([len(single), 4],dtype=int))
    for kk in range(0,len(single)):
        
        # Get the region for single coutour        
        region = contours[single[kk]]        
        up, down, left, right, flag = region_check(region,filled)
        temp = filled[up:down,left:right]
    #    plt.imshow(temp,cmap = plt.cm.gray)
        
        if (flag == 1):
            continue
        
        # For anconda use
        circles = cv2.HoughCircles(temp,cv2.cv.CV_HOUGH_GRADIENT,1,10,
                                   param1=100,param2=2,
                                   minRadius=4,maxRadius=10)
        # For terminal use                            
#        circles = cv2.HoughCircles(temp,cv2.HOUGH_GRADIENT,1,10,
#                                   param1=100,param2=1,
#                                   minRadius=2,maxRadius=5)                           
                                   
        if (circles is None or
            circles[0,0,0]+circles[0,0,2] >= right-left or
            circles[0,0,0]-circles[0,0,2] <= 1 or
            circles[0,0,1]+circles[0,0,2] >= down-up or
            circles[0,0,1]-circles[0,0,2] <= 1):
            continue                        
                                                        
        sin_par.center[kk,:] = circles[0,0,0:2] + np.array([left,up])                                                           
        sin_par.radius[kk] = circles[0,0,2]
        sin_par.boundary[kk,:] = np.array([up,down,left,right])

    # Remove all zero parts
    temp = np.where(sin_par.radius==0)
    sin_par.center = np.delete(sin_par.center,temp,0)
    sin_par.radius = np.delete(sin_par.radius,temp)
    sin_par.boundary = np.delete(sin_par.boundary,temp,0)

    return sin_par

def show_single_image():

    # Check the boundary        
    up = sin_par.boundary[num_single-1][0]
    down = sin_par.boundary[num_single-1][1] 
    left = sin_par.boundary[num_single-1][2]
    right = sin_par.boundary[num_single-1][3] 
        
    temp_par = data(np.zeros([1, 2],dtype=int),
                        np.zeros(1,dtype=int),
                        None)
    temp_par.center[0,:] = sin_par.center[num_single-1,:] - np.array([left,up])
    temp_par.radius[0] = sin_par.radius[num_single-1]

    # Get the single region and show the figure        
    temp_image = img[up:down,left:right,:]     
    
    return temp_image, temp_par

def show_contour_image(temp_type,temp_num_type,rgb_color):

     region = contours[temp_type[temp_num_type-1]] 
     up, down, left, right, flag = region_check(region,img)

     # Draw the contours        
     img_temp = img.copy()
     cv2.drawContours(img_temp, contours, temp_type[temp_num_type-1], rgb_color, 1)
     
    # draw the single particle 
     for ii in range(0,len(sin_par.radius)):
         cv2.circle(img_temp,(sin_par.center[ii,0],sin_par.center[ii,1]),
                    sin_par.radius[ii],(0,0,0),1)     
     
     temp_image = img_temp[up:down,left:right,:]
     temp_boundary = np.array([up,down,left,right])
     
     return temp_image, temp_boundary

def region_check(region,filled):
    
    flag = 0 # Check the boundary is on the limit or not   
    
    temp_x = region[:,0,0]
    temp_y = region[:,0,1]
    
    length_x = (np.max(temp_x)-np.min(temp_x))/3
    length_y = (np.max(temp_y)-np.min(temp_y))/3
    
    if (np.min(temp_x)-length_x < 1):
        left = 1
    else:
        left = np.ceil(np.min(temp_x)-length_x).astype(int)
    
    if (np.max(temp_x)+length_x > filled.shape[1]):
        right = filled.shape[1]
    else:
        right = np.ceil(max(temp_x)+length_x).astype(int)
        
    if (np.min(temp_y)-length_y < 1):
        up = 1
    else:
        up = np.ceil(np.min(temp_y)-length_y).astype(int)

    if (np.max(temp_y)+length_y > filled.shape[0]):
        down = filled.shape[0]
    else:
        down = np.ceil(np.max(temp_y)+length_y).astype(int)
        
    if (np.min(temp_x) == 1 or np.max(temp_x)==filled.shape[1] 
        or np.min(temp_y) == 1 or np.max(temp_y)==filled.shape[0]):    
        flag = 1
    
    return up, down, left, right, flag

#----------------------------------------------------------------------
# Parameters for the global parts
returned_values = {}   # This is to get the path of file
choices = ('Single', 'Double', 'Multiple')
num_single = 0
num_double = 0
num_multiple = 0

# Begin the GUI interface
root = Tk()
root.title('Particle Counting')

# Decide the frames
left = Frame(root)
right = Frame(root)

left_top = Frame(left)
left_down = Frame(left)

left.grid(row = 0, column = 0)
right.grid(row = 0, column = 1)

left_top.grid(row = 0, column = 0)
left_down.grid(row = 1, column = 0)

# Plot of figure
# The text widget to show the x, y positions
info_xy = Text(left_top, height = 1)  
info_xy.tag_configure('center', justify = 'center')
info_xy.pack()

# The setting of figure
mat_plot = Figure(figsize=(6,4), dpi = 125)
fig = mat_plot.add_subplot(111)
fig.set_xlabel('x')
fig.set_ylabel('y')

# Draw the figure by tk.DrawingArea
canvas_frame = FigureCanvasTkAgg(mat_plot, master = left_top)
canvas_frame.show()
canvas_frame.get_tk_widget().pack(side = 'top', fill = 'both', expand = 1)

toolbar = NavigationToolbar2TkAgg(canvas_frame, left_top)
toolbar.update()

canvas_frame._tkcanvas.pack(side = 'top', fill = 'both', expand = 1)

# Help to grab the (x,y positions) of center for single particle
# There is the problem for the magnification here
canvas_frame.mpl_connect('button_press_event', add_new_single_particle)

# Load the file
button_load = Button(right, text = 'Load', command = open_file)
button_load.pack()

# Analyze the image to get the initial contours
button_load = Button(right, text = 'Analyze', command = analyze_image)
button_load.pack()

# Show the opened files
var = StringVar(root)
var.set(choices[0])

# all submenus to main menu
network_select = OptionMenu(right, var, *choices)
network_select.pack()

# Show the map of each contour
button_show = Button(right, text = 'Show', 
                     command = show_image)
button_show.pack()

# Show the number of contour
var_label = StringVar()
var_label.set('0/0')
con_num = Label(left_down, textvariable = var_label)
con_num.pack(side = LEFT)

# Show the previous image
button_previous = Button(left_down, text = 'Previous', 
                         command = previous_image)
button_previous.pack(side = LEFT)

# Show the next image
button_next = Button(left_down, text = 'Next', command = next_image)
button_next.pack(side = LEFT)

# Add the single particle
var_add = IntVar()   # boolean value: var_cut.get()
check_add = Checkbutton(left_down, text = 'Add (Double, Multiple)', 
                        variable = var_add)
check_add.pack(side = LEFT)

# Delete the single particle
button_delete = Button(left_down, text = 'Delete (Single)', 
                       command = delete_single)
button_delete.pack(side = LEFT)

# Show the intial image
button_raw = Button(right, text = 'Raw Image', 
                    command = lambda: refreshFigure(img,filepath))
button_raw.pack()

# Check the single pariticle
button_check = Button(right, text = 'Check Particle',
                      command = lambda: refreshFigure_contours(img,sin_par,
                                                               [],[],[]))
button_check.pack()

# Export the data
button_export = Button(right, text = 'Export',
                       command = lambda: export_data(sin_par, filepath))
button_export.pack()

# Quit the program
button_quit = Button(right, text = 'Quit', command = root.destroy)
button_quit.pack()

root.mainloop()





