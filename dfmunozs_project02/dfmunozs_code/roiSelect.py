'''
Adapted from code from Bill BEGUERADJ on Stackoverflow 
https://stackoverflow.com/questions/36381684/how-to-make-a-free-hand-shaperandom-on-an-image-in-python-using-opencv

'''
import cv2
import textbox
import numpy as np 
import tkinter as tk
from tkinter import messagebox
import sys
drawing=False
    
def Mbox():
    root1 = tk.Tk()
    messagebox.showwarning('ROI Selection Instructions', '1) Move the mouse to the area you want to start selecting in the image \n2) Left click and drag the mouse around the ROI while holding the left mouse button \n3)Let go of the mouse button to stop selecting. \n4) After you are done, hit the ESC key to go to the next polygon (if applicable) and then to view the mask. \n5) Then hit any key to save the mask and continue with blending the images')
    root1.destroy()

def roiSelect(img):
    def polygon_draw(event,former_x,former_y,flags,param):

        global current_former_x,current_former_y,drawing, mode

        if event==cv2.EVENT_LBUTTONDOWN:
            drawing=True
            current_former_x,current_former_y=former_x,former_y

        elif event==cv2.EVENT_MOUSEMOVE:
            if drawing==True:
                cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),5)
                current_former_x = former_x
                current_former_y = former_y
                polygonCoord.append((current_former_x,current_former_y))
        elif event==cv2.EVENT_LBUTTONUP:
            drawing=False
            cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),5)
            current_former_x = former_x
            current_former_y = former_y
        return former_x,former_y     
    
    im = img.copy()
    poly_list=[] #used to store the multiple polygons if there are more than one
    Mbox()
    root2 = tk.Tk()
    messagebox.showwarning('Input', 'How many polygons? Enter the number you want to draw in the textbox that will show up and then press the continue button to continue' )
    root2.destroy()
    num_poly= int(textbox.get_polys())
    print(num_poly)
    
    for i in range(num_poly):
        polygonCoord=[]#used to store the coordinates of the polygon roi for the mask
        cv2.namedWindow("Mask Selection")
        cv2.setMouseCallback('Mask Selection',polygon_draw)
    
        while(1):
            cv2.imshow('Mask Selection',im)
            k=cv2.waitKey(1)&0xFF
            if k==27:
                polygonCoord= np.array(polygonCoord)
                poly_list.append(polygonCoord)
                break
            
    mask=np.zeros((im.shape[0], im.shape[1]),dtype='float32')
    if len(polygonCoord>0):
        cv2.fillPoly(mask, poly_list , 1)
    else: 
        messagebox.showwarning('Error', "Error: No region selected. Please select a region before pressing the ESC key")
        sys.exit(2)
    mask=mask.astype(np.bool)
    out= np.zeros_like(im)
    out_image= np.zeros_like(im)
    out=out.astype(np.float32)
    out[mask]= (1,1,1)
    out_image[mask]=255
    cv2.imshow('Extracted Image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return out, out_image

