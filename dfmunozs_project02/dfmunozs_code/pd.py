import numpy as np 
import cv2 as cv
def pad(img ,pad_width, padType):
    if padType== 'edge':
        top_left_corner= img[0][0]
        top_right_corner= img[0][-1]
        bottom_left_corner= img[-1][0]
        bottom_right_corner= img[-1][-1]

        num_og_rows, num_og_cols = img.shape
        top_edge_vals= img[0]
        bottom_edge_vals= img[num_og_rows-1]
        left_col_vals=img[:,[0]]
        right_col_vals= img[:,[num_og_cols-1]]
        left_chunk=left_col_vals
        right_chunk= right_col_vals
        top_chunk= np.reshape(top_edge_vals,(1, top_edge_vals.size))
        bottom_chunk = np.reshape(bottom_edge_vals,(1, bottom_edge_vals.size))
        if pad_width>1:
            for units in range(pad_width-1):
                left_chunk= np.hstack((left_chunk,left_col_vals))
            for units in range(pad_width-1):
                right_chunk= np.hstack((right_chunk,right_col_vals))

            for units in range(pad_width-1):
                top_chunk= np.vstack((top_chunk,top_edge_vals))
            for units in range(pad_width-1):
                bottom_chunk= np.vstack((bottom_chunk,bottom_edge_vals))

        
        img = np.hstack((left_chunk, img, right_chunk))
        
        top_right_chunk= np.zeros((pad_width,pad_width))+top_right_corner
        top_left_chunk= np.zeros((pad_width, pad_width))+top_left_corner
        bottom_right_chunk=np.zeros((pad_width,pad_width))+bottom_right_corner
        bottom_left_chunk=np.zeros((pad_width,pad_width))+bottom_left_corner
        top_pad= np.hstack((top_left_chunk,top_chunk,top_right_chunk))
        bottom_pad= np.hstack((bottom_left_chunk,bottom_chunk,bottom_right_chunk))
        
        img= np.vstack((top_pad,img,bottom_pad))
        
        return img

    elif padType== 'zero':
        
        for units in range(pad_width):
            zerosRow=np.zeros(img.shape[1])+0
            bottomPadded= np.vstack((img, zerosRow))
            topPadded= np.vstack((zerosRow, bottomPadded))
            numColsnow= topPadded.shape[0]
            zerosCol= np.array([np.zeros(numColsnow)]).T
            zerosCol= zerosCol+0
            rightPad=np.hstack((topPadded, zerosCol))
            leftPad=np.hstack((zerosCol,rightPad))
            img= leftPad
        return (img)

    elif padType=='wrap':
        top_edge_chunk= img[-pad_width:]
        bottom_edge_chunk= img[0:pad_width]
        left_col_chunk=img[:,-pad_width:]
        right_col_chunk= img[:,:pad_width]
        bottom_right_chunk= img[:pad_width,:pad_width]
        top_right_chunk= img[-pad_width:,:pad_width]
        bottom_left_chunk= img[:pad_width,-pad_width:]
        top_left_chunk=img[-pad_width:,-pad_width:]
        padded_img=np.vstack(((np.hstack(((top_left_chunk,top_edge_chunk,top_right_chunk))),(np.hstack((left_col_chunk,img,right_col_chunk))),(np.hstack((bottom_left_chunk,bottom_edge_chunk,bottom_right_chunk))))))
        return padded_img

    elif padType== 'reflect':
        bottom_edge_chunk= np.flipud(img[-pad_width-1:-1])
        top_edge_chunk= np.flipud(img[1:pad_width+1])
        newimg=np.vstack((top_edge_chunk,img,bottom_edge_chunk))
        left_chunk=np.fliplr(newimg[::,1:pad_width+1])
        right_chunk=np.fliplr(newimg[::,-pad_width-1:-1]) 
        padded_img =np.hstack((left_chunk,newimg,right_chunk))
        
        return padded_img
    