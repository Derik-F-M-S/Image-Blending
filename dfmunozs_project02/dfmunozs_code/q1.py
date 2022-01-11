import numpy as np
import cv2 as cv
import sys 
import getopt
import pd 
import math

def conv2( f, w, pad):
    """This function perfoms convolution on a given image (f) with a given kernel (w) and a specified padding type
    Parameters:
    f: numpy.ndarray, required
        The input image passed into the function as a numpy.ndarray that is either a grayscale image (two dimensional) or a color image (three dimensional) 
    w: numpy.ndarray, required
        The kernel or filter that the image will be convolved with. This is also a numpy.ndarray and is two dimensional and can be any dimension such as 3x3 4x4 1x2 etc.
    pad: int or str, required
        The padding type for the input image. Can be either an integer or a string representing the 4 types of padding from:
        0 or 'zero'
            Pads the image with zero padding
        1 or 'wrap'
            Pads the image with wrap around padding
        2 or 'edge' 
            Pads the image with copy edge padding
        3 or 'reflect' 
            Pads the image with reflect across edge padding
    Returns:
    image as numpy.ndarray
    """
    #Checks for inputs
    if not type(f) is np.ndarray:
        raise TypeError("This convolution function is designed only to take in a numpy array as the fist argument, Please make sure that you are passing it a np.ndarray")
    if not type(f) is np.ndarray:
        raise TypeError("This convolution function is designed only to take in a numpy array as the second argument, Please make sure that you are passing it a np.ndarray")

    if ((pad == 0)or (pad == 1)or(pad == 2)or (pad == 2)or(pad == 3)or (pad  == 'wrap') or (pad  ==  'edge') or (pad  == 'reflect')):
        pass 
    else:
        print(pad, "pad type", type(pad))
        raise TypeError("The padding argument should be either be an integer between 0 and 3 or one of 3 keywords( 'wrap', 'edge','reflect') , Please make sure you are passing it the correct value")

    if (pad < 0 or pad> 3): 
        raise Exception("The padding type should be one of 4 options: 0 for zero padding, 1 for wrap-around, 2 for copy edge, or 3 for reflect across edge") 

    if ((pad == 0) or (pad== 'zero')):
        myPadArg = 'zero'
    if ((pad == 1) or (pad=='wrap')):
        myPadArg= 'wrap'
    if ((pad == 2) or (pad== 'edge')):
        myPadArg= 'edge'
    if ((pad == 3) or (pad== 'reflect')):
        myPadArg= 'reflect'
    if((len(f.shape))==3):
        gray_or_color=1

    elif((len(f.shape))==2):
        gray_or_color=0 
    else:
        raise Exception("This function is designed for only grayscale and color images, ie ndarrays with 2 or 3 dimensions only")
    # pad_dim1, pad_dim2 = w.shape
    pad_dimLR= math.ceil(((w.shape[1])-1)/2)
    pad_dimTB= math.ceil(((w.shape[0])-1)/2)
    #pad_width=
    #pad_dim1=np.repeat(pad_dimLR,2)
    #pad_dim2=np.repeat(pad_dimTB,2)
    pad_width= np.array((pad_dimLR,pad_dimLR,pad_dimTB,pad_dimTB))
    pad_width=2

    if (gray_or_color==0):
        kernel_h, kernel_w= w.shape
        
        
        padded_image= pd.pad(f,pad_width, myPadArg)
        padded_h, padded_w= padded_image.shape
        
        output_h= (padded_h- kernel_h) //1 +1 
        output_w= (padded_w- kernel_w) //1 +1  
        new_image= np.zeros((output_h,output_w)).astype(np.float32) 

        for y in range(0, output_h):
            for x in range( 0, output_w):
                new_image[y][x] = np.sum(padded_image[y * 1:y * 1+ kernel_h, x * 1:x * 1 + kernel_w] * w).astype(np.float32)
        return new_image

    if (gray_or_color==1):
        kernel_h, kernel_w= w.shape
        print("kernel shape:", w.shape)

        blue_c= f[:,:,0]
        green_c= f[:,:,1]
        red_c= f[:,:,2]
        
        p_b_c= pd.pad(blue_c,pad_width,myPadArg)
        p_g_c= pd.pad(green_c,pad_width,myPadArg)
        p_r_c= pd.pad(red_c,pad_width, myPadArg)
        padded_h, padded_w= (p_b_c.shape)

        
        output_h= (padded_h- kernel_h) //1 +1 
        output_w= (padded_w- kernel_w) //1 +1 
        new_b=np.zeros((output_h,output_w)).astype(np.float32)
        new_g=np.zeros((output_h,output_w)).astype(np.float32)
        new_r=np.zeros((output_h,output_w)).astype(np.float32)
 

        for y in range(0, output_h):
           for x in range( 0, output_w):
               new_b[y][x]=np.sum(p_b_c[y * 1:y * 1+ kernel_h, x * 1:x * 1 + kernel_w] * w).astype(np.float32)
               new_g[y][x]=np.sum(p_g_c[y * 1:y * 1+ kernel_h, x * 1:x * 1 + kernel_w] * w).astype(np.float32)
               new_r[y][x]=np.sum(p_r_c[y * 1:y * 1+ kernel_h, x * 1:x * 1 + kernel_w] * w).astype(np.float32)
	#Stacking the slices 
        new_image= np.dstack((new_b,new_g,new_r) )
        return new_image

def main(argv):#Driver program to run the convolution function takes in inputs from the terminal 
    COLOR= 0
    kernel=0
    PADARG=0
    PATHC=False
    KC=False
    PC=False
    CC=False
    OC=False
    HC=False
    try:
        opts, args= getopt.getopt(argv,"i:k:p:c:o:h")
    except getopt.GetoptError:
            print ('-i <pathToInputFile> -k <kernelType> -p <paddingType> -o <outpurfilepath> -c <color>')
            sys.exit(2)
    for opt, arg in opts:
        if opt== '-i':
            imgPATH= arg
            PATHC= True
        elif opt== '-k':
            KC=True
            if (arg == 'box'):
                kernel=(1/9)* np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.float32)
            elif (arg== 'row derivative'):
                kernel= np.array([[-1],[1]])
            elif (arg== 'col derivative'):
                kernel= np.array([[1],[-1]])
            elif (arg== 'prewitt1'):#
                kernel= np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
            elif (arg== 'prewitt2'):
                kernel= np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            elif (arg== 'sobel1'):
                kernel= np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            elif (arg== 'sobel2'):
                kernel= np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
            elif (arg== 'roberts1'):
                kernel= np.array([[0,1],[-1,0]])
            elif (arg== 'roberts2'):
                kernel= np.array([[1,0],[0,-1]])
        elif opt== '-p':
            PC=True
            PADARG= int(arg)
        elif opt== '-c':
            CC=True
            argc=int(arg)
            if (argc== 0):
                COLOR=0
            if (argc== 1):
                COLOR=1
        elif opt== '-o':
            OC=True
            outputFilePath=arg 
        elif opt== '-h':
            HC= True
            print("Example run : python3 q1.py -i 'lena.png' -k 'box' -p 0 -c 1 -o 'convlena.png' ")
            print("Possible -k arguments: 'box' , 'row derivative' , 'col derivative' , 'prewitt1' , 'prewitt2' , 'sobel1' , 'sobel2' , 'roberts1' , 'roberts2'")
            print("Possible padding arguments: 0 , 1 , 2 , 3 (zero padding, wrap, edge, reflect")
            print("Possible color arguments: 0 , 1 (grayscale , RGB)" )
            sys.exit(0)
    if ((CC != True or PC !=True or KC!=True or PATHC != True or OC!= True) and HC==False):
        print("Missing a required argument!, run -h flag to get more info on how to run this program")
        print ('-i <pathToInputFile> -k <kernelType> -p <paddingType> -o <outpurfilepath> -c <color>')
        print('\n Got Color Argument?:',CC,'\n Got Padding Argument?:',PC,'\n Got Kernel Parameter?:',KC,'\n Got Input Image Path?:',PATHC,'\n Got Output Image Path?:',OC,'\n')
        sys.exit(2)
    print("COLOR:",COLOR,"ImgPATH:",imgPATH,"Kernel: \n", kernel," \n PADARG: \n",PADARG)
    if(COLOR==0):
        img= cv.imread(imgPATH, 0) 
    else:
        img= cv.imread(imgPATH) 
    
    out_img= conv2(img, kernel,PADARG)

    cv.imwrite(outputFilePath, out_img) 

if __name__== "__main__":
    main(sys.argv[1:])
