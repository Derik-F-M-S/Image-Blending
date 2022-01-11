# Image-Blending
ECE-558 Digital Imaging Systems course project 2: This project involved Laplacian and Gaussian Image Pyramids and using these pyramids to blend two images together using a binary mask. A GUI is used to select the region of the foreground image to blend with the background image.

###### This repo contains the code as well as the outputs for the for the Second project from the ECE-558 Digital Imaging Systems course at NC State University. 

The task for this project included:
- Creating Gaussian and Laplacian Pyramids
- Creating GUI to Select the ROI for the Mask
- Image Blending using the image pyramids and the mask
## Dependencies 
This project requires Python 3 along with the following python modules:
- Numpy 
- CV2
- scipy
- tkinter

## Structure 
The [code](./dfmunozs_project02dfmunozs_code/) for this project is contained in a sigle directory with the following files: 
- [pd.py](./dfmunozs_project02/dfmunozs_code/pd.py) contains the padding function
- [q1.py](./dfmunozs_project02/dfmunozs_code/q1.py) contains the convolution function as well as the driver functions to test the convolution. 
- [pyramid.py](./dfmunozs_project02/dfmunozs_code/pyramid.py) contains functions for the Gaussian and Laplacian Pyramids
- [roiSelect.py](./dfmunozs_project02/dfmunozs_code/roiSelect.py) contains functions for the selection of the mask region for the foreground image
- [textbox.py](./dfmunozs_project02/dfmunozs_code/textbox.py) contains helper functions for displaying the instructions for the ROI selection GUI 

## Outputs
[penguin.png](penguin.png)
Examples of the blended images as well as the background and foreground images can be seen in the PDF report as well as the mask image in the root directory of the project


## Runing the code
The code for the padding and convolution was designed to be run on a Linux system as a python file using the python3 interpreter with command line arguments used to determine what files to run the blending on. 

Example run : `python3 pyramid.py --image2=./marslandscape.jpg --image1=./penguin.jpg -o=./penguin3.png -om penguinMask3.png `

Running `python3 pyramid.py -h` should give a list of the valid command line arguments
- **-i1 or --image1:** Path to the foreground image
- **-i2 or --image2:** Path to the background image
- **-o or --output:** Path for the output image
- **-om or --outputmask:** Path for the mask output image
