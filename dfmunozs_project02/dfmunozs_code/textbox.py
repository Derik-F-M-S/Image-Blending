'''
Small module used to display a textbox to get user input and return it 
used to determine how many polygons to draw in the image to 
create a mask used for blending
'''
from tkinter import *
import sys

root3=Tk()
numlist=[]
textBox=Text(root3, height=2, width=10)
textBox.pack()

def retrieve_input():
    inputValue=textBox.get("1.0","end-1c")
    num=inputValue
    numlist.append(inputValue)
    Button(root3, text="Continue", command=root3.destroy).pack()

def get_polys():
    buttonCommit=Button(root3, height=1, width=10, text="Ok", command=lambda: retrieve_input() )
    #command=lambda: retrieve_input() >>> just means do this when i press the button
    buttonCommit.pack()
    root3.mainloop()
    if len(numlist)>0:
        num_poly=(numlist[-1])
        #print(numlist[-1])
        return num_poly
    else:
        print("Did not select a number of polygons to draw")
    sys.exit(2)
