import cv2
from field_detection import detection as detection
#from ctpn.main import demo_old as text_extraction
import re
import config 
import pytesseract
import PIL
import csv
import pandas as pd
import os


image=cv2.imread("a7.jpg")
def extract(image):
    data_list=detection.getFields(image)
    #print(data_list)
    #print("############")
    print(data_list)
    #print("************")
    #master_list = []
    #response_dict={}
    
    #text_list=text_extraction.get_ctpn(data_list)
    #print(text_list)
    
    """
    #for dict1 in data_list:
        #if dict1["class_name"] == "invoice_details":
            #text_list=text_extraction.get_ctpn([dict1])
            #response_dict["invoice_details"]=text_list
            #master_list.append(response_dict)
            #response_dict = {}
            #print(master_list)
        #elif dict1["class_name"] == "address":
            #text_list=text_extraction.get_ctpn([dict1])
            #response_dict["address"]=text_list
            #master_list.append(response_dict)
            #response_dict = {}
            #print(master_list)
        #elif dict1["class_name"] == "invoice_summary":
            #text_list=text_extraction.get_ctpn([dict1])
            #response_dict["invoice_summary"]=text_list
            #master_list.append(response_dict)
            #response_dict = {}
            #print(master_list)
    #print(master_list)
    #print("***********")
    """
    #return text_list
 
    text_name = 0
    name = None
    dob = None
    gender = None
    aadhaar_no = None
    address = None
    master_list = []
    
    for i in data_list:
        if i["class_name"] == "name":
            name=pytesseract.image_to_string(i["value"], lang="eng")
            master_list.append(text_name)
            
        elif i["class_name"] == "dob":
            dob=pytesseract.image_to_string(i["value"], lang="eng")
            master_list.append(text_name)
            
        elif i["class_name"] == "gender":
            gender=pytesseract.image_to_string(i["value"], lang="eng")
            master_list.append(text_name)
           
        elif i["class_name"] == "aadhaar_no":
            aadhaar_no=pytesseract.image_to_string(i["value"], lang="eng")
            master_list.append(text_name)
            
        elif i["class_name"] == "address":
            address=pytesseract.image_to_string(i["value"], lang="eng")
            master_list.append(text_name)
            
    master_list = [name, dob, gender, aadhaar_no, address]        
            
    print(master_list)
    return master_list
    #with open("out.csv", "w", newline="") as f:
        #writer = csv.writer(f)
        #writer.writerows(master_list)
    #my_df = pd.DataFrame(master_list)
    #my_df_trans = my_df.T
    #my_df_trans.to_csv('info3.csv', index=False, header=False)
    
##    
#final_list = []
#for image_name in os.listdir("data/"):
    #img = cv2.imread("data/"+str(image_name))
    #output = extract(img)
    #final_list.append(output)
    
#final_out = pd.DataFrame(final_list,columns=["Name", "Father/mother/husband_name", "Date of birth", "Pan number"])
#final_out.to_csv('pan_data1.csv')
##

extract(image)
