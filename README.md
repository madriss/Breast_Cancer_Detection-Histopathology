# Breast_Cancer_Detection-Histopathology
A Flask app that classifies slices to detect breast cancer tissues vs normal tissues

Context
Invasive Ductal Carcinoma (IDC) is the most common subtype of all breast cancers. To assign an aggressiveness grade to a whole mount sample, pathologists typically focus on the regions which contain the IDC. As a result, one of the common pre-processing steps for automatic aggressiveness grading is to delineate the exact regions of IDC inside of a whole mount slide.

Dataset can be found here : https://www.kaggle.com/paultimothymooney/breast-histopathology-images

---
To build the image : docker image build -t breast_cancer_detection .  
To run the image : docker run -d -p 5000:5000 breast_cancer_detection  

A live demo for this app can be accessed here : https://breast-cancer-flask.ew.r.appspot.com/