# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np


data = pd.read_csv("C:/Users/Chaitanya Narhe/Desktop/DS Project/mushrooms.csv", encoding='latin')



data=data.drop(["veil-type"], axis=1)

le = LabelEncoder()
for column in data.columns:
    data[column]=le.fit_transform(data[column])

x = data.drop('class',axis=1)
y = data['class']

P = PCA(n_components=8)
pca = P.fit_transform(x)



loded_model = pickle.load(open("C:/Users/Chaitanya Narhe/Desktop/DS Project/Mushroom Model.sav", 'rb'))

def Mushroom_tp(input_data):

    ip_n_a = np.asarray(input_data)

    ip_data = ip_n_a.reshape(1,-1)

    pca_data = P.transform(ip_data)

    m = loded_model.predict(pca_data)

    print(m)

    if m[0]==1:
        return('poisonous')
    else:
        return('edible')

def main():
    
    st.title('Mashroom prediction web app')
    
    Cap_shape=st.text_input('Shape of Cap')
    Cap_surface = st.text_input('types of cap-surface')
    Cap_color = st.text_input('types of cap-color')
    bruises = st.text_input('Type of bruises')
    Odor= st.text_input('types of odor')
    Gill_attachment=st.text_input('types of gill-attachment')
    gill_spacing=st.text_input('types of gill-spacing')
    gill_size=st.text_input('types of gill-size')
    gill_color=st.text_input('types of gill-color')
    stalk_shape = st.text_input('type of stalk shape')
    stalk_root=st.text_input('stalk-root')
    stalk_surface_above_ring=st.text_input('stalk-surface-above-ring')
    stalk_surface_below_ring=st.text_input('stalk-surface-below-ring')
    stalk_color_above_ring=st.text_input('stalk-color-above-ring')
    stalk_color_below_ring=st.text_input('stalk-color-below-ring')
    veil_color=st.text_input('veil-color')
    ring_number=st.text_input('ring-number')
    ring_type=st.text_input('ring-type')
    spore_print_color=st.text_input('spore-print-color')
    population=st.text_input('population')
    habitat=st.text_input('habitat')
    
    
    prediction = ''
    
    if st.button('Mashroom Result'):
        prediction= Mushroom_tp([Cap_shape,Cap_surface,Cap_color,bruises,Odor,Gill_attachment,gill_spacing,gill_size,gill_color,stalk_shape,stalk_root,stalk_surface_above_ring,stalk_surface_below_ring,stalk_color_above_ring,stalk_color_below_ring,veil_color,ring_number,ring_type,spore_print_color,population,habitat])
        
        st.success(prediction)
        
        
if __name__ == '__main__':
    main()
    
    
    
    
