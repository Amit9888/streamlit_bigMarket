import streamlit as st
import pandas as pd

import pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


st.write("""
# Big Markt price Prediction App
This app predicts the **Big Maket Prices**!
""")

st.write('---')

le = LabelEncoder()

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')



def user_input_features():
    Item_Weight = st.sidebar.slider('Item_Weight', 4.555000,21.350000)
    Item_Visibility = st.sidebar.slider('Item_Visibility', 0.000000,0.328391)
    Item_MRP = st.sidebar.slider('Item_MRP', 31.290000,300.888400)
    OutletEstablishmentYear = st.sidebar.slider('OutletEstablishmentYear',1985,2009, 1997)
    Outlet_Type = st.sidebar.selectbox('Outlet_Type',('Supermarket Type1','Grocery Store','Supermarket Type3','Supermarket Type2'))
    New_Item_Type = st.sidebar.selectbox('New_Item_Type',('Food' ,'Drinks','Non-Consumable'))
    Item_type = st.sidebar.selectbox('Item_Type',('Fruits and Vegetables' ,'Snack Foods','Household','Frozen Foods','Dairy','Canned','Baking Goods','Health and Hygiene','Soft Drinks','Meat','Breads','Hard Drinks','Others','Starchy Foods','Breakfast','Seafood'))
    Outlet_Location_Type = st.sidebar.selectbox('Outlet_Location_Type',('Tier 1' ,'Tier 2','Tier 3'))
    Item_Fat_Content = st.sidebar.selectbox('Item_Fat_Content',('Low Fat' ,'Regular','Non-Edible'))
    Outlet_Size = st.sidebar.selectbox('Outlet_Size',('Medium' ,'Small','High'))

    Outlet = st.sidebar.selectbox('Outlet_Identifier',(0,1,2,3,4,5,6,7,9))

    Outlet_Years=2013-OutletEstablishmentYear
       
    data = {'Item_Weight': Item_Weight,
            'Item_Visibility': Item_Visibility,
            'Item_MRP': Item_MRP,
            'New_Item_Type': New_Item_Type,
            'Item_type':Item_type,
            'Outlet_Location_Type': Outlet_Location_Type,
            'Outlet': Outlet,
            'Outlet_Years':Outlet_Years,
            'Item_Fat_Content':Item_Fat_Content,
            'Outlet_Size':Outlet_Size,
            'Outlet_Type':Outlet_Type}   
    features = pd.DataFrame(data, index=[0])

    return data

df = user_input_features()


st.text(df['Outlet_Type'])







def encoder(variable_value,classes_list):

    encoder_list = [0] * len(classes_list)
    for index,classe  in enumerate(classes_list):
            if(variable_value==classe):
                encoder_list[index]=1        
    return encoder_list



Item_Fat_Content_classes=['Low Fat','Regular','Non-Edible']
Outlet_Type_classes=['Supermarket Type1','Grocery Store','Supermarket Type3','Supermarket Type2']    
new_variable_typeClasses=["Food","Drinks","Non-Consumable"]
Outlet_Location_TypeClasses=['Tier 3','Tier 2','Tier 1']
Outlet_SizeTypes=["Medium","Small","High"]


Item_Fat_Content_Encoding=encoder(df['Item_Fat_Content'],Item_Fat_Content_classes)

EncodedValues_ItemFatContent=encoder(df['Item_Fat_Content'],Item_Fat_Content_classes)

EncodedValues_OutletType=encoder(df['Outlet_Type'],Outlet_Type_classes)

EncodedValues_OutletSize=encoder(df['Outlet_Size'],Outlet_SizeTypes)

EncodedValues_OutletLocationType=encoder(df['Outlet_Location_Type'],Outlet_Location_TypeClasses)

EncodedValues_newVariableType=encoder(df['New_Item_Type'],new_variable_typeClasses)



Low_Fat,Regular,Non_Edible = EncodedValues_ItemFatContent[0],EncodedValues_ItemFatContent[1],EncodedValues_ItemFatContent[2]

Supermarket_Type1,Grocery ,Supermarket_Type3,Supermarket_Type2 = EncodedValues_OutletType[0],EncodedValues_OutletType[1],EncodedValues_OutletType[2],EncodedValues_OutletType[3]

Food,Drinks,Non_Consumable=EncodedValues_newVariableType[0],EncodedValues_newVariableType[1],EncodedValues_newVariableType[2]

Tier_3,Tier_2,Tier_1=EncodedValues_OutletLocationType[0],EncodedValues_OutletLocationType[1],EncodedValues_OutletLocationType[2]

Medium,Small,High=EncodedValues_OutletSize[0],EncodedValues_OutletSize[1],EncodedValues_OutletSize[2]


features_df = pd.DataFrame(df, index=[0])

# st.write(Tier_3)



cat_col = ['Item_Fat_Content', 'Item_type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
for col in cat_col:
    features_df[col] = le.fit_transform(features_df[col])





num_df= ['Item_Weight','Item_Visibility','Item_type','Item_MRP','Outlet_Years','Outlet']

scaler = StandardScaler()
scaledd_values=scaler.fit_transform(features_df[num_df])
scaledd_df=pd.DataFrame(scaledd_values,columns=num_df)

Item_Weight=scaledd_df['Item_Weight']
Item_Visibility=scaledd_df['Item_Visibility']
Item_Type=scaledd_df['Item_type']
Item_MRP=scaledd_df['Item_MRP']
Outlet_Years=scaledd_df['Outlet_Years']
Outlet=scaledd_df['Outlet']


load_clf = pickle.load(open('model_extra1.pkl', 'rb'))





prediction=load_clf.predict([[Item_Weight,
                        Item_Visibility,Item_Type,Item_MRP,Outlet_Years,Outlet,Low_Fat,
                        Regular,Non_Edible,Supermarket_Type1,Grocery,Supermarket_Type3,
                        Supermarket_Type2,Food,Drinks,Non_Consumable, Tier_3,
                        Tier_2,Tier_1,Medium,Small,High]])

            


st.header('Specified Input parameters')
st.write(df)
st.write('---')






st.subheader('Prediction')

st.write(prediction)



