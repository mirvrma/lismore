import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import math
from shapely.geometry import box
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(layout="wide") #initial_sidebar_state="collapsed", menu_items=None



distinct_colors = [
    [166, 206, 227],
    [31, 120, 180],
    [178, 223, 138],
    [51, 160, 44],
    [251, 154, 153],
    [227, 26, 28],
    [253, 191, 111],
    [255, 127, 0],
    [202, 178, 214],
    [106, 61, 154],
    [255, 255, 153],
    [177, 89, 40],
    [141, 211, 199],
    [255, 255, 179],
    [190, 186, 218],
    [251, 128, 114],
    [128, 177, 211],
    [253, 180, 98],
    [179, 222, 105],
    [252, 205, 229],
    [217, 217, 217],
    [188, 128, 189],
    [204, 235, 197],
    [255, 237, 111],
    [255, 255, 204],
]


@st.cache_data
def get_alloc():
    alloc = pd.read_csv("/Users/tonyweir/Documents/code/projects/lismore/data/sa1_allocations.csv")
    alloc = alloc[['SA1_CODE_2021','LGA_NAME_2023']]
    alloc.columns=['code','lga']
    alloc['code']=alloc['code'].astype(str)
    return alloc
alloc = get_alloc()


@st.cache_data
def get_base():
    base = pd.read_csv("/Users/tonyweir/Documents/code/projects/lismore/data/BASELINE DATA FOR GEODEM.CSV")
    base = base[['RecID',"SEG_50_NUM"]]
    base.columns = ['code','seg']
    base['code']=base['code'].astype(str)
    return base
base = get_base()

@st.cache_data
def get_vectors():
    geodem_scheme = pd.DataFrame({'seg':range(1,51)})

    lga_list = alloc['lga'].unique().tolist()
    vector_list = []
    for lga in lga_list:
        sa1_list = alloc[alloc['lga']==lga]['code'].tolist()
        lga_base = base[base['code'].isin(sa1_list)]
        lga_geodem = lga_base.groupby('seg').size().reset_index()
        lga_geodem = geodem_scheme.merge(lga_geodem, on='seg',how='left').fillna(0)
        lga_geodem.columns = ['seg','count']
        lga_geodem['fraction']=lga_geodem['count']/len(sa1_list)
        vector = lga_geodem['fraction'].tolist()
        vector_list.append(vector)
    lga_vectors = pd.DataFrame({'lga':lga_list,'vector':vector_list})
    return lga_vectors
lga_vectors = get_vectors()

def find_index_by_name(df, name):
    try:
        index = df.loc[df['lga'] == name].index[0]
        return index
    except IndexError:
        print(f"Name '{name}' not found in the DataFrame.")
        return None

def order_rows_by_similarity(df, given_row_index):
    # Extract the vector of the given row
    given_vector = np.array(df.iloc[given_row_index]['vector']).reshape(1, -1)

    # Calculate cosine similarity between the given vector and all other vectors in the dataframe
    similarities = [(given_row_index, 1.0)] 
    for index, row in df.iterrows():
        if index != given_row_index:  # Exclude the given row
            vector = np.array(row['vector']).reshape(1, -1)
            similarity = cosine_similarity(given_vector, vector)[0][0]
            similarities.append((index, similarity))

    # Sort the rows based on similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Extract the sorted indices
    sorted_indices = [index for index, _ in similarities]

    # Reorder the dataframe based on sorted indices
    ordered_df = df.iloc[sorted_indices].copy()

    # Add a new column to the dataframe with similarity values
    ordered_df['similarity'] = [similarity for _, similarity in similarities]
    ordered_df = ordered_df[['lga','similarity']]
    # Return the ordered dataframe
    return ordered_df



@st.cache_data
def get_lga_shapes():
    lga_shapes = gpd.read_file("data/geometry_files/LGA_simple_boundaries_2023/LGA_simple_boundaries_2023.shp")
    lga_ss = gpd.read_file("data/geometry_files/LGA_super_simple/LGA_super_simple.shp")
    lga_shapes['geometry']=lga_ss['geometry']
    lga_shapes = lga_shapes.dropna(subset=['AREASQKM'])
    return lga_shapes
lga_shapes = get_lga_shapes()




@st.cache_data
def get_aust_shape():
    aust = gpd.read_file("/Users/tonyweir/Documents/code/projects/lismore/data/geometry_files/AUST_SIMPLE/AUST_SIMPLE.shp")
    return aust
aust_shape = get_aust_shape()

lga_list = alloc['lga'].unique().tolist()
lis_index = lga_list.index('Lismore')
chosen_lga = st.selectbox("Select LGA",lga_list,lis_index)
sim = order_rows_by_similarity(lga_vectors,find_index_by_name(lga_vectors,chosen_lga))
sim['percent']=(round(sim['similarity']*100,0)).astype(str)+"%"
st.write(sim)

lga_shapes = lga_shapes.merge(sim, left_on='LGA_NAME23',right_on='lga',how='left')
lga_opacity = lga_shapes['similarity'].tolist()
colors = []
for opac in lga_opacity:
    colors.append([255,0,0,opac*255])
lga_shapes['color']=colors
# st.write(lga_shapes[['lga','color']])


# Map the 'segment' column to the 'color' column using the rgb_mapping
# st.write(sa2s_in_lga.head(10))

def get_map_layers(gdfs):
    first_gdf = gdfs[0]
    representative_point = first_gdf.head(1).geometry.representative_point().iloc[0]
    map_lat = representative_point.y
    map_lon = representative_point.x
    

    zoom = 3


    layers = []
 

    layers.append(pdk.Layer("GeoJsonLayer", data=gdfs[1], 
        pickable=True,
        get_fill_color=  'color',
        lineWidthScale= 1,
        getLineColor = [0,0,0,255],
        lineWidthMinPixels= 1,
        tooltip={"html": "{lga}"}
    ))


    return map_lon, map_lat, layers, zoom

long, lat, layers, zoom= get_map_layers([aust_shape, lga_shapes])
view = pdk.View(controller=True)
tool_tip = {"html": "{lga} - {percent}",
                    "style": { "backgroundColor": "darkslategray",
                                "color": "white"}
                  }


loc = st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v11',
        api_keys=None,
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=long,
            height=1000, width='100%',
            zoom=zoom,
            min_zoom = 1,
            max_zoom = 20,
            pitch=0,
        ),
        views=view,
        layers=layers,
        tooltip=tool_tip
        
    )
    )
