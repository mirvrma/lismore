import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import math
from shapely.geometry import box
import seaborn as sns
import numpy as np
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
def get_lga_shape():
    lga_shapes = gpd.read_file("data/geometry_files/lismore_lga/lismore_lga.shp")
    lga_shapes = lga_shapes.dropna(subset=['AREASQKM'])
    return lga_shapes
lis_lga23_shape = get_lga_shape()




@st.cache_data
def get_sa2_shapes():
    sa2_shapes = gpd.read_file("data/geometry_files/lismore_sa2/lismore_sa2.shp")
    sa2_shapes = sa2_shapes.dropna(subset=['geometry'])
    sa2_shapes = sa2_shapes[sa2_shapes['STE_CODE21']!="9"]
    sa2_shapes = sa2_shapes.rename(columns={"SA2_MAIN16":"SA2_CODE_2016"})
    return sa2_shapes
lis_sa216_shapes = get_sa2_shapes()
# st.write(lis_sa216_shapes.tail())


@st.cache_data
def get_sa1_shapes():
    sa1_shapes = gpd.read_file("data/geometry_files/lismore_sa1/lismore_sa1.shp")
    sa1_shapes = sa1_shapes.dropna(subset=['geometry'])
    sa1_shapes = sa1_shapes[sa1_shapes['STE_CODE21']!="9"]
    sa1_shapes = sa1_shapes.rename(columns={"SA1_MAIN16":"SA1_CODE_2016"})
    sa1_shapes['SA1_7DIG16']=sa1_shapes['SA1_7DIG16'].astype(str)
    return sa1_shapes
lis_sa1 = get_sa1_shapes()
# st.write(lis_sa1.tail())


@st.cache_data
def get_geodem_allocations():
    file_path = 'data/geodem/Lismore indexes.xlsx'
    sheet_name = '2016_SA1'
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.rename(columns={'SA1_16':'SA1_CODE_2016'})
    df['SA1_CODE_2016']=df['SA1_CODE_2016'].astype(str)
    return df
geodem_alloc = get_geodem_allocations()

@st.cache_data
def get_geodem_groups():
    file_path = 'data/geodem/Lismore indexes.xlsx'
    sheet_name = 'Lismore Groups'
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df
lis_groups = get_geodem_groups()
cols = lis_groups.columns.tolist()
cols = ['Segment number','Description']+cols[2:]
lis_groups.columns = cols

lis_sa1 = lis_sa1.merge(geodem_alloc, left_on="SA1_7DIG16", right_on='SA1_CODE_2016', how='left')

lis_sa1 = lis_sa1.merge(lis_groups, on='Segment number',how='left')
lis_sa1 = lis_sa1.rename(columns = {"SA1_CODE_2016_x":"SA1_CODE_2016"})
lis_sa1 = lis_sa1.dropna(subset=['Segment number'])
lis_sa1 = lis_sa1.rename(columns = {'Segment number':'segment'})
colors = sns.color_palette("husl", n_colors=26)




chosen_state = "New South Wales"
chosen_lga = "Lismore"



st.header(chosen_lga)
lis_sa216_shapes['representative_point']= lis_sa216_shapes["geometry"].representative_point()
sa2s_in_lga = lis_sa216_shapes[lis_sa216_shapes['representative_point'].within(lis_lga23_shape.geometry.iloc[0])].drop(columns = ['representative_point'])
sa1s_in_lga = lis_sa1[lis_sa1['SA2_MAIN16'].isin(sa2s_in_lga['SA2_CODE_2016'].tolist())]
# Map the unique segments to distinct colors
unique_segments = set(sa1s_in_lga['segment'].unique())

rgb_mapping = {segment: color for segment, color in zip(unique_segments, distinct_colors)}

sa1s_in_lga['color'] = sa1s_in_lga['segment'].map(rgb_mapping)
def fill_nan_with_black(color):
    return color if isinstance(color, list) else [0, 0, 0, 255]

# Apply the function to fill NaN values in the 'color' column with black
sa1s_in_lga['color'] = sa1s_in_lga['color'].apply(fill_nan_with_black)


# Map the 'segment' column to the 'color' column using the rgb_mapping
# st.write(sa2s_in_lga.head(10))
columns = st.columns([3,6])
with columns[0]:
    opacity = st.slider("Select Opacity", 0.0, 1.0, 0.5)



def get_map_layers(gdfs):
    first_gdf = gdfs[0]
    representative_point = first_gdf.head(1).geometry.representative_point().iloc[0]
    map_lat = representative_point.y
    map_lon = representative_point.x
    area = first_gdf.head(1)['AREASQKM'].iloc[0]

    base_zoom = 13
    zoom_factor = 1.4  # Adjust as needed

    zoom = base_zoom - zoom_factor * math.log(area, 10)
    zoom =  max(6, min(13, zoom))


    layers = []
    layers = []
    fill_colors =[[0,0,255,20],[0,0,0,0],[0,0,0,0]]
    line_colors =[[0,0,255,255],[255,0,0,255],[255,255,0,255]]
    line_widths = [20,10,5]
    line_min_widths = [4,2,1]
# 
    #lga
    layers.append(pdk.Layer("GeoJsonLayer", data=gdfs[0], 
    pickable=True,
    get_fill_color=  fill_colors[0],
    opacity=0.5,        
    lineWidthScale= line_widths[0],
    getLineColor = line_colors[0],
    lineWidthMinPixels= line_min_widths[0],)
    ) 
    # #sa2
    # layers.append(pdk.Layer("GeoJsonLayer", data=gdfs[1], 
    # pickable=True,
    # get_fill_color=  fill_colors[1],
    # opacity=0.5,        
    # lineWidthScale= line_widths[1],
    # getLineColor = line_colors[1],
    # lineWidthMinPixels= line_min_widths[1],)
    # )    

    #sa1
    layers.append(pdk.Layer("GeoJsonLayer", data=gdfs[2], 
        pickable=True,
        get_fill_color=  'color',
        opacity=opacity*0.3,        
        lineWidthScale= line_widths[2],
        getLineColor = [0,0,0,255],
        lineWidthMinPixels= line_min_widths[2],
        update_triggers={"opacity": opacity},
        tooltip={"html": "<b>Description:</b> {Description}"}
    ))


    return map_lon, map_lat, layers, zoom

long, lat, layers, zoom= get_map_layers([lis_lga23_shape, sa2s_in_lga, sa1s_in_lga])
view = pdk.View(controller=True)
tool_tip = {"html": "{Description}",
                    "style": { "backgroundColor": "darkslategray",
                                "color": "white"}
                  }

columns = st.columns([3,6,3])
with columns[1]:
    loc = st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v11',
            api_keys=None,
            initial_view_state=pdk.ViewState(
                latitude=lat,
                longitude=long,
                height=500, width='100%',
                zoom=zoom,
                min_zoom = 5,
                max_zoom = 20,
                pitch=0,
            ),
            views=view,
            layers=layers,
            tooltip=tool_tip
            
        )
        )
st.markdown("<h3>Legend</h3>", unsafe_allow_html=True)

# Get unique segments and their corresponding colors
unique_segments = sa1s_in_lga['segment'].unique()

# Divide the unique segments into 4 columns
num_columns = 4
columns = st.columns(num_columns)

# Iterate through unique segments
for i, segment in enumerate(unique_segments):
    # Find the first row with the current segment
    legend_row = sa1s_in_lga[sa1s_in_lga['segment'] == segment].iloc[0]

    # Extract information for the legend
    description = legend_row['Description']
    color = legend_row['color']

    # Check if color is a valid RGBA list
    if isinstance(color, list):
        # Create a swatch with the color and a tooltip
        swatch_html = f'<div style="background-color: rgba({", ".join(map(str, color))}); width: 20px; height: 20px;"></div>'

        # Display the swatch and tooltip in the corresponding column
        columns[i % num_columns].markdown(f'{swatch_html} {description}', unsafe_allow_html=True)
    else:
        st.warning(f"Invalid color for segment {segment}")
        st.write(legend_row)


st.subheader("Summary for Lismore")

lis_groups = lis_groups.sort_values(by='Total pop', ascending=False)
st.dataframe(lis_groups)

st.write("The tables below show the overrepresentation or underrepresentation of attributes for the top 10 segments (80% of the population).")
def get_attributes(sheet_name):
    file_path = 'data/geodem/Lismore indexes.xlsx'
    sheet_name = sheet_name
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.iloc[:, :12].copy()
    cols = df.columns.tolist()
    cols = cols[1:]
    for col in cols:
        df[col]=df[col].astype(float)
        df[col]=round(df[col]/100,2)
       
    # df = df.iloc[:, :11]
    # Define custom coloring function
    def color_cells(value):
        dark_green = 'background-color: darkgreen; color: white'
        medium_green = 'background-color: mediumseagreen; color: white'
        light_green = 'background-color: lightgreen; color: black'
        no_color = ''
        light_red = 'background-color: lightcoral; color: black'
        medium_red = 'background-color: indianred; color: white'
        dark_red = 'background-color: darkred; color: white'

        if isinstance(value, (int, float)):
            if value >= 5:
                return dark_green
            elif 3 <= value < 5:
                return medium_green
            elif 1.25 <= value < 3:
                return light_green
            elif 0.8 <= value < 1.25:
                return no_color
            elif 0.33 <= value < 0.8:
                return light_red
            elif 0.2 <= value < 0.33:
                return medium_red
            elif value < 0.2:
                return dark_red
            else:
                return no_color
        else:
            return no_color

    # Apply the custom coloring function to the DataFrame
    styled_df = df.style.applymap(color_cells, subset=cols)

    # Display the styled DataFrame
    st.dataframe(styled_df)

st.subheader("Occupations")
get_attributes('Occupations')

st.subheader("Characteristics of people")
get_attributes('Person')

st.subheader("Characteristics of families")
get_attributes('Family')

st.subheader("Birthplace")
get_attributes('Birthplace')

st.subheader("Field of study")
get_attributes('Field of study')


st.subheader("Education attained")
get_attributes('Ed attained')

st.subheader("Ancestry")
get_attributes('Sheet8 (2)')

st.subheader("Work hours")
get_attributes('work hours')


