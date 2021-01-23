import pandas as pd

df_train = pd.read_csv('./nyc-taxi-trip-duration/train.csv')
#print(df_train.head())

# data in day and hours
df = df_train[['pickup_datetime', 'pickup_longitude', 'pickup_latitude']].copy()
df['day_of_week'] = pd.to_datetime(df['pickup_datetime']).dt.dayofweek  #0,1,2,3,...
df['hour_of_day'] = pd.to_datetime(df['pickup_datetime']).dt.hour #int
#print(df.head())

import folium
from folium.plugins import HeatMap
from IPython.display import IFrame
#define a map function
def embed_map(map, filename):
    map.save(filename)
    return IFrame(filename, width='100%', height = '500px')
# #plot map by hour and day of week
# day_dict = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
# for i in range(df.day_of_week.min(), df.day_of_week.max()+1):
#     for j in range(df.hour_of_day.min(), df.hour_of_day.max()+1):
#         # Filter to include only data for each day of week and hour of day
#         df_geo = df.loc[(df.day_of_week == i)&(df.hour_of_day == j)][['pickup_latitude', 'pickup_longitude']].copy()
#         # Instantiate map object
#         map_5 = folium.Map(location=[40.75, -73.96], tiles='openstreetmap', zoom_start=12)
#         # Plot heatmap
#         HeatMap(data=df_geo, radius=10).add_to(map_5)
#         # Get day of week string from dow_dict
#         d = day_dict.get(i)
#         # Add title to heatmap
#         title_html = f'''<h3 align="center" style="font-size:20px">
#                                 <b>NYC Cab Pickups at {j}:00 on {d}: {len(df_geo)} rides</b></h3>
#                              '''
#         map_5.get_root().html.add_child(folium.Element(title_html))
#         # Save map
#         embed_map(map_5, f'./savemap_html/{i}_{j}_heatmap.html')


from selenium import webdriver
#
# for i in range(0, 7):
#     for j in range(0, 24):
#         # Set file path
#         tmpurl =f'file:////Users/momo/Documents/mhf/ NYC Cab Trip Duration/savemap_html/{i}_{j}_heatmap.html'
#
#         # Set browser to Chrome
#         browser = webdriver.Chrome()
#
#         # Open file in browser
#         browser.get(tmpurl)
#
#         # If hour is < 10, add 0 for sorting purposes and to keep chronological order
#         if j < 10:
#             browser.save_screenshot(f'./maps_png_pickup/{i}_0{j}_heatmap.png')
#         else:
#             browser.save_screenshot(f'./maps_png_pickup/{i}_{j}_heatmap.png')
#
#         # Close browser
#         browser.quit()

from PIL import Image
import glob
def png_to_gif(path_to_images, save_file_path, duration=500):
    frames = []
    # Retrieve image files
    images = glob.glob(f'{path_to_images}')
    # Loop through image files to open, resize them and append them to frames
    for i in sorted(images):
        im = Image.open(i)
        im = im.resize((550, 389), Image.ANTIALIAS)
        frames.append(im.copy())
        # Save frames/ stitched images as .gif
        frames[0].save(f'{save_file_path}', format='GIF', append_images=frames[1:], save_all=True,
                       duration=duration, loop=0)
png_to_gif(path_to_images='./maps_png_pickup/*.png',
           save_file_path='./plot/pickup_heatmap.gif',
           duration=500)


