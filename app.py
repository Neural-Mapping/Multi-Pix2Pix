import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from dotenv import load_dotenv
load_dotenv()

from sentinelhub import SHConfig
import os
config_sentinel = SHConfig(sh_client_id=os.environ.get("sh_client_id"), sh_client_secret=os.environ.get("sh_client_secret"))
print(config_sentinel.sh_client_id)

from sentinelhub import SHConfig
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)

from dotenv import load_dotenv
import numpy as np
from utils import get_map
import datetime
import os
import matplotlib.pyplot as plt
from sentinelhub import CRS, BBox, bbox_to_dimensions
from utils import evalscript_true_color, evalscript_ndvi, evalscript_ndwi
from utils import center_crop
from config import IMAGE_SIZE
from utils import to_grayscale
import cv2
from utils import generated_lsm_mask
from PIL import Image
from torch.utils.data import DataLoader
from dataset import Image_dataset
from generator_model import Generator
from torch import optim
import torch
from config import DEVICE
import folium

# Load model
gen = Generator(in_channels=3, inter_images=4, out_channels=3)
checkpoint = torch.load("model/gen_LSM_v2.pth.tar", map_location=torch.device(DEVICE))
gen.load_state_dict(checkpoint['state_dict'])

optimizer = optim.Adam(gen.parameters())
optimizer.load_state_dict(checkpoint['optimizer'])


from mapping_automation import find_best_date, subtract_km_from_coordinates, get_cloud_coverage, search_available_dates, get_access_token, get_slope_elevation
from scripts import *
from utils import *
import os


import gradio as gr

def get_images(km, grid, grid_dim, script, box_dim=400, date_start = "2024-04-12", date_end = "2024-04-12", res=2100, file_name=None):
    _box_dim = 1000 if km < 100 else km
    canvas = np.zeros(((grid_dim) * _box_dim, (grid_dim) * _box_dim, 3), dtype=np.uint8)

    row = 0
    col = 0

    for idx, i in enumerate(range(len(grid))):
        y_start = _box_dim * col
        y_end = _box_dim * (col + 1)
        x_start = _box_dim * row
        x_end = _box_dim * (row + 1)

        print(col, row, grid[idx], "->", y_start, y_end, x_start, x_end) 

        image_rgba = cv2.resize(
        get_suseptibility_mapping(grid[idx], script, date_start=date_start, date_end=date_end, res=res, box_dim=box_dim), (_box_dim,_box_dim)
        )
        if image_rgba.shape[-1] == 4:
            image_rgb = image_rgba[..., :3]
        else: image_rgb = image_rgba
        canvas[y_start:y_end, x_start:x_end] = image_rgb

        if file_name: 
            plt.imsave(f"{file_name}.png", canvas)
            print(f"Saved: {file_name}.png")

        row += 1  # Move to the next column
        if (idx + 1) % math.sqrt(len(grid)) == 0:
            print("----") 
            col += 1  # Move to the next row
            row = 0  # Reset column position
    return canvas


def generate_image(lat, lon, box_dim, grid, date):
    box_dim = int(box_dim)
    lat = float(lat)
    lon = float(lon)
    box_dim = int(box_dim)
    grid = int(grid)
    g = generate_grid(lat, lon, distance=box_dim*1000, grid_side=grid)
    m = folium.Map(
        location=(lat, lon),
        zoom_start=15,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri'
    )   
    for i in range(len(g)):
        folium.Rectangle([(g[i][:2]), (g[i][2:])], color='red', fill='pink',fill_opcity=0.5).add_to(m)
    res = 1000
    row = 0
    col = 0
    get_new_dates_for_each = True

    _box_dim = 1000 if box_dim < 100 else box_dim

    canvas_True_Color_After = np.zeros(((grid) * _box_dim, (grid) * _box_dim, 3), dtype=np.uint8)
    canvas_NDVI_Before = np.zeros(((grid) * _box_dim, (grid) * _box_dim, 3), dtype=np.uint8)
    canvas_Elevation = np.zeros(((grid) * _box_dim, (grid) * _box_dim, 3), dtype=np.uint8)
    canvas_Slope = np.zeros(((grid) * _box_dim, (grid) * _box_dim, 3), dtype=np.uint8)
    canvas_NDWI_Before = np.zeros(((grid) * _box_dim, (grid) * _box_dim, 3), dtype=np.uint8)
    canvas_generated_output = np.zeros(((grid) * _box_dim, (grid) * _box_dim, 3), dtype=np.uint8)/255

    for idx, i in enumerate(range(len(g))):
        if get_new_dates_for_each or idx==0:
            lat, lon = g[idx][0], g[idx][1]
            print("Getting dates for new lat lon")
            available_dates = search_available_dates(target_date=date,
                                                    lat = lat,
                                                    lon = lon)
            available_dates_cloud_coverage = get_cloud_coverage(lat=lat, lon=lon, date_list=available_dates)
            min_before_date, min_before_cc, min_after_date, min_after_cc = find_best_date({k: v for k, v in available_dates_cloud_coverage.items() if v is not None}, target_date=date)
            print(min_before_date,":" , min_before_cc, min_after_date,":" ,min_after_cc)
        
        y_start = _box_dim * col
        y_end = _box_dim * (col + 1)
        x_start = _box_dim * row
        x_end = _box_dim * (row + 1)

        print("NDVI_Before")
        NDVI_Before = get_images(km=box_dim, grid=[g[i]], grid_dim=1, script=evalscript_True_Color, 
            date_start=min_before_date, date_end=min_before_date, res=res, 
            box_dim=box_dim)

        print("True_Color_After")
        True_Color_After = get_images(km=box_dim, grid=[g[i]], grid_dim=1, script=evalscript_True_Color, 
                date_start=min_before_date, date_end=min_before_date, res=res,                            ## idk fix this: True_Color_After should have been True_Color_Before
                box_dim=box_dim)
        canvas_True_Color_After[y_start:y_end, x_start:x_end] = True_Color_After

        print("NDWI_Before")
        NDWI_Before = get_images(km=box_dim, grid=[g[i]], grid_dim=1, script=evalscript_NDWI, 
                date_start=min_before_date, date_end=min_before_date, res=res, 
                box_dim=box_dim)
        canvas_NDWI_Before[y_start:y_end, x_start:x_end] = NDWI_Before

        slope_arr, elevation_arr = get_slope_elevation([g[idx]])
        slope_arr = cv2.resize(slope_arr, (1000,1000))
        elevation_arr = cv2.resize(elevation_arr, (1000,1000))

        print("Generated Output")
        generated_output = gen(torch.tensor(cv2.resize(True_Color_After, (512, 512))).permute(2, 0, 1).unsqueeze(0).to(torch.float32),
        z1 = torch.tensor(cv2.resize(NDVI_Before, (512, 512))).permute(2, 0, 1).unsqueeze(0).to(torch.float32),
        z2 = torch.tensor(cv2.resize(slope_arr, (512, 512))).unsqueeze(-1).repeat(1, 1, 3).permute(2, 0, 1).unsqueeze(0).to(torch.float32),
        z3 = torch.tensor(cv2.resize(elevation_arr, (512, 512))).unsqueeze(-1).repeat(1, 1, 3).permute(2, 0, 1).unsqueeze(0).to(torch.float32),
        z4 = torch.tensor(cv2.resize(NDWI_Before, (512, 512))).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        )
        canvas_generated_output[y_start:y_end, x_start:x_end] = cv2.resize(generated_output[0].permute(1, 2, 0).detach().cpu().numpy(), (1000,1000))*0.5+0.5

        row += 1  # Move to the next column
        if (idx + 1) % math.sqrt(len(g)) == 0:
            print("----") 
            col += 1  # Move to the next row
            row = 0  # Reset column position

    return canvas_generated_output

# Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Latitude", max_lines=1),
        gr.Textbox(label="Longitude", max_lines=1),
        gr.Textbox(label="Box Dimension", max_lines=1),
        gr.Textbox(label="Grid", max_lines=1),
        gr.Textbox(label="Date", max_lines=1),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Landslide Susceptiblity Mapping"
)

iface.launch()