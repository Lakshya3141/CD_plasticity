# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:59:49 2023
@author: laksh
Creating plots similar to Figure 4
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os # For path names working under Linux and Windows
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

def fluc_pred(x, sig_z2, B, N, K, rho, sig_eps2, r):
    dummy = (np.sqrt(((1 + x/sig_z2)**3)/(x / sig_z2 * (N/K)**2)) + B**2*sig_eps2/2/sig_z2)/r
    return dummy*sig_z2

def nofluc_pred(x, sig_z2, N, K, r):
    return np.sqrt(((1 + x/sig_z2)**3)/(x / sig_z2 * (N/K)**2 * r**2))*sig_z2

def plast_pred(x, sig_z2, B, N, K, rho, sig_eps2, r):
    dum1 = np.sqrt(((1 + x/sig_z2)**3)/(x/sig_z2*N**2/K**2))
    dum2 = ((1 - rho**2)*B**2*sig_eps2/2/sig_z2)
    return (dum1 + dum2)/r*sig_z2

def fluc_pred_new(x, sig_z2, B, sig_eps2, r):
    return (4*x + 5*sig_z2 + (B**2)*sig_eps2)/(2*r)


def nofluc_pred_new(x, sig_z2, r):
    return (4*x + 5*sig_z2)/(2*r)

def plast_pred_new(x, sig_z2, B, rho, sig_eps2, r):
    return (4*x + 5*sig_z2 + (B**2)*sig_eps2*(1-rho**2))/(2*r)

def CD_delz_plotter(fn, subfold):
    data = pd.read_csv(filepath_or_buffer = os.path.join("hdf5", subfold, fn))
    data['sig_s2'] = data['sig_s']**2
    data['sig_u2'] = data['sig_u']**2
    data['sig_z2'] = data['sp1.Gaa'] + data['sp1.Gbb']*data.sig_eps**2 + data.sig_e**2
    
    sig_z2 = np.unique(data.sig_z2)[0]
    N = 0.5
    K = 1
    rho = np.unique(data.rho)[0]
    r = np.unique(data.r)[0]
    sig_eps2 = np.unique(data.sig_eps**2)[0]
    B= np.unique(data['sp1.B'])
    
    x = np.arange(np.min(data.sig_u)**2, np.max(data.sig_u)**2, (np.max(data.sig_u)**2 - np.min(data.sig_u)**2)/100)
    # fl = fluc_pred(x, sig_z2, B, N, K, rho, sig_eps2, r)
    fl = fluc_pred_new(x, sig_z2, B, sig_eps2, r)
    # nfl = nofluc_pred(x, sig_z2, N, K, r)
    nfl = nofluc_pred_new(x, sig_z2, r)
    # pls = plast_pred(x, sig_z2, B, N, K, rho, sig_eps2, r)
    pls = plast_pred_new(x, sig_z2, B, rho, sig_eps2, r)
    
    fig, ax = plt.subplots(figsize=(10,8))
    plt.xlabel("Width of resource utilization function ($\sigma_u^2$)", fontsize=22) 
    plt.ylabel("Width of stabilizing selection function ($\sigma_s^2$)", fontsize=22)  # Introduce newline
    cm = plt.cm.get_cmap('gist_yarg')
    plt.plot(x, fl, c = 'tab:blue', label = 'Fluctuating environment', linewidth=3, alpha=1.0)
    plt.plot(x, pls, c = 'tab:orange', label = 'Evolving plasticity', linewidth=3, alpha=1.0)
    plt.plot(x, nfl, c = 'tab:green', label = 'No fluctuations', linewidth=3, alpha=1.0)
    im = plt.scatter(data.sig_u2, data.sig_s2, c = data.delz, s = 30, cmap = cm)
    plt.ylim((np.min(data.sig_s2), np.max(data.sig_s2)))
    plt.xlim((np.min(data.sig_u2), np.max(data.sig_u2)))
    clb = fig.colorbar(im, ax=ax, ticks=np.arange(0, 13, 2), pad=0.02)  # Set ticks from 0 to 12, spaced by 2 units
    clb.ax.set_ylabel('CD : |$z̅_1$ - $z̅_2$|', fontsize=22)  # Set colorbar label
    clb.ax.tick_params(labelsize=19)  # Increase colorbar tick size
    # clb.set_ticks(np.arange(0, 13, 2))  # Set ticks from 0 to 12, spaced by 2 units
    plt.legend(loc='lower right', fontsize=19)  # Move legend to bottom right and increase font size
    plt.xticks(fontsize=18)  # Increase x-axis tick size
    plt.yticks(fontsize=18)  # Increase y-axis tick size
    plt.tight_layout()

    # To neutralize the effect of colorbar ticks on plot dimensions
    plt.subplots_adjust(right=0.85)  # Adjust right margin to make space for colorbar

    # Save the figure
    dumsav = 'images/' + subfold + '/' + fn[:-4] + "_delz" + '.jpg'
    fig.savefig(dumsav, dpi=550)

def CD_delb_plotter(fn, subfold):
    data = pd.read_csv(filepath_or_buffer = os.path.join("hdf5", subfold, fn))
    data['sig_s2'] = data['sig_s']**2
    data['sig_u2'] = data['sig_u']**2
    data['sig_z2'] = data['sp1.Gaa'] + data['sp1.Gbb']*data.sig_eps**2 + data.sig_e**2
    
    sig_z2 = np.unique(data.sig_z2)[0]
    N = 0.5
    K = 1
    rho = np.unique(data.rho)[0]
    r = np.unique(data.r)[0]
    sig_eps2 = np.unique(data.sig_eps**2)[0]
    B= np.unique(data['sp1.B'])
    
    x = np.arange(np.min(data.sig_u)**2, np.max(data.sig_u)**2, (np.max(data.sig_u)**2 - np.min(data.sig_u)**2)/100)
    fl = fluc_pred_new(x, sig_z2, B, sig_eps2, r)
    nfl = nofluc_pred_new(x, sig_z2, r)
    # pls = plast_pred(x, sig_z2, B, N, K, rho, sig_eps2, r)
    pls = plast_pred_new(x, sig_z2, B, rho, sig_eps2, r)
    
    fig, ax = plt.subplots(figsize=(11,8))
    plt.xlabel("Width of resource utilization function ($\sigma_u^2$)", fontsize=22) 
    plt.ylabel("Width of stabilizing selection function ($\sigma_s^2$)", fontsize=22)  # Introduce newline
    cm = plt.cm.get_cmap('gist_yarg')
    
    plt.plot(x, nfl, c = 'tab:orange', label = 'No fluctuations', linewidth=3, alpha=1.0)
    plt.plot(x, fl, c = 'tab:blue', label = 'Fluctuating environment', linewidth=3, alpha=1.0)
    plt.plot(x, pls, c = 'tab:green', label = 'Evolving plasticity', linewidth=3, alpha=1.0)
    im = plt.scatter(data.sig_u2, data.sig_s2, c = data.delb, s = 30, cmap = cm)
    plt.ylim((np.min(data.sig_s2), np.max(data.sig_s2)))
    plt.xlim((np.min(data.sig_u2), np.max(data.sig_u2)))
    clb = fig.colorbar(im, ax=ax, ticks=np.arange(-1, 1, 0.01), pad=0.02, fraction=0.1, shrink=0.8)  # Set ticks from 0 to 12, spaced by 2 units
    original_string = 'CD : |$z̅_1$ - $z̅_2$|'
    modified_string = original_string.replace('z', 'b')
    clb.ax.set_ylabel(modified_string, fontsize=22)  # Set colorbar label
    clb.ax.tick_params(labelsize=19)  # Increase colorbar tick size
    # clb.set_ticks(np.arange(-2, 2, 0.05))  # Set ticks from 0 to 12, spaced by 2 units
    plt.legend(loc='lower right', fontsize=19)  # Move legend to bottom right and increase font size
    plt.xticks(fontsize=18)  # Increase x-axis tick size
    plt.yticks(fontsize=18)  # Increase y-axis tick size
    plt.tight_layout()

    # To neutralize the effect of colorbar ticks on plot dimensions
    plt.subplots_adjust(right=0.85)  # Adjust right margin to make space for colorbar

    # Save the figure
    dumsav = 'images/' + subfold + '/' + fn[:-4] + "_delb" + '.jpg'
    fig.savefig(dumsav, dpi=550)

fnl = ["NoFluc_main.csv" , "Fluc_main.csv", "EvolvingPlasticity_main.csv"]
subfold = "final_sim"

# Create 'images' folder if it doesn't exist
if not os.path.exists(os.path.join('images', subfold)):
    os.makedirs(os.path.join('images', subfold))
    

for i in fnl:
    print(f'STARTING {i}')
    # CD_delz_plotter(i, subfold)
    
CD_delb_plotter(fnl[2], subfold)

# List to hold the opened images
images = []

# Read the generated EPS images and append them to the 'images' list
for i in fnl:
    image_path = os.path.join('images', subfold, f'{i[:-4]}' + '_delz' + '.jpg')
    images.append(Image.open(image_path))

# Create a new blank image with the size of the concatenated images
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_image = Image.new('RGB', (total_width, max_height))

# Paste the images into the new blank image with tags A, B, and C
x_offset = 0
# Define the font
font = ImageFont.load_default()
for i, img in enumerate(images):
    new_image.paste(img, (x_offset, 0))
    x_offset += img.size[0]
    draw = ImageDraw.Draw(new_image)
    draw.text((x_offset - img.size[0] + 10, 10), f'Tag {chr(65+i)}', (255, 255, 255), font=font)

# Save the concatenated image
new_image_path = os.path.join('images', subfold, 'concatenated_image.png')
new_image.save(new_image_path)

print('Concatenated image saved successfully.')

