import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def fluc_pred_new(x, sig_z2, B, sig_eps2, r):
    return (4*x + 5*sig_z2 + (B**2)*sig_eps2)/(2*r)


def nofluc_pred_new(x, sig_z2, r):
    return (4*x + 5*sig_z2)/(2*r)

def plast_pred_new(x, sig_z2, B, rho, sig_eps2, r):
    return (4*x + 5*sig_z2 + (B**2)*sig_eps2*(1-rho**2))/(2*r)


fnl = ["NoFluc_main.csv" , "Fluc_main.csv", "EvolvingPlasticity_main.csv"]
subfold = "final_sim"

# Create 'images' folder if it doesn't exist
if not os.path.exists(os.path.join('images', subfold)):
    os.makedirs(os.path.join('images', subfold))

datas = []
for i, fn in enumerate(fnl):
    datas.append(pd.read_csv(filepath_or_buffer=os.path.join("hdf5", subfold, fn)))
    datas[i]['sig_s2'] = datas[i]['sig_s']**2
    datas[i]['sig_u2'] = datas[i]['sig_u']**2
    datas[i]['sig_z2'] = datas[i]['sp1.Gaa'] + datas[i]['sp1.Gbb']*datas[i].sig_eps**2 + datas[i].sig_e**2

sig_z2 = np.unique(datas[0].sig_z2)[0]
rho = np.unique(datas[0].rho)[0]
r = np.unique(datas[0].r)[0]
sig_eps2 = np.unique(datas[0].sig_eps**2)[0]
B = np.unique(datas[0]['sp1.B'])

x = np.arange(np.min(datas[0].sig_u)**2, np.max(datas[0].sig_u)**2, (np.max(datas[0].sig_u)**2 - np.min(datas[0].sig_u)**2)/100)
nfl = nofluc_pred_new(x, sig_z2, r)
fl = fluc_pred_new(x, sig_z2, B, sig_eps2, r)
pls = plast_pred_new(x, sig_z2, B, rho, sig_eps2, r)

leg_size = 23
## Plotting the no fluctuations scenario
fig, ax = plt.subplots(figsize=(10,8))
plt.xlabel("Resource utilization breadth ($\sigma_u^2$)", fontsize=28) 
plt.ylabel("Stabilizing selection breadth ($\sigma_s^2$)", fontsize=28)  
plt.ylim((np.min(datas[0].sig_s2), np.max(datas[0].sig_s2)))
plt.xlim((np.min(datas[0].sig_u2), np.max(datas[0].sig_u2)))
cm0 = plt.cm.get_cmap('gist_yarg')
im0 = plt.scatter(datas[0].sig_u2, datas[0].sig_s2, c=datas[0].delz, s=40, cmap=cm0, alpha=1.0)
plt.plot(x, nfl, c = 'tab:orange', label = 'No fluctuations', linewidth=3, alpha=1.0)
clb0 = fig.colorbar(im0, ax=ax, ticks=np.arange(0, 13, 2), pad=0.02, fraction=0.1, shrink=0.8) # Adjust shrink value
clb0.ax.set_ylabel('CD : |$z̅_1$ - $z̅_2$|', fontsize=25)
clb0.ax.tick_params(labelsize=18)
ax.tick_params(axis='both', which='major', labelsize=23)
plt.legend(loc='lower right', fontsize=leg_size)
plt.tight_layout()
plt.show()
dumsav = 'images/' + subfold + '/' + fnl[0][:-4] + '.jpg'
fig.savefig(dumsav, dpi=550)

## Plotting the fluctuations scenario
fig, ax = plt.subplots(figsize=(10,8))
plt.xlabel("Resource utilization breadth ($\sigma_u^2$)", fontsize=28) 
plt.ylabel("Stabilizing selection breadth ($\sigma_s^2$)", fontsize=28)  
plt.ylim((np.min(datas[1].sig_s2), np.max(datas[1].sig_s2)))
plt.xlim((np.min(datas[1].sig_u2), np.max(datas[1].sig_u2)))
cm0 = plt.cm.get_cmap('gist_yarg')
im0 = plt.scatter(datas[1].sig_u2, datas[1].sig_s2, c=datas[1].delz, s=40, cmap=cm0, alpha=1.0)
plt.plot(x, nfl, c = 'tab:orange', label = 'No fluctuations', linewidth=3, alpha=1.0)
plt.plot(x, fl, c = 'tab:blue', label = 'Fluctuating env.', linewidth=3, alpha=1.0)
clb0 = fig.colorbar(im0, ax=ax, ticks=np.arange(0, 13, 2), pad=0.02, fraction=0.1, shrink=0.8) # Adjust shrink value
clb0.ax.set_ylabel('CD : |$z̅_1$ - $z̅_2$|', fontsize=25)
clb0.ax.tick_params(labelsize=18)
ax.tick_params(axis='both', which='major', labelsize=23)
plt.legend(loc='lower right', fontsize=leg_size)
plt.tight_layout()
plt.show()
dumsav = 'images/' + subfold + '/' + fnl[1][:-4] + '.jpg'
fig.savefig(dumsav, dpi=550)
# plt.plot(x, fl, c = 'tab:blue', label = 'Fluctuating environment', linewidth=3, alpha=1.0)
# plt.plot(x, pls, c = 'tab:green', label = 'Evolving plasticity', linewidth=3, alpha=1.0)

## Plotting the evolving scenario
fig, ax = plt.subplots(figsize=(10,8))
plt.xlabel("Resource utilization breadth ($\sigma_u^2$)", fontsize=28) 
plt.ylabel("Stabilizing selection breadth ($\sigma_s^2$)", fontsize=28)  
plt.ylim((np.min(datas[2].sig_s2), np.max(datas[2].sig_s2)))
plt.xlim((np.min(datas[2].sig_u2), np.max(datas[2].sig_u2)))
cm0 = plt.cm.get_cmap('gist_yarg')
im0 = plt.scatter(datas[2].sig_u2, datas[2].sig_s2, c=datas[2].delz, s=40, cmap=cm0, alpha=1.0)
plt.plot(x, nfl, c = 'tab:orange', label = 'No fluctuations', linewidth=3, alpha=1.0)
plt.plot(x, fl, c = 'tab:blue', label = 'Fluctuating env.', linewidth=3, alpha=1.0)
plt.plot(x, pls, c = 'tab:green', label = 'Evolving plasticity', linewidth=3, alpha=1.0)
clb0 = fig.colorbar(im0, ax=ax, ticks=np.arange(0, 13, 2), pad=0.02, fraction=0.1, shrink=0.8) # Adjust shrink value
clb0.ax.set_ylabel('CD : |$z̅_1$ - $z̅_2$|', fontsize=25)
clb0.ax.tick_params(labelsize=18)
ax.tick_params(axis='both', which='major', labelsize=23)
plt.legend(loc='lower right', fontsize=leg_size)
plt.tight_layout()
plt.show()
dumsav = 'images/' + subfold + '/' + fnl[2][:-4] + '.jpg'
fig.savefig(dumsav, dpi=550)

## Concatenation
# List to hold the opened images
images = []

# Read the generated EPS images and append them to the 'images' list
for i in fnl:
    image_path = os.path.join('images', subfold, f'{i[:-4]}' + '.jpg')
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

## Plasticity plotter
fig, ax = plt.subplots(figsize=(10,8))
plt.xlabel("Resource utilization breadth ($\sigma_u^2$)", fontsize=28) 
plt.ylabel("Stabilizing selection breadth ($\sigma_s^2$)", fontsize=28)  
plt.ylim((np.min(datas[2].sig_s2), np.max(datas[2].sig_s2)))
plt.xlim((np.min(datas[2].sig_u2), np.max(datas[2].sig_u2)))
cm0 = plt.cm.get_cmap('gist_yarg')
im0 = plt.scatter(datas[2].sig_u2, datas[2].sig_s2, c=datas[2].delb, s=40, cmap=cm0, alpha=1.0)
plt.plot(x, nfl, c = 'tab:orange', label = 'No fluctuations', linewidth=3, alpha=1.0)
plt.plot(x, fl, c = 'tab:blue', label = 'Fluctuating env.', linewidth=3, alpha=1.0)
plt.plot(x, pls, c = 'tab:green', label = 'Evolving plasticity', linewidth=3, alpha=1.0)
clb0 = fig.colorbar(im0, ax=ax, ticks=np.arange(0, 13, 0.01), pad=0.02, fraction=0.1, shrink=0.8) # Adjust shrink value
clb0.ax.set_ylabel('CD : |$b̅_1$ - $b̅_2$|', fontsize=25)
clb0.ax.tick_params(labelsize=18)
ax.tick_params(axis='both', which='major', labelsize=23)
plt.legend(loc='lower right', fontsize=leg_size)
plt.tight_layout()
plt.show()
dumsav = 'images/' + subfold + '/' + fnl[2][:-4] + "_delb" + '.jpg'
fig.savefig(dumsav, dpi=550)