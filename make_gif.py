# import imageio

# images = []

# for filename in './vqt_result/*.png':
#     images.append(imageio.imread(filename))
# imageio.mimsave('./vqt_result/vqt.gif', images)

from PIL import Image
import glob
 
# Create the frames
frames = []
# imgs = glob.glob("./vqt_result/*.png")
for i in range(26):
    new_frame = Image.open(f"./vqt_complex_result/ddpm_sample_vqt{i}.png")
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('./vqt_complex.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)