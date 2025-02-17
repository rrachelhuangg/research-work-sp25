import os
import ollama
import time
from PIL import Image
import numpy as np

svbrdfs_dir = "matfusion/datasets/cc0_svbrdfs"
diffuse_dir = svbrdfs_dir + "/diffuse"
normals_dir = svbrdfs_dir + "/normals"
roughness_dir = svbrdfs_dir + "/roughness"
specular_dir = svbrdfs_dir + "/specular"

def generate_caption(img_path, svbrdf_descriptions):
    """Generates a caption describing the input diffuse image"""
    model = "llava:7b-v1.6"
    response = ollama.chat(
        model=model,
        messages = [
            {
                "role":"user",
                "content":f"Please use one sentence to describe the material in this image. Take note of the given characteristics of {svbrdf_descriptions[0], svbrdf_descriptions[1], svbrdf_descriptions[2]}. Include a description of the lighting and roughness of the material. Be as specific as possible and make sure to condense your response into a single sentence.",
                "images": [img_path]
            }
        ]
    )
    return response["message"]["content"]

def average_value(img_path):
    """Returns the average pixel value of the input image"""
    img = Image.open(img_path)
    img_array = np.array(img)
    value = np.mean(img_array)
    return round(float(value),6)

def standard_deviation(img_path):
    """Returns the standard deviation of pixel values in the input image"""
    img = Image.open(img_path)
    img_array = np.array(img)
    value = np.std(img_array)
    return round(float(value),6)

def describe_specular(value):
    if value in range(10, 15):
        return "dull"
    elif value in range(15, 20):
        return "glossy"
    else:
        return "extra shiny"

def describe_roughness(value):
    if value in range(50, 100):
        return "decently smooth"
    elif value in range(100, 150):
        return "a bit rough"
    else:
        return "pretty rough"

def describe_normals(avg_value, stdev_value):
    if avg_value < 160:
        return "low roughness"
    elif avg_value in range(160, 170):
        return "average roughness"
    else:
        return "high roughness"

output_directory = "training_dataset"
start_time = time.time()
count = 0
for file_name in os.listdir(diffuse_dir):
    image = file_name[:-12]
    svbrdf_descriptions = []
    diffuse_file = os.path.join(diffuse_dir, image+"_diffuse.png")
    specular_file = os.path.join(specular_dir, image+"_specular.png")
    roughness_file = os.path.join(roughness_dir, image+"_roughness.png")
    normals_file = os.path.join(normals_dir, image+"_normals.png")
    if os.path.isfile(diffuse_file):
        new_file = os.path.join(output_directory, f"{image}.txt")
        with open(new_file, "w") as n:
            svbrdf_descriptions += [describe_specular(average_value(specular_file))]
            svbrdf_descriptions += [describe_roughness(average_value(roughness_file))]
            svbrdf_descriptions += [describe_normals(average_value(normals_file), 0)]
            n.write(generate_caption(diffuse_file, svbrdf_descriptions)+"\n")
        count += 1
    if count % 1000 == 0:
        print(f"Working on image: {count}...")
end_time = time.time()
print(f"Took {round(end_time-start_time,3)}s for {count} images.")
