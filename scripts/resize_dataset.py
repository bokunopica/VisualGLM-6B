import os
from PIL import Image
from tqdm import trange

if __name__ == "__main__":
    file_path =  "/home/qianq/data/OpenI-zh"
    save_path = "/home/qianq/data/OpenI-zh-resize-384"
    image_name_list = os.listdir(f"{file_path}/images")
    for i in trange(len(image_name_list)):
        image_name = image_name_list[i]
        image_path = f"{file_path}/images/{image_name}"
        image_save_path = f"{save_path}/images/{image_name}"
        Image.open(image_path).resize((384, 384)).save(image_save_path)

    
    
