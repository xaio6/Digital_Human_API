import base64
import json
import os
import re
import shutil
from PIL import Image

#验证账号密码
def Verification(name, password):
    with open("./DataBase/Login.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if name in data:
        if data[name] == password:
            return True
        else:
            return False
    else:
        return False

# 修改照片大小
def Change_image_Size(image_path):
    # 打开原图像
    original_image = Image.open(image_path)
    # 获取照片大小
    width, height = original_image.size
    if width >= 1500 and height >= 1500:
        # 缩小尺寸
        width *= 0.4
        height *= 0.4

        # 调整图像大小
        reduced_image = original_image.resize((int(width), int(height)))

        # 替换原图像
        reduced_image.save(image_path)
        return image_path
    else:
        return image_path


# 清空文件夹
def Clear_File(file_path):
    file_list = os.listdir(file_path)
    if (len(file_list) > 0):
        shutil.rmtree(file_path)
        os.makedirs(file_path)
        
#编码拼接后的视频
def Encode_Video(video):
    with open(video, 'rb') as f:
        video_data = f.read()
        video_data_base64 = base64.b64encode(video_data).decode('utf-8')
    
    return video_data_base64
    
def Sort_Key(file_name):
    # 使用正则表达式找到文件名中的数字部分
    match = re.search(r'\d+', file_name)
    if match:
        return int(match.group(0))
    else:
        return 0