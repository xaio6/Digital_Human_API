"""
    添加可选择对指定页数插入音频
    修复了：第二次上传自定义音频没有清空上一次的音频
            数字人拼接没有贴边
            数字人太小
    增加了初始化文件夹（删除文件）
"""

import base64
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from Main import *
from util.Function import Verification, Encode_Video, Write_Json

import concurrent.futures

# 创建线程池
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

#登录
@app.route('/Login', methods=['POST'])
def Login():
    POST_JSON = request.get_json()
    user_name = POST_JSON.get("User")
    user_password = POST_JSON.get("Password")
    
    try:
        if(Verification(str(user_name), str(user_password))):
            user_result_vits_path, user_result_sadtalker_path, user_result_wav2lip_path, user_data_save_path = Create_File(str(user_name))
            Init_File(user_result_vits_path, user_result_sadtalker_path, user_result_wav2lip_path, user_data_save_path)
            return jsonify(result="Success")
        else:
            return jsonify(result="Failed")
        
    except:
        return jsonify(result="Failed")
    
#获取状态
@app.route('/Get_State', methods=['POST'])
def Get_State():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    task = POST_JSON.get('Task')
    
    try:
        _, _, _, save_user_path = Create_File(str(user))
        task_state = Task_State(save_user_path,str(task))
        return jsonify(result=task_state)
    except:
        return jsonify(result="Failed")
    
# 保存PPT备注信息
@app.route('/Send_PPT_Remakes', methods=['POST'])
def Set_PPT_Remakes():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    ppt_remakes = POST_JSON.get("PPT_Remakes")

    try:
        _, _, _, save_user_path = Create_File(str(user))
        ppt_remake_filename = Save_PPT_Remake(save_user_path,ppt_remakes)
        
        with open(ppt_remake_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if(data != {}):
            return jsonify(result="Success")
        else:
            return jsonify(result="Failed")
    except:
        return jsonify(result="Failed")
    
#保存真人照片
@app.route('/Send_Image', methods=['POST'])
def Send_Image():
    # POST_JSON = request.get_json()
    # user = POST_JSON.get('User')
    # img = request.files.get('Image')  # 从post请求中获取图片数据
    
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    img_data_base64 = POST_JSON.get('Img')
    
    try:
        _, _, _, save_user_path = Create_File(str(user))
        
        # 解码base64字符串并保存到文件
        img_data = base64.b64decode(img_data_base64)
        with open("img.png", "wb") as img_file:
            img_file.write(img_data)
        Save_Image(save_user_path,"img.png")
        
        return jsonify(result="Success")
    except:
        return jsonify(result="Failed")

#保存教师视频
@app.route('/Send_Teacher_Video', methods=['POST'])
def Send_Teacher_Video():
    string = request.form.get('Json')
    video = request.files['File'].read()
    try:
        json_data = json.loads(string)
        user = json_data.get('User')
        _, _, _, save_user_path = Create_File(str(user))
        
        # 写入文件
        with open(os.path.join(save_user_path, "Video.mp4"), "wb") as video_file:
            video_file.write(video)
        
        # 处理视频文件的保存逻辑
        # Save_Video(save_user_path, "video.mp4")
        
        return jsonify(result="Success")
    except Exception as e:
        print(e)
        return jsonify(result="Failed")
    
#获取VITS音频时长
@app.route('/Recive_Wav_Time', methods=['POST'])
def Recive_Wav_Time():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')

    try:
        result_vits_user_path, _, _, save_user_path = Create_File(str(user))
        wav_time_dict = Save_Tiem(save_user_path, result_vits_user_path)
        return  jsonify(result = wav_time_dict)
        
    except:
        return jsonify(result="Failed")
 
# 获取用户音频时长
@app.route('/Recive_User_Wav_Time', methods=['POST'])
def Recive_User_Wav_Time():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')

    try:
        _, _, _, save_user_path = Create_File(str(user))
        ppt_audio_dir = os.path.join(save_user_path, "PPT_Audio")
        audio_json_save_path = os.path.join(save_user_path, "Audio_save_path.json")
        user_wav_path = os.path.join(save_user_path, "PPT_Audio")
        
        Write_Json(ppt_audio_dir, audio_json_save_path)
        wav_time_dict = Save_Tiem(save_user_path, user_wav_path)
        return  jsonify(result = wav_time_dict)
        
    except:
        return jsonify(result="Failed")
 
#接收前端视频
@app.route('/Send_Video', methods=['POST'])
def Send_Video():
    string = request.form.get('Json')
    video = request.files['File'].read()
    
    try:
        json_data = json.loads(string)
        user = json_data.get('User')
        _, _, _, save_user_path = Create_File(str(user))
        
        # 写入文件
        with open(os.path.join(save_user_path, "PPT_Video.mp4"), "wb") as video_file:
            video_file.write(video)
        
        # 处理视频文件的保存逻辑
        # Save_Video(save_user_path, "video.mp4")
        
        return jsonify(result="Success")
    except Exception as e:
        print(e)
        return jsonify(result="Failed")

# 保存数字人插入页数的json
@app.route('/Send_People_Location', methods=['POST'])
def Send_People_Location():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    people_location = POST_JSON.get("People_Location")

    try:
        _, _, _, save_user_path = Create_File(str(user))
        people_location_filename = Save_People_Location(save_user_path,people_location)
        
        with open(people_location_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if(data != {}):
            return jsonify(result="Success")
        else:
            return jsonify(result="Failed")
    except:
        return jsonify(result="Failed")   

# 保存用于插入PPT的音频
@app.route('/Send_PPT_Audio', methods=['POST'])
def Send_PPT_Audio():
    string = request.form.get('Json')
    audio = request.files['File'].read()
    
    try:
        json_data = json.loads(string)
        user = json_data.get('User')
        audio_name = json_data.get('Audio_Name')
        
        _, _, _, save_user_path = Create_File(str(user))
        Save_Insert_Audio(save_user_path, audio_name, audio)
        
        return jsonify(result="Success")
    except:
        return jsonify(result="Failed")

#####################################################################################
#                                 配置参数                                           #
#####################################################################################

# 配置所有模型参数
@app.route('/Send_Config', methods=['POST'])
def Send_Config():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    vits_config = POST_JSON.get('VITS_Config')
    sadtalker_config = POST_JSON.get('SadTalker_Config')
    
    try:
        _, _, _, save_user_path = Create_File(str(user))
        
        Config_SadTalker_Parmes(save_user_path, sadtalker_config)
        Config_VITS_Parmes(save_user_path, vits_config)
        
        return jsonify(result="Success")
    except:
        return jsonify(result="Failed")
    
    
#配置wav2lip参数
@app.route('/Send_Wav2Lip_Config', methods=['POST'])
def Send_Wav2Lip_Config():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    wav2lip_config = POST_JSON.get('Wav2Lip_Config')
    
    try:
        _, _, _, save_user_path = Create_File(str(user))
        
        Config_Wav2Lip_Parmes(save_user_path, wav2lip_config)
        
        return jsonify(result="Success")
    except:
        return jsonify(result="Failed")
    
#选择训练的VITS模型
@app.route('/Send_Select_Train_VITS_Model', methods=['POST'])
def Send_Select_Train_VITS_Model():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    
    try:
        _, _, _, save_user_path = Create_File(str(user))
        Select_Train_VITS_Model(save_user_path,user)
        
        return jsonify(result="Success")
    
    except:
        return jsonify(result="Failed")

#选择VITS模型
@app.route('/Send_Select_VITS_Model', methods=['POST'])
def Send_Select_VITS_Model():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    index = POST_JSON.get('Index')
    
    try:
        _, _, _, save_user_path = Create_File(str(user))
        
        Select_VITS_Model(save_user_path,str(index))
        
        return jsonify(result="Success")
    except:
        return jsonify(result="Failed")
   
    
#####################################################################################
#                                VITS功能                                           #
#####################################################################################

#保存用于训练VITS的音频
@app.route('/Send_Tarin_Audio', methods=['POST'])
def Send_Tarin_Audio():
    string = request.form.get('Json')
    audio = request.files['File'].read()
    
    try:
        json_data = json.loads(string)
        user = json_data.get('User')
        audio_name = json_data.get('Audio_Name')
        
        _, _, _, save_user_path = Create_File(str(user))
        Save_Train_Audio(save_user_path, audio_name, audio)
        
        return jsonify(result="Success")
    except:
        return jsonify(result="Failed")

#训练VITS模型
@app.route('/Train_VITS_Model', methods=['POST'])
def Train_VITS_Model():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    json_data = POST_JSON.get('Label')
    
    try:
        _, _, _, save_user_path = Create_File(str(user))
        Task_State(save_user_path, "VITS_Train", False)
        executor.submit(Train_VITS, save_user_path, user, json_data)

        return jsonify(result="VITS_Train")
    except:
        return jsonify(result="Failed")
    
#保存VITS的参照音频跟文字
@app.route('/Send_Ref_Wav_And_Text', methods=['POST'])
def Send_Ref_Wav_And_Text():
    string = request.form.get('Json')
    audio = request.files['File'].read()
    
    try:
        json_data = json.loads(string)
        user = json_data.get('User')
        ref_text = json_data.get('Ref_Text')
        _, _, _, save_user_path = Create_File(user)
        
        file =  save_user_path + "/" + "Audio.mp3"
         # 写入文件
        with open(file, "wb") as audio_file:
            audio_file.write(audio)
            
        Save_VITS_Ref_Wav_And_Text(save_user_path, file,  {"Text" : ref_text}, "None")
            
        return jsonify(result="Success")
    except Exception as e:
        print(e)
        return jsonify(result="Failed")
    
#获取训练后的VITS模型的名字
@app.route('/Get_Train_VITS_Model_Name', methods=['POST'])
def Get_Train_VITS_Model_Name():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    
    try:
        _, _, _, save_user_path = Create_File(str(user))
        weightPath = os.path.join(save_user_path, "Weight")
        #判断文件夹不为空
        if len(os.listdir(weightPath)) > 0:
            return jsonify(result=f"{user}")
        else:
            return jsonify(result="Failed")
        
    except:
        return jsonify(result="Failed")
    
#####################################################################################
#                                      推理                                         #
#####################################################################################
   
#推理效果展示视频
@app.route('/Get_Test_Inference', methods=['POST'])
def Get_Test_Inference():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    try:
        result_vits_user_path, result_sadtalker_user_path, _, save_user_path = Create_File(user)
        DH = VITS_Sadtalker_Join(result_vits_user_path, result_sadtalker_user_path, save_user_path)
        sad_parames_yaml_path, vits_parames_yaml_path = Get_Parmes(save_user_path)
        DH.Set_Params_and_Model(sad_parames_yaml_path, vits_parames_yaml_path)
        
        with open(os.path.join(save_user_path,"Ref_text.json"), "r", encoding='utf-8') as f:
            data = json.load(f)
            
        ref_wav_path = os.path.join(save_user_path,"Ref_Wav.wav")
        ref_text = data["Text"]
        test = "你好，我是数字人授课录制系统，很高兴为您服务。"
        video_output_path = os.path.join(save_user_path,"Test")
        imag_path = os.path.join(save_user_path,"Image.png")
        
        audio_path = DH.Inference_VITS_test(ref_wav_path, ref_text, test)
        DH.Inference_SadTalker_test(imag_path, audio_path, video_output_path)
        
        video_data_base64 = Encode_Video(os.path.join(save_user_path,"Test.mp4"))
        
        return jsonify(result = video_data_base64)
    except:
        return jsonify(result="Failed")
       
#推理VITS(单个)
@app.route('/Get_Inference_VITS', methods=['POST'])
def Get_Inference_VITS():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    try:
        result_vits_user_path, result_sadtalker_user_path, _, save_user_path = Create_File(user)
        DH = VITS_Sadtalker_Join(result_vits_user_path, result_sadtalker_user_path, save_user_path)
        sad_parames_yaml_path, vits_parames_yaml_path = Get_Parmes(save_user_path)
        DH.Set_Params_and_Model(sad_parames_yaml_path, vits_parames_yaml_path)
        
        with open(os.path.join(save_user_path,"Ref_text.json"), "r", encoding='utf-8') as f:
            data = json.load(f)
        
        ref_wav_path = os.path.join(save_user_path,"Ref_Wav.wav")
        ref_text = data["Text"]
        test = "你好，我是数字人授课录制系统，很高兴为您服务。"

        
        DH.Inference_VITS_test(ref_wav_path, ref_text, test)
        
        
        return jsonify(result="Success")
    except:
        return jsonify(result="Failed")
      
#推理VITS(多个)
@app.route('/Get_Inference_VITS_Multiple', methods=['POST'])
def Get_Inference_VITS_Multiple():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    try:
        result_vits_user_path, result_sadtalker_user_path, _, save_user_path = Create_File(user)
        Task_State(save_user_path, "VITS_Inference", False)
        # 将任务提交到线程池 
        # executor.submit(VITS_Multiple_Inference, result_vits_user_path, result_sadtalker_user_path, save_user_path)
        VITS_Multiple_Inference(result_vits_user_path, result_sadtalker_user_path, save_user_path)
        return jsonify(result="VITS_Inference")
    except:
        return jsonify(result="Failed")
      
# 推理VITS跟Sadtalker
@app.route('/Get_Inference_VITS_Sadtalker', methods=['POST'])
def Get_Inference_VITS_Sadtalker():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    try:
        result_vits_user_path, result_sadtalker_user_path, _, save_user_path = Create_File(user)
        Task_State(save_user_path, "Audio_Video_Inference", False)
        # 将任务提交到线程池 
        executor.submit(VITS_Sadtalker_Inference, result_vits_user_path, result_sadtalker_user_path, save_user_path)

        return jsonify(result="Audio_Video_Inference")
    except:
        return jsonify(result="Failed")
    
# 推理用户音频跟Sadtalker
@app.route('/Get_Inference_User_Audio_Sadtalker', methods=['POST'])
def Get_Inference_User_Audio_Sadtalker():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    try:
        _, result_sadtalker_user_path, _, save_user_path = Create_File(user)
        Task_State(save_user_path, "Audio_Video_Inference", False)
        # 将任务提交到线程池 
        executor.submit(User_Wav_Sadtalker_Inference, result_sadtalker_user_path, save_user_path)

        return jsonify(result="Audio_Video_Inference")
    except:
        return jsonify(result="Failed")
    
    
# 推理VITS跟Wav2Lip
@app.route('/Get_Inference_VITS_Wav2Lip', methods=['POST'])
def Get_Inference_VITS_Wav2Lip():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    
    result_vits_user_path, _, result_wav2lip_user_path, save_user_path = Create_File(user)
    Task_State(save_user_path, "Audio_Video_Inference", False)
    VITS_Wav2Lip_Inference(result_vits_user_path, result_wav2lip_user_path, save_user_path)
    # 将任务提交到线程池 
    # executor.submit(VITS_Wav2Lip_Inference, result_vits_user_path, result_wav2lip_user_path, save_user_path)
    
    return jsonify(result="Audio_Video_Inference")

# 推理用户音频跟Wav2Lip
@app.route('/Get_Inference_User_Audio_Wav2Lip', methods=['POST'])
def Get_Inference_User_Audio_Wav2Lip():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    
    _, _, result_wav2lip_user_path, save_user_path = Create_File(user)
    Task_State(save_user_path, "Audio_Video_Inference", False)
    # 将任务提交到线程池 
    executor.submit(User_Wav_Wav2Lip_Inference, result_wav2lip_user_path, save_user_path)
    
    return jsonify(result="Audio_Video_Inference")

# ppt跟视频合成（合成最终效果视频，全插入数字人） 
@app.route('/PPT_Video_Merge', methods=['POST'])
def PPT_Video_Merge():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    try:
        _, _, _, save_user_path = Create_File(user)
        # 将任务提交到线程池 
        Task_State(save_user_path, "Video_Merge", False)
        executor.submit(Video_Merge,save_user_path)
        
        return jsonify(result = "Video_Merge")
    except:
        return jsonify(result="Failed")

# ppt跟视频合成（合成最终效果视频，可选择插入数字人）
@app.route('/PPT_Video_Merge_Select_Into', methods=['POST'])
def PPT_Video_Merge_Select_Into():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    try:
        _, _, _, save_user_path = Create_File(user)
                # 将任务提交到线程池 
        Task_State(save_user_path, "Video_Merge", False)
        executor.submit(Video_Merge_Select_Into,save_user_path)
        
        return jsonify(result = "Video_Merge")
    except:
        return jsonify(result="Failed")
        
# ppt跟视频合成（没有数字人）
@app.route('/PPT_Video_Merge_No_Into', methods=['POST'])
def PPT_Video_Merge_No_Into():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    try:
        _, _, _, save_user_path = Create_File(user)
        Video_Join_Audio(save_user_path)
        
        return jsonify(result = "Success")
    except:
        return jsonify(result="Failed")

#####################################################################################
#                                    获取                                           #
#####################################################################################

# 拉取视频
@app.route('/Pull_Video_Merge', methods=['POST'])
def Pull_Video_merge():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    try:
        _, _, _, save_user_path = Create_File(user)
        last_video = os.path.join(save_user_path,"last_video.mp4")
        video_data_base64 = Encode_Video(last_video)
        return jsonify(result = video_data_base64)
    except:
        return jsonify(result="Failed")
 
 
# 拉取推理的VITS声音
@app.route('/Pull_VITS_Audio', methods=['POST'])
def Pull_VITS_Audio():
    POST_JSON = request.get_json()
    user = POST_JSON.get('User')
    try:
        _, _, _, save_user_path = Create_File(user)
        vits_wav = os.path.join(save_user_path,"Test_VITS.wav")
        wav_data_base64 = Encode_Video(vits_wav)
        return jsonify(result=wav_data_base64)
    except:
        return jsonify(result="Failed")
 
 
    
if __name__ == '__main__':
    
    app.run("0.0.0.0")


