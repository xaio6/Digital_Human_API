import sys
sys.path.append("SadTalker")
from SadTalker.Inference import SadTalker_Model

sys.path.append("VITS")
sys.path.append("VITS/GPT_SoVITS")
from VITS.Inference import GPT_SoVITS_Model
from VITS.train import GPT_SoVITS_Tarin

from util.PPT2Video import Ppt_2_Video
from util.Function import Clear_File, Change_image_Size, Sort_Key
from util.WavJoin import Add_Wav_Processor

import json
import os
import shutil
import yaml
import wave
#####################################################################################
#                           需要线程池完成的任务功能                               #
#####################################################################################

#推理VIDS跟Sadtalker
def Audio_Video_Inference(result_vits_user_path, result_sadtalker_user_path, save_user_path):
    DH = Digital_Human_PPT(result_vits_user_path, result_sadtalker_user_path, save_user_path)
    sad_parames_yaml_path, vits_parames_yaml_path = Get_Parmes(save_user_path)
    DH.Set_Params_and_Model(sad_parames_yaml_path, vits_parames_yaml_path)
    
    
    ref_wav_path = os.path.join(save_user_path,"Ref_Wav.wav")
    with open(os.path.join(save_user_path,"Ref_text.json"), "r", encoding='utf-8') as f:
        data = json.load(f)
    ref_text = data["Text"]
    print(ref_wav_path)
    print(ref_text)
        
    DH.Inference_VITS(ref_wav_path,ref_text)
    
    imag_path = os.path.join(save_user_path,"Image.png")
    DH.Inference_SadTalker(imag_path)
    Task_State(save_user_path, "Audio_Video_Inference", True)
    
    
#生成ppt融合视频
def Video_Merge(save_user_path):
    Remove_Video_Background(save_user_path)
    video_path = Video_Joint(save_user_path)
    Video_Join_Audeo(save_user_path, video_path)
    
    Task_State(save_user_path, "Video_Merge", True)
    
#训练VITS
def Train_VITS(save_user_path, user, json_data):
        VT = VITS_Train(user)
        list_file_path, audio_data_path = VT.Create_Audio_Label(save_user_path,json_data)
        Clear_File(os.path.join(save_user_path,"Weight"))
        Clear_File(os.path.join("VITS/logs",user))
        #格式化
        VT.Format_Data(list_file_path, audio_data_path)

        #训练SoVITS
        VT.Train_SoVITS(save_user_path,10,10)

        #训练GPT
        VT.Train_GPT(save_user_path,18,18)

        Task_State(save_user_path, "VITS_Train", True)

#####################################################################################
#                                一些功能                                            #
#####################################################################################


#创建user_path文件夹
def Create_File(user_name):
    
    user_result_vits_path = os.path.join("Result/VITS", f"{user_name}")
    user_result_sadtalker_path = os.path.join("Result/SadTalker", f"{user_name}")
    user_data_save_path = os.path.join("Data", f"{user_name}")
    
    if not os.path.exists(user_result_vits_path):
        os.makedirs(user_result_vits_path)
        
    if not os.path.exists(user_result_sadtalker_path):
        os.makedirs(user_result_sadtalker_path)


    if not os.path.exists(user_data_save_path):
        os.makedirs(user_data_save_path)
        os.makedirs(os.path.join(user_data_save_path,"output_frames"))
        os.makedirs(os.path.join(user_data_save_path,"input_frames"))
        os.makedirs(os.path.join(user_data_save_path,"Mov_Video"))
        os.makedirs(os.path.join(user_data_save_path,"PPT_Video"))
        os.makedirs(os.path.join(user_data_save_path,"Audio_Data"))
        os.makedirs(os.path.join(user_data_save_path,"Weight"))
        
        #创建空的json文件
        audio_filename = os.path.join(user_data_save_path,"Audio_save_path.json")
        with open(audio_filename, "w") as json_file:
            json.dump({}, json_file)
            
        video_filename = os.path.join(user_data_save_path,"Video_save_path.json")
        with open(video_filename, "w") as json_file:
            json.dump({}, json_file)
        
        ppt_remake_filename = os.path.join(user_data_save_path,"PPT_Remake.json")
        with open(ppt_remake_filename, "w") as json_file:
            json.dump({}, json_file)
            
        ref_wav_text = os.path.join(user_data_save_path,"Ref_text.json")
        with open(ref_wav_text, "w") as json_file:
            json.dump({}, json_file)
            
        wav_time = os.path.join(user_data_save_path,"Time.json")
        with open(wav_time, "w") as json_file:
            json.dump({}, json_file)
            
        state = os.path.join(user_data_save_path,"State.json")
        with open(state, "w") as json_file:
            json.dump({}, json_file)
        
    
    return user_result_vits_path, user_result_sadtalker_path, user_data_save_path

#设置跟拉取任务状态
def Task_State(user_data_save_path, task, methods=None):
    
    json_file_path = os.path.join(user_data_save_path, "State.json")
    # 读取目标文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        target_data = yaml.safe_load(f)
    if methods != None:
        
        target_data[str(task)] = methods
        
        with open(json_file_path, "w", encoding='utf-8') as json_file:
            json.dump(target_data, json_file, ensure_ascii=False)

    return target_data[str(task)]
    
# 配置SadTalker参数
def Config_SadTalker_Parmes(user_data_save_path, json_dict):
    target_file_path = os.path.join(user_data_save_path, "SadTalker_config.yaml")
    # 检查文件是否存在
    if not os.path.exists(target_file_path):
            shutil.copy("SadTalker/SadTalker_config.yaml", user_data_save_path)

    # 读取目标文件
    with open(target_file_path, 'r', encoding='utf-8') as f:
        target_data = yaml.safe_load(f)

    # 替换值
    for key, value in json_dict.items():
        if key in target_data:
            target_data[key] = value

    # 写入目标文件
    with open(target_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(target_data, f)

    return target_file_path

# 配置VITS参数
def Config_VITS_Parmes(user_data_save_path, json_dict):
    target_file_path = os.path.join(user_data_save_path, "GPT-SoVITS_config.yaml")
    
    # 检查文件是否存在
    if not os.path.exists(target_file_path):
        shutil.copy("VITS/GPT-SoVITS_config.yaml", user_data_save_path)
    
    # 读取目标文件
    with open(target_file_path, 'r', encoding='utf-8') as f:
        target_data = yaml.safe_load(f)

    # 替换值
    for key, value in json_dict.items():
        if key in target_data:
            target_data[key] = value

    # 写入目标文件
    with open(target_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(target_data, f)
    
    return target_file_path

#获取参数
def Get_Parmes(user_data_save_path):
    vits = os.path.join(user_data_save_path, "GPT-SoVITS_config.yaml")
    sadtdlker = os.path.join(user_data_save_path, "SadTalker_config.yaml")
    
    # 检查文件是否存在
    if not os.path.exists(vits):
        shutil.copy("VITS/GPT-SoVITS_config.yaml", user_data_save_path)
    if not os.path.exists(sadtdlker):
            shutil.copy("SadTalker/SadTalker_config.yaml", user_data_save_path)
    
    return sadtdlker, vits
    
# 保存PPT备注
def Save_PPT_Remake(user_data_save_path, ppt_remakes):
    ppt_remake_filename = os.path.join(user_data_save_path, "PPT_Remake.json")
    
    with open(ppt_remake_filename, "w", encoding='utf-8') as json_file:
        json.dump(ppt_remakes, json_file, ensure_ascii=False)
         
    return ppt_remake_filename
 
#保存真人照片
def Save_Image(user_data_save_path, image_path):
    img_name = os.path.join(user_data_save_path, "Image.png")
    if os.path.exists(img_name):
        os.remove(img_name)

    #移动
    shutil.move(image_path, user_data_save_path)
    #修改保存在save_user_path照片的名字
    os.rename(os.path.join(user_data_save_path, image_path.split("/")[-1]), img_name)
    Change_image_Size(img_name)

#保存接收的视频
def Save_Video(user_data_save_path, video_at_path):
    video_name = os.path.join(user_data_save_path, "PPT_Video.mp4")
    
    if os.path.exists(video_name):
        os.remove(video_name)
    #移动
    shutil.move(video_at_path, user_data_save_path)
    os.rename(os.path.join(user_data_save_path, video_at_path.split("/")[-1]), video_name)

#保存音频时长
def Save_Tiem(user_data_save_path, result_vits_user_path):
    wav_tiem = os.path.join(user_data_save_path,"Time.json")
    time_dict = {}

    file_dir = os.listdir(result_vits_user_path)
    file_list = sorted(file_dir, key=Sort_Key)
    
    if (len(file_list) > 0):
        #获取音频文件时长
        for _, file in enumerate(file_list):
            #获取文件名
            file_name = file.split(".")[0]
            file_path = os.path.join(result_vits_user_path, file)
            with wave.open(file_path, 'rb') as wav_file:
                # 获取帧数
                frames = wav_file.getnframes()
                # 获取帧速率（每秒的帧数）
                frame_rate = wav_file.getframerate()
                # 计算时长（以秒为单位）
                duration = frames / float(frame_rate)
            time_dict[str(file_name)] = duration
            
        with open(wav_tiem, 'w', encoding='utf-8') as f:
            json.dump(time_dict, f)
        return time_dict
    return None
#保存用于训练的音频
def Save_Audio(user_data_save_path, name, audio_data):
    audio_data_path = os.path.join(user_data_save_path, "Audio_Data")
    audio = os.path.join(audio_data_path, name)
    
    if os.path.exists(audio):
        os.remove(audio)
        
    #保存到文件
    with open(audio, "wb") as img_file:
        img_file.write(audio_data)
        
    
#####################################################################################
#                                VITS功能                                           #
#####################################################################################

#训练VITS模型
class VITS_Train():
    def __init__(self, user):
        self.GST = GPT_SoVITS_Tarin(user)
        
    #对音频标注保存
    def Create_Audio_Label(self, user_data_save_path, data_json):
        audio_data_path = os.path.join(user_data_save_path, "Audio_Data")
        
        list_file_path = os.path.join(user_data_save_path, "Train.list")
        with open(list_file_path, "w", encoding='utf-8') as text_file:
            for key, value in data_json.items():
                name = os.path.join(audio_data_path,key)
                text_file.write(f"{name}|split|ZH|{value}\n")
        return list_file_path, audio_data_path

    #数据格式化
    def Format_Data(self, train_list, train_audio_path):
        if(self.GST.Format_Data(train_list,train_audio_path)):
            return True
        return False


    #训练VITS模型
    def Train_SoVITS(self, user_data_save_path, total_epoch, save_every_epoch):
        model_path = self.GST.Train_SoVITS(total_epoch, save_every_epoch)
        
        if(model_path != None):
            path = os.path.join(user_data_save_path, "Weight")
            shutil.move(model_path, path)
            return True
        return False

    #训练GPT模型
    def Train_GPT(self, user_data_save_path, total_epoch, save_every_epoch):

        model_path = self.GST.Train_GPT(total_epoch, save_every_epoch)
        
        if(model_path != None):
            path = os.path.join(user_data_save_path, "Weight")
            shutil.move(model_path, path)
            return True
        return False


#保存VITS的参照音频跟文字
def Save_VITS_Ref_Wav_And_Text(user_data_save_path, wav_path, ref_text, methods="move"):
    wav_name = os.path.join(user_data_save_path, "Ref_Wav.wav")
    if os.path.exists(wav_name):
        os.remove(wav_name)
    #移动或者copy
    if(methods == "copy"):
        shutil.copy(wav_path, user_data_save_path)
    elif(methods == "move"):
        shutil.move(wav_path, user_data_save_path)
        
    os.rename(os.path.join(user_data_save_path, wav_path.split("/")[-1]), wav_name)
    
    ref_wav_text = os.path.join(user_data_save_path,"Ref_text.json")
    
    with open(ref_wav_text, "w", encoding='utf-8') as json_file:
        json.dump(ref_text, json_file, ensure_ascii=False)
        
    return ref_wav_text


#//////////////////////////////// VITS模型二选一(改模型路径) ///////////////////////////////////////////////////////////////////////////////////////////////////
 
# 选择自定义的VITS模型
def Select_Train_VITS_Model(user_data_save_path, user):
    pth = os.path.join(user_data_save_path, "Weight", f"{user}.pth")
    ckpt = os.path.join(user_data_save_path, "Weight", f"{user}.ckpt")

    model_dict = {
        "GPT_model_path" : ckpt,
        "SoVITS_model_path" : pth
    }
    
    Config_VITS_Parmes(user_data_save_path, model_dict)
 
    
    
# 选择预训练的VITS模型 （把ref_wav和ref_test复制到保存路径，然后再改掉GPT-SoVITS_config.yaml模型位置，改掉参考文字）
def Select_VITS_Model(user_data_save_path, index):
    target_file_path = os.path.join(user_data_save_path, "GPT-SoVITS_config.yaml")
    
    # 检查文件是否存在
    if not os.path.exists(target_file_path):
        shutil.copy("VITS/GPT-SoVITS_config.yaml", user_data_save_path)
    
    with open("VITS/Model.json",'r', encoding='utf-8') as f:
        model_json = json.load(f)
    
    model = model_json[str(index)]
    Config_VITS_Parmes(user_data_save_path, model)
 
        
    Save_VITS_Ref_Wav_And_Text(user_data_save_path, model["Ref_Wav"], {"Text" : model["Ref_Text"]}, "copy")
        
    return target_file_path
    
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#####################################################################################
#                                视频拼接                                            #
#####################################################################################
     
#把sadtalker的每一个视频去背景
def Remove_Video_Background(user_data_save_path):
    output_frames = os.path.join(user_data_save_path, "output_frames")
    input_frames = os.path.join(user_data_save_path, "input_frames")
    mov_video = os.path.join(user_data_save_path, "Mov_Video")
    
    
    Clear_File(output_frames)
    Clear_File(input_frames)
    Clear_File(mov_video)
    
    PV = Ppt_2_Video(output_frames, input_frames, mov_video)
    
    with open(os.path.join(user_data_save_path, "Video_save_path.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for key, video_file_path in enumerate(data.values()):
        PV.Video_To_Frames(video_file_path)
        PV.Remove_Background()
        PV.Create_Video(key)
    

#sadtalker视频跟PPT合成视频
def Video_Joint(user_data_save_path):
    
    output_frames = os.path.join(user_data_save_path, "output_frames")
    input_frames = os.path.join(user_data_save_path, "input_frames")
    Mov_Video = os.path.join(user_data_save_path, "Mov_Video")
    ppt_video_path = os.path.join(user_data_save_path, "PPT_Video.mp4")
    output_ppt_video = os.path.join(user_data_save_path, "PPT_Video")
    
    Clear_File(output_ppt_video)
    
    PV = Ppt_2_Video(output_frames, input_frames, Mov_Video)
    transition_time = 0
    
    with open(os.path.join(user_data_save_path, "Time.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # keys = list(data.keys())
    time_list = list(data.values())
        
    for i in range(len(time_list)):

        # mov_path = os.path.join(Mov_Video, mov_list[i])
        mov_path = os.path.join(Mov_Video,  f"{i}" + "_Mov.mov")
        
        if i == 0:
            one_ppt_video_path = os.path.join(output_ppt_video, f"{i}_ppt_video.mp4")
            
            transition_time += 3
            PV.Insert_Video(ppt_video_path, mov_path, one_ppt_video_path, transition_time, time_list[i])
        else:
            one_ppt_video_path = os.path.join(output_ppt_video, f"{i - 1}_ppt_video.mp4")
            two_ppt_video_path = os.path.join(output_ppt_video, f"{i}_ppt_video.mp4")
            
            transition_time += 4
            transition_time += time_list[i - 1]
            PV.Insert_Video(one_ppt_video_path, mov_path, two_ppt_video_path, transition_time, time_list[i])
            
    return two_ppt_video_path
    

#给PPT视频拼接音频
def Video_Join_Audeo(user_data_save_path, input_video):
    AWP = Add_Wav_Processor()
    
    audio_save_path = os.path.join(user_data_save_path, "Audio_save_path.json")
    last_video = os.path.join(user_data_save_path, "last_video.mp4")
    join_audio = os.path.join(user_data_save_path, "Join_Audio.wav")
    
    with open(audio_save_path, 'r', encoding='utf-8') as f:
        path_dict = json.load(f)
        
    for i in path_dict.keys():
        if i == "0":
            print(path_dict[str(i)])
            result_audio = AWP.Add_Silence_At_Beginning(path_dict[str(i)])
            result_audio.export(join_audio, format="wav")
        else:
            print(path_dict[str(i)])
            result_audio = AWP.Add_Silence_Between_Tracks(join_audio,path_dict[str(i)])
            result_audio.export(join_audio, format="wav")
            print(i)

    #  当不是第一个视频的时候就执行add_silence_between_tracks和上一个wav结合空出几秒自己选择
    #  生成完成add_audio_to_video贴入MP4
    AWP.Add_Audio_To_Video(input_video, join_audio, last_video)
    
    return last_video
  
  
#####################################################################################
#                                模型推理                                            #
#####################################################################################
  
    
class Digital_Human_PPT():
    def __init__(self,vits,sadtalker,save):
        
        #声明模型
        self.VITS = None
        self.Sad = None
        
        
        #实例化
        self.VITS = GPT_SoVITS_Model()
        self.Sad = SadTalker_Model()
        
        #全局变量
        self.user_result_vits_path = vits
        self.user_result_sadtalker_path = sadtalker
        self.user_data_save_path = save
        
        
        #清空文件
        Clear_File(self.user_result_vits_path)
        Clear_File(self.user_result_sadtalker_path)
        
           
    #预加载参数和模型
    def Set_Params_and_Model(self,sad_parames_yaml_path=None,vits_parames_yaml_path=None):
        self.VITS.Initialize_Parames(vits_parames_yaml_path)
        
        self.Sad.Initialize_Parames(sad_parames_yaml_path)
        self.Sad.Initialize_Models()
    

    #根据备注推理音频
    def Inference_VITS(self, ref_wav_path, ref_text): 
        path = os.path.join(self.user_data_save_path,"PPT_Remake.json")
        #读取Json_Data/PPT_Remake.json里面的键值对
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        #把键跟值分别保存list
        ip_keys = list(data.keys())
        PPT_Remake_values = list(data.values())
        
        dict = {}

        for i in range(len(ip_keys)):
            
            text = PPT_Remake_values[i]
            output_path = os.path.join(self.user_result_vits_path, f"{ip_keys[i]}.wav") 
            
            self.VITS.Initialize_Models()
            self.VITS.Perform_Inference(
                ref_wav_path = ref_wav_path,
                prompt_text = ref_text,
                prompt_languageself = self.VITS.i18n("中文"),
                target_text = text,
                target_text_language = self.VITS.i18n("中文"),
                cut = self.VITS.i18n("凑50字一切"),
                output_path=output_path
            )
            
            dict[ip_keys[i]] = output_path

        # 指定要写入的JSON文件路径
        json_file_path =os.path.join(self.user_data_save_path,"Audio_save_path.json")

        # 使用with语句打开文件，确保在写入完成后自动关闭文件
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            # 使用json.dump()函数将数据写入JSON文件
            json.dump(dict, json_file)
            
        Save_Tiem(self.user_data_save_path, self.user_result_vits_path)


    #生成数字人视频
    def Inference_SadTalker(self,image): 
        path = os.path.join(self.user_data_save_path,"Audio_save_path.json")
        #读取Json_Data/Audio_save_path.json里面的键值对
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        #把键跟值分别保存list
        ip_keys = list(data.keys())
        audio_path_values = list(data.values())
    
        dict = {}
        
        for i in range(len(ip_keys)):
            save_path = os.path.join(self.user_result_sadtalker_path,ip_keys[i])
            
            self.Sad.Perform_Inference(image, audio_path_values[i],save_path)
            
            dict[ip_keys[i]] = save_path + ".mp4"
            
        # 指定要写入的JSON文件路径
        json_file_path =os.path.join(self.user_data_save_path,"Video_save_path.json")

        # 使用with语句打开文件，确保在写入完成后自动关闭文件
        with open(json_file_path, 'w',encoding='utf-8') as json_file:
            # 使用json.dump()函数将数据写入JSON文件
            json.dump(dict, json_file)

    #根据备注推理音频_test
    def Inference_VITS_test(self, ref_wav_path, ref_text, text):
        output_path = os.path.join(self.user_data_save_path, "Test_VITS.wav") 
        
        self.VITS.Initialize_Models()
        self.VITS.Perform_Inference(
            ref_wav_path = ref_wav_path,
            prompt_text = ref_text,
            prompt_languageself = self.VITS.i18n("中文"),
            target_text = text,
            target_text_language = self.VITS.i18n("中文"),
            cut = self.VITS.i18n("凑50字一切"),
            output_path=output_path
        )
        return output_path
    
    
    #生成数字人视频_test
    def Inference_SadTalker_test(self, image, audio_path, save_path):
        self.Sad.Perform_Inference(image, audio_path, save_path)

if __name__ == '__main__':

    # result_vits_user_path, result_sadtalker_user_path, user_data_save_path = Create_File("Hui")
    
    # di = {
    #     "1.wav":"首先你要准备一个说话人的音频的数据集。",
    #     "2.wav":"然后根据这个数据集，让AI去训练模型。"
    # }
    
    # VT = VITS_Train("Hui")
    # list_file_path, audio_data_path = VT.Create_Audio_Label(user_data_save_path,di)
    # VT.Format_Data(list_file_path, audio_data_path)
    # VT.Train_SoVITS(user_data_save_path,8,8)
    # VT.Train_GPT(user_data_save_path,15,15)
    
    # Remove_Video_Background(user_data_save_path)
    # Video_Joint(user_data_save_path)
    #Video_Join_Audeo(user_data_save_path, "Data/Hui/PPT_Video/4_ppt_video.mp4")
    # Select_VITS_Model(save_user_path,"0")
    
    # # Save_VITS_Wav(save_user_path,"F:/Learning/Digital_Human/7.WAV")
    # # Save_PPT_Remake(save_user_path,{"a":1,"b":2})
    # # Save_Image()
    # sadtalker_config_path = Config_SadTalker_Parmes(save_user_path,{"expression_scale" : 1.5})
    # vits_congfig_path = Config_VITS_Parmes(save_user_path, {"top_k" : 6})
    
    # DH = Digital_Human_PPT(result_vits_user_path,
    #                        result_sadtalker_user_path,
    #                        save_user_path)
    
    # DH.Set_Params_and_Model(sad_parames_yaml_path = sadtalker_config_path, 
    #                         vits_parames_yaml_path = vits_congfig_path)
    
    # ref_wav_path = os.path.join(save_user_path,"WAV.wav")
    # # #推理VITS
    # DH.Inference_VITS(ref_wav_path,"不是啊，过年前也要搞呀，只不过过年前我妈妈他们搞")

    # DH.Inference_SadTalker(os.path.join(save_user_path,"image.png"))

    # Save_Tiem(save_user_path,result_vits_user_path)

    print("完成")
        
        
