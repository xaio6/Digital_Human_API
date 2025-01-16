from rembg import remove, new_session
import cv2
import os
import shutil
import subprocess
from moviepy.video.io.VideoFileClip import VideoFileClip



class Ppt_2_Video():

    def __init__(self,output_frames, input_frames, Mov_Video):

        self.outputFolder = output_frames
        self.inputFolder = input_frames
        self.outputVideo = Mov_Video

        self.time = 0
        self.fps = 25
 
         
    def Set_Ppt_Transtion_Speed(self,SlideIndex, transition_time):

        # 获取指定索引的幻灯片对象
        slide = self.presentation.Slides(SlideIndex)

        # 获取幻灯片的切换设置
        slide_show_transition = slide.SlideShowTransition

        # 设置切换时间
        slide_show_transition.AdvanceOnTime = True
        slide_show_transition.AdvanceTime = transition_time


    def Ppt_Add_Replace(self , SlideIndex , VideoPpath):
   
 
        # 获取指定索引的幻灯片对象
        slide = self.presentation.Slides(SlideIndex)

        # 获取幻灯片的切换设置
        slide_show_transition = slide.SlideShowTransition

    
        left = self.presentation.PageSetup.SlideWidth - 100  # 调整左侧距离以适应您的幻灯片布局
        top = self.presentation.PageSetup.SlideHeight - 100  # 调整顶部距离以适应您的幻灯片布局
        video = slide.Shapes.AddMediaObject2(FileName=VideoPpath, LinkToFile=False,   Left=left, Top=top, Width=10, Height=10)

        # 设置视频播放选项
        video.AnimationSettings.PlaySettings.PlayOnEntry = True
        video.AnimationSettings.PlaySettings.LoopUntilStopped = True


    def Create_Replace_Video(self ,end_time):
 
        start_time = 0  # 剪辑起始时间（秒）
        video_clip = VideoFileClip(self.m_ReplaceMainPath)
        trimmed_video = video_clip.subclip(start_time, end_time)
        trimmed_video.write_videofile(self.m_ReplacePath, codec="libx264", fps=24)  # 可以根据需要更改编解码器和帧率


    def Create_Ppt_Base_Video(self , OutPptMainBaseVideoPath):
 
        video_output_path = os.path.abspath(OutPptMainBaseVideoPath)
        
        # 导出为视频
        self.presentation.CreateVideo(video_output_path)


        while(self.presentation.CreateVideoStatus == 1):
            print(self.presentation.CreateVideoStatus )
        

    def Get_Video_Time(self , video_path):
        # 使用OpenCV打开视频文件
        video = cv2.VideoCapture(video_path)
        
        # 获取视频的帧数和帧率
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
        
        # 计算视频时长（单位：秒）
        Time = frame_count / 25
        
        return Time
    
    def Insert_Video(self, MainVideo, AuxiliaryVideo, OutPutVideoName, InsertionTime, Time):
        # 获取主视频的高度
        ffprobe_command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=height",
            "-of", "csv=s=x:p=0",
            MainVideo
        ]
        process = subprocess.Popen(ffprobe_command, stdout=subprocess.PIPE)
        out, _ = process.communicate()
        main_height = int(out.decode())

        # 计算辅助视频的缩放尺寸
        # auxiliary_height = main_height // 3 * 2
        auxiliary_height = main_height // 5 * 4
        auxiliary_width = -2  # 使用-2表示宽度按比例缩放

        ffmpeg_command = [
            "ffmpeg",
            "-i", MainVideo,
            "-itsoffset", str(InsertionTime),  # 将 itsoffset 放在辅助视频的输入之前
            "-i", AuxiliaryVideo,
            "-filter_complex", f"[1:v]scale={auxiliary_width}:{auxiliary_height},setsar=1[scaled];[0:v][scaled]overlay=W-w-10:H-h:enable='between(t,{InsertionTime},{InsertionTime}+{Time})'",
            "-c:v", "libx264", 
            "-preset", "slow", 
            "-crf", "0",  # 调整 CRF 参数为更低的值
            "-c:a", "copy",  # 复制音频流以保持原始音频
            OutPutVideoName
        ]

        # 调用FFmpeg命令
        subprocess.run(ffmpeg_command)
    
    # def Insert_Video(MainVideo, AuxiliaryVideo, OutPutVideoName, InsertionTime, Time):
    #     # 获取主视频的高度
    #     ffprobe_command = [
    #         "ffprobe",
    #         "-v", "error",
    #         "-select_streams", "v:0",
    #         "-show_entries", "stream=height",
    #         "-of", "csv=s=x:p=0",
    #         MainVideo
    #     ]
    #     process = subprocess.Popen(ffprobe_command, stdout=subprocess.PIPE)
    #     out, _ = process.communicate()
    #     main_height = int(out.decode())

    #     # 计算辅助视频的缩放尺寸
    #     auxiliary_height = main_height // 3 * 2
    #     auxiliary_width = -2  # 使用-2表示宽度按比例缩放

    #     # 将持续时间转换为整数
    #     # duration_int = float(Time)

    #     ffmpeg_command = [
    #         "ffmpeg",
    #         "-i", MainVideo,
    #         "-itsoffset", str(InsertionTime),  # 将 itsoffset 放在辅助视频的输入之前
    #         "-i", AuxiliaryVideo,
    #         "-filter_complex", f"[1:v]scale={auxiliary_width}:{auxiliary_height},setsar=1[scaled];[0:v][scaled]overlay=W-w-10:H-h-10:enable='between(t,{InsertionTime},{InsertionTime}+{Time})'",
    #         "-c:v", "libx264",  # 指定视频编码器
    #         "-crf", "1",  # 设置质量参数，数值越低，质量越好
    #         "-preset", "slow",  # 设置编码速度与质量平衡
    #         "-c:a", "copy",  # 复制音频流以保持原始音频
    #         OutPutVideoName
    #     ]

    #     # 调用FFmpeg命令
    #     subprocess.run(ffmpeg_command)

    def Remove_Background(self):
        session = new_session()
        
        if os.path.exists(self.inputFolder):
            shutil.rmtree(self.inputFolder)
            os.makedirs(self.inputFolder)

        # 处理每张图像
        for file in os.listdir(self.outputFolder):
            input_path = os.path.join(self.outputFolder, file) 
            output_path = os.path.join(self.inputFolder, file)  # Output path should be in the input_frames folder

            with open(input_path, 'rb') as i:
                input_image = i.read()
                output_image = remove(input_image, session=session)

                #将处理后的图像写入输出文件
                with open(output_path, 'wb') as o:
                    o.write(output_image)


    def Create_Video(self, key):
        Mov_Video = os.path.join(self.outputVideo, f"{key}_Mov.mov")
        # if os.path.exists(Mov_Video):
        #     os.remove(Mov_Video)
            
        ffmpeg_command = [
            "ffmpeg",
            "-framerate", str(self.fps),
            "-i", f"{self.inputFolder}/frame_%05d.png",
            "-c:v", "qtrle",
            Mov_Video
        ]

        # 调用 FFmpeg 命令
        subprocess.run(ffmpeg_command)


    def Video_To_Frames(self,video_path):
 
        if os.path.exists(self.outputFolder):
            shutil.rmtree(self.outputFolder)
            os.makedirs(self.outputFolder)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 获取视频帧数
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 获取视频帧率
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        # 获取视频帧尺寸
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 输出视频信息
        print("Frame count:", frame_count)
        print("FPS:", self.fps)
        print("Frame size:", (frame_width, frame_height))
        
        aspect_ratio_width = 9
        aspect_ratio_height = 16

        # 计算截取区域的宽度，保持高度不变
        crop_width = int(frame_height * (aspect_ratio_width / aspect_ratio_height))
        # crop_height = frame_height

        # 确保截取区域的宽度不超过帧宽度
        if crop_width > frame_width:
            print("Error: Calculated crop width is larger than the frame width.")
            exit()

        # 计算截取区域的起始和结束坐标
        start_x = (frame_width - crop_width) // 2
        end_x = start_x + crop_width
        # start_y = 0
        # end_y = frame_height

        # 逐帧读取视频，并保存成图片
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:, start_x:end_x]
            
            # 保存图片
            frame_filename = os.path.join(self.outputFolder, f"frame_{frame_number:05d}.png")
            cv2.imwrite(frame_filename, frame)

            frame_number += 1

            # 显示进度
            print(f"Processed frame {frame_number}/{frame_count}")

        # 释放视频对象
        cap.release()

        print("Frames extracted successfully!")
    



if __name__ == '__main__':


 

    Ppt = Ppt_2_Video()
    # TimeList = []
    # path = r"VideoSavePath.json"
    # with open(path, "r", encoding="utf-8") as f:
    #     data = json.load(f)
    #     print(data)

    # KeysList = list(data.keys())
    # Ppt.Open_Ppt(r"C:\Users\lin\Desktop\HALCON.pptx")

    # for i in KeysList:
    #     print(i)
    #     TimeList.append(Ppt.GetVideoTime(data[i]))
    #     print(data[i])
    
    
    # Ppt.Video_To_Frames(data[i])
    # Ppt.outputVideo = "OutPutVideo" + "//"+f"{i}.mov"
    # Ppt.Remove_Background()
    # Ppt.Create_Video()
    
    # Ppt.SetPptTranstionSpeed(int(i),Ppt.GetVideoTime(data[i]))


    # print(TimeList)
    # Ppt.Create_Ppt_Base_Video(r"C:\Users\lin\Desktop\2.mp4")


    # for i in TimeList:
    #    Ppt.time +=  TimeList


        




    



