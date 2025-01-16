 
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
import json

class Add_Wav_Processor:

    def Add_Silence_Between_Tracks(self,track1, track2, silence_duration=4):
        # 加载音频文件
        audio1 = AudioSegment.from_file(track1)
        audio2 = AudioSegment.from_file(track2)

        silence_duration = silence_duration * 1000
        # 计算无声音频的长度
        silence_audio = AudioSegment.silent(duration=silence_duration)
        
        # 将无声音频插入到两个音频文件之间
        result = audio1 + silence_audio + audio2
        
        return result

    def Add_Silence_At_Beginning(self,audio_file, silence_duration=3):
        # 加载音频文件
        audio = AudioSegment.from_file(audio_file)
        silence_duration = silence_duration * 1000
        # 计算无声音频的长度
        silence_audio = AudioSegment.silent(duration=silence_duration)
        
        # 将无声音频插入到音频文件前面
        result = silence_audio + audio
        
        return result
    
    def Add_Silence_At_Ending(self,audio_file, silence_duration=2):
        # 加载音频文件
        audio = AudioSegment.from_file(audio_file)
        silence_duration = silence_duration * 1000
        
        # 计算无声音频的长度
        silence_audio = AudioSegment.silent(duration=silence_duration)
        
        # 将无声音频插入到音频文件前面
        result = audio + silence_audio 
        
        return result


    def Add_Audio_To_Video(self,video_path, audio_path, output_path):
        # 加载视频和音频文件
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        # 如果音频比视频长，剪辑音频
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)

        # 为视频添加音频
        video_clip = video_clip.set_audio(audio_clip)

        # 保存输出视频文件
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # 关闭视频和音频文件
        video_clip.close()
        audio_clip.close()

    # # 调用函数添加音频到视频
    # add_audio_to_video(r"C:\Users\lin\Desktop\2.mp4", "result.mp3", "output_video.mp4")



if __name__ == '__main__':
    AWP = Add_Wav_Processor()
    #  当是第一个视频的时候就执行add_silence_at_beginning添加两秒的空白给他
    with open("Data/Hui/Audio_save_path.json", 'r', encoding='utf-8') as f:
        LoadJsons = json.load(f)

    for i in LoadJsons.keys():
        if i == "0":
        
            print(LoadJsons[str(i)])
            result_audio = AWP.Add_Silence_At_Beginning(LoadJsons[str(i)])
            result_audio.export("result.mp3", format="mp3")
        else:
            result_audio = AWP.Add_Silence_Between_Tracks("result.mp3",LoadJsons[str(i)])
            result_audio.export("result.mp3", format="mp3")
            print(i)

    #  当不是第一个视频的时候就执行add_silence_between_tracks和上一个wav结合空出几秒自己选择
    #  生成完成add_audio_to_video贴入MP4
    AWP.Add_Audio_To_Video("Video.mp4", "result.mp3", "output_video.mp4")