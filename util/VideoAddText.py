from moviepy.audio.fx.all import audio_left_right
from moviepy.audio.AudioClip import AudioArrayClip
import time
from moviepy.editor import VideoFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import os
import subprocess  
  
srt_path = ""
def list_files(path):
    fileMP4List = []
    fileSRTList = []
    count=1
    # 遍历目录下的所有文件和子目录，并输出
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("mp4"):
                fileMP4List.append([count,os.path.join(root, file)])
                count=count+1
            if file.endswith("srt"):
                fileSRTList.append(os.path.join(root, file)[:-4])
    for i in range(len(fileMP4List)):
        if fileMP4List[i][1][:-4] in fileSRTList:
            fileMP4List[i][1]+="    已有srt"
    return fileMP4List


def getWav(videoPath):
    
    print("提取音频")

    clip = VideoFileClip(videoPath)
    audio = clip.audio

    # 将音频转换为单通道
    audio_array = audio.to_soundarray()
    audio_left_right(audio_array)
    # 获取音频剪辑的持续时间
    duration = audio.duration
    # 将单通道音频转换为音频剪辑对象
    audio_mono = AudioArrayClip(audio_array, fps=audio.fps)
    newWavePath=videoPath[:-4]+'.wav'
    # 保存音频为WAV文件
    audio_mono.write_audiofile(newWavePath)
    print("音频生成完成，准备输出srt")
    return newWavePath
def getSrt(newWavePath):

    global srt_path
    print("提取srt中...")
    print("开始加载识别模型")
    import subprocess
    import os
    import sys
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SAMPLE_RATE = 16000
    SetLogLevel(-1)
    # 解压的模型文件，英文，中文用对应model
    getCn = r"vosk-model-small-cn-0.22"
    getCn1=r"F:/Learning/Digital_Human/vosk-model-cn-0.22"
    getJP=r"D:\Mycode\pythonCode\voice_txt\code\model\jp\vosk-model-ja-0.22"
    getEn=r"D:\Mycode\pythonCode\voice_txt\code\model\en\vosk-model-en-us-0.42-gigaspeech"
    model = Model(getCn1)
    print("模型加载完毕,开始识别...")
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    # 修改需要识别的语音文件路径
    wavPath=newWavePath
    rec.SetWords(True)
    result = []
    with subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i",
                                wavPath,
     "-ar", str(SAMPLE_RATE) , "-ac", "1", "-f", "s16le", "-"],
     stdout=subprocess.PIPE).stdout as stream:
        word=rec.SrtResult(stream)
        result.append(word)
        print(word)
    
    print(result)
  
    # 生成srt文件
    srt_path = wavPath[:-4] + '.srt'
    with open(srt_path, 'w', encoding='utf-8') as output:
        output.write("\n".join(result))
 
    output.close()
    print("srt输出完成")
    os.remove(wavPath)
    # 列出当前目录下的所有文件和子目录
    

if __name__ == "__main__":
     
    allFile = r"Data/Hui/mp4.mp4"
     
 
    print("开始识别的文件为:"+allFile)
            
    getSrt(getWav(allFile))
          
    time.sleep(3)
    print("识别完成")
 
 
    output_file = "out.mp4"  
    
    # 构建FFmpeg命令  s
    ffmpeg_cmd = [  
        "ffmpeg",  
        "-i", allFile,  
        "-vf", f"subtitles={srt_path}",  
        output_file  
    ]  
    
    # 执行FFmpeg命令  
    try:  
        subprocess.run(ffmpeg_cmd, check=True)  
        print("FFmpeg 命令执行成功！")  
    except subprocess.CalledProcessError as e:  
        print(f"FFmpeg 命令执行失败: {e}")