from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
from tools.i18n.i18n import I18nAuto
from argparse import ArgumentParser
import soundfile as sf
import yaml

class GPT_SoVITS_Model():
    def __init__(self):
        self.i18n = I18nAuto()
        
        #全局变量,
        self.args = None
      
    def Initialize_Parames(self,params=None):
        self.parser = ArgumentParser()
        if params == None:    
            # 获取参数
            with open('VITS/GPT-SoVITS_config.yaml', 'r') as stream:
                config = yaml.safe_load(stream)
            for key, value in config.items():
                self.parser.add_argument(f"--{key}", default=value)
            self.args = self.parser.parse_args()
        else:
            # 获取参数
            with open(params, 'r') as stream:
                config = yaml.safe_load(stream)
            for key, value in config.items():
                self.parser.add_argument(f"--{key}", default=value)
                
            self.args = self.parser.parse_args()
        
    def Initialize_Models(self):
        self.GPT_model_path = self.args.GPT_model_path
        self.SoVITS_model_path = self.args.SoVITS_model_path
        
        # 加载模型
        change_gpt_weights(gpt_path=self.GPT_model_path)
        change_sovits_weights(sovits_path=self.SoVITS_model_path)
    
    def Perform_Inference(self,ref_wav_path,prompt_text,prompt_languageself,target_text,target_text_language,cut,output_path):
          
        if (cut == "凑四句一切"):
            how_to_cut = self.i18n(cut)
        elif (cut == "凑50字一切"):
            how_to_cut = self.i18n(cut)
        elif (cut == "按中文句号。切"):
            how_to_cut = self.i18n(cut)
        elif (cut == "按英文句号.切"):
            how_to_cut = self.i18n(cut)
        elif (cut == "按标点符号切"):
            how_to_cut = self.i18n(cut)

        top_k = self.args.top_k
        top_p = self.args.top_p
        temperature = self.args.temperature
        ref_free = self.args.ref_free
        
        # 推理
        synthesis_result = get_tts_wav(
          ref_wav_path, #参考音频文件路径
          prompt_text, #参考音频文本
          prompt_languageself, #参考音频语言
          target_text, #目标文本
          target_text_language, #目标文本语言
          
          how_to_cut,
          top_k,
          top_p,
          temperature,
          ref_free,
        )
        result_list = list(synthesis_result)

        if result_list:
            last_sampling_rate, last_audio_data = result_list[-1]
            output_wav_path = output_path
            sf.write(output_wav_path, last_audio_data, last_sampling_rate)
            print("保存名称："+output_wav_path)
        
if __name__ == '__main__':
    
    GPT = GPT_SoVITS_Model()
    
    GPT.Initialize_Parames(params=None)
    
    while True:
        text = input("请输入文本：")
        
        GPT.Initialize_Models()
        
        GPT.Perform_Inference(
          ref_wav_path = "1.WAV",
          prompt_text = '汗流浃背，汗流浃背啊，我说我说我说能说的吧',
          prompt_languageself = GPT.i18n("中文"),
          target_text = text,
          target_text_language = GPT.i18n("中文"),
          cut = ("凑50字一切"),
          output_path = "2.WAV"
        )
    