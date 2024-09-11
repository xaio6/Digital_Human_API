import os

from webui import open1abc_Train, open1Ba_Train, open1Bb_Train

now_dir = os.path.dirname(os.path.abspath(__file__))
print(now_dir)

class GPT_SoVITS_Tarin():
    def __init__(self,name):
        
        #数据格式化参数
        self.name = name
        self.gpu_numbers1a = '0-0'
        self.gpu_numbers1Ba = '0-0'
        self.gpu_numbers1c = '0-0'
        self.bert_pretrained_dir = os.path.join(now_dir, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
        self.ssl_pretrained_dir = os.path.join(now_dir, "GPT_SoVITS/pretrained_models/chinese-hubert-base")
        self.pretrained_s2G_path = os.path.join(now_dir,"GPT_SoVITS/pretrained_models/s2G488k.pth")
        
        #SoVITS训练参数跟GPT训练参数
        self.batch_size = 3
        self.text_low_lr_rate = 0.4 #文本模块学习率权重
        self.if_save_latest = True #是否仅保存最新的ckpt文件以节省硬盘空间
        self.if_save_every_weights = True #是否在每次保存时间点将最终小模型保存至weights文件夹
        self.gpu_numbers1Ba = "0"
        self.pretrained_s2G = os.path.join(now_dir, "GPT_SoVITS/pretrained_models/s2G488k.pth")
        self.pretrained_s2D = os.path.join(now_dir, "GPT_SoVITS/pretrained_models/s2D488k.pth")
    
        self.if_dpo = False
        self.gpu_numbers = "0"
        self.pretrained_s1 = os.path.join(now_dir, "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
        
        
    def Format_Data(self, inp_text,inp_wav_dir):
        result = open1abc_Train(inp_text,inp_wav_dir,
                    self.name,
                    self.gpu_numbers1a,
                    self.gpu_numbers1Ba,
                    self.gpu_numbers1c,
                    self.bert_pretrained_dir,
                    self.ssl_pretrained_dir,
                    self.pretrained_s2G_path)
        if(result):
            return True
        else:
            return False
        
    def Train_SoVITS(self, total_epoch, save_every_epoch):
        result = open1Ba_Train(total_epoch=total_epoch,  save_every_epoch=save_every_epoch,
                      
            exp_name=self.name,
            batch_size=self.batch_size,
            text_low_lr_rate=self.text_low_lr_rate,
            if_save_latest=self.if_save_latest,
            if_save_every_weights=self.if_save_every_weights,
            gpu_numbers1Ba=self.gpu_numbers1Ba,
            pretrained_s2G=self.pretrained_s2G,
            pretrained_s2D=self.pretrained_s2D
            )
        if(result):
            path = os.path.join(now_dir, "SoVITS_weights",f"{self.name}.pth")
            return path
        else:
            return None
        
    def Train_GPT(self, total_epoch, save_every_epoch):
        result = open1Bb_Train(total_epoch=total_epoch,save_every_epoch=save_every_epoch,
            
            batch_size=self.batch_size,
            exp_name=self.name,
            if_dpo=self.if_dpo,
            if_save_latest=self.if_save_latest,
            if_save_every_weights=self.if_save_every_weights,
            gpu_numbers=self.gpu_numbers,
            pretrained_s1=self.pretrained_s1
            )
        if(result):
            path = os.path.join(now_dir, "GPT_weights",f"{self.name}.ckpt")
            return path
        else:
            return None
    
if __name__ == '__main__':
    GST = GPT_SoVITS_Tarin("Wei")
    
    # GST.Format_Data("C:\\Users\\XiaoHui\\Desktop\\man\\list\\split.list",
    #                     "C:\\Users\\XiaoHui\\Desktop\\man\\slicer_opt")
    
    # x=GST.Train_SoVITS(total_epoch=8,save_every_epoch=8)
    x = GST.Train_GPT(15,15)
    print(x)

    