from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

import yaml

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path


class SadTalker_Model():
    def __init__(self):
        
        #全局变量
        self.parser = None
        self.args = None
        self.sadtalker_paths = None #绝对路径
        
        #模型
        self.preprocess = None
        self.audio_to = None
        self.animate_from = None
    
    #初始化其他变量
    def Initialize_Parames(self, parames=None): 
        self.parser = ArgumentParser()
        if parames == None:
            # 添加默认参数
            with open('SadTalker/SadTalker_config.yaml', 'r') as stream:
                config = yaml.safe_load(stream)
            for key, value in config.items():
                self.parser.add_argument(f"--{key}", default=value)
                
            self.args = self.parser.parse_args()
        
        else:
            with open(parames, 'r') as stream:
                config = yaml.safe_load(stream)
            for key, value in config.items():
                self.parser.add_argument(f"--{key}", default=value)
                
            self.args = self.parser.parse_args()
        
        # 设置设备
        if torch.cuda.is_available() and not self.args.cpu:
            self.args.device = "cuda"
        else:
            self.args.device = "cpu"
            
        # 初始化路径
        current_root_path = os.path.split(sys.argv[0])[0]
        self.sadtalker_paths = init_path(self.args.checkpoint_dir, os.path.join(current_root_path, 'SadTalker/src/config'), self.args.size, self.args.old_version, self.args.preprocess)
    
    #初始化模型
    def Initialize_Models(self):
        self.preprocess = CropAndExtract(self.sadtalker_paths, self.args.device)
        self.audio_to = Audio2Coeff(self.sadtalker_paths, self.args.device)
        self.animate_from = AnimateFromCoeff(self.sadtalker_paths, self.args.device)
        
    #推理
    def Perform_Inference(self,image_save_path,audio_save_path,save_path_name):
        #torch.backends.cudnn.enabled = False
        args = self.args

        pic_path = image_save_path #照片位置
        audio_path = audio_save_path #音频位置
        save_dir = save_path_name #保存位置   
        
        os.makedirs(save_dir, exist_ok=True)
        pose_style = args.pose_style
        device = args.device
        batch_size = args.batch_size
        input_yaw_list = args.input_yaw
        input_pitch_list = args.input_pitch
        input_roll_list = args.input_roll
        ref_eyeblink = args.ref_eyeblink
        ref_pose = args.ref_pose

        #初始化模型
        preprocess_model = self.preprocess
        audio_to_coeff = self.audio_to
        animate_from_coeff = self.animate_from

        ##裁剪图像并从图像中提取3dmm
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        print('3DMM Extraction for source image')
        first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,\
                                                                                source_image_flag=True, pic_size=args.size)
        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        if ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
            ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing eye blinking')
            ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
        else:
            ref_eyeblink_coeff_path=None

        if ref_pose is not None:
            if ref_pose == ref_eyeblink: 
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print('3DMM Extraction for the reference video providing pose')
                ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
        else:
            ref_pose_coeff_path=None

        #audio2ceoff
        batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
        coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

        # 3dface render
        if args.face3dvis:
            from src.face3d.visualize import gen_composed_video
            gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
        
        #coeff2video
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                    batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                    expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)
        
        result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                    enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size)
        
        shutil.move(result, save_dir+'.mp4')
        print('生成的视频命名为:', save_dir+'.mp4')

        if not args.verbose:
            shutil.rmtree(save_dir)



# if __name__ == '__main__':
    # "该项目由Ai Horizons团队开源"
    # "2024/8/31"
    # "开发成员：魏伟辉、林训仪"
        
    # 创建一个线程池
    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)  # 根据需求设置最大工作线程数量

    # Sta = SadTalker_Model()
    # Sta.Initialize_Params()
    # Sta.Initialize_Models()
    
    # while True:
    #     x = input("请输入")
    #     if(x != "q"):
    #         Sta.Perform_Inference(x)
    #     elif(x == "q"):
    #         break

    # 提交任务给线程池处理
    # executor.submit(perform_inference, args, preprocess, audio_to, animate_from)

