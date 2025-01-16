import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import math
import os
import subprocess
import pickle
import cv2
import yaml
import audio
from batch_face import RetinaFace
from functools import partial
from tqdm import tqdm
import contextlib
from moviepy.editor import VideoFileClip

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from enhance import upscale
from enhance import load_sr
from easy_functions import load_model, get_input_length

class Wav2Lip_Model:
    def __init__(self):
        self.kernel = None
        self.last_mask = None
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        
        self.model = None
        self.detector = None
        self.detector_model = None
        
        with open(os.path.join("Easy_Wav2Lip/checkpoints", "predictor.pkl"), "rb") as f:
            self.predictor = pickle.load(f)
            
        with open(os.path.join("Easy_Wav2Lip/checkpoints", "mouth_detector.pkl"), "rb") as f:
            self.mouth_detector = pickle.load(f)
            
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.gpu_id = 0 if torch.cuda.is_available() else -1

    def Initialize_Parames(self, user, parames=None):
        self.user_data_save_path = user
        
        #临时文件位置
        self.wav2lip_temp = os.path.join(self.user_data_save_path, "wav2lip_temp")
        self.detected_face_pkl = os.path.join(self.wav2lip_temp,"last_detected_face.pkl")
        parser = ArgumentParser()
        
        if parames == None:
            # 添加默认参数
            with open('Easy_Wav2Lip/Wav2Lip_config.yaml', 'r') as stream:
                config = yaml.safe_load(stream)
            for key, value in config.items():
                parser.add_argument(f"--{key}", default=value)
            self.args = parser.parse_args()
            
        else:
            with open(parames, 'r') as stream:
                config = yaml.safe_load(stream)
            for key, value in config.items():
                parser.add_argument(f"--{key}", default=value)
            self.args = parser.parse_args()

    def Initialize_Models(self):
        checkpoint_path = self.args.checkpoint_path
        self.model = load_model(checkpoint_path)
        
        self.detector = RetinaFace(
            gpu_id=self.gpu_id, model_path="Easy_Wav2Lip/checkpoints/mobilenet.pth", network="mobilenet"
        )
        
        self.detector_model = self.detector.model

    def Face_Rect(self, images):
        face_batch_size = 8
        num_batches = math.ceil(len(images) / face_batch_size)
        prev_ret = None
        for i in range(num_batches):
            batch = images[i * face_batch_size : (i + 1) * face_batch_size]
            all_faces = self.detector(batch)  # return faces list of all images
            for faces in all_faces:
                if faces:
                    box, _, _ = faces[0]
                    prev_ret = tuple(map(int, box))
                yield prev_ret

    def Create_Tracked_Mask(self, img, original_img):
        # Convert color space from BGR to RGB if necessary
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

        # Detect face
        faces = self.mouth_detector(img)
        if len(faces) == 0:
            if self.last_mask is not None:
                self.last_mask = cv2.resize(self.last_mask, (img.shape[1], img.shape[0]))
                mask = self.last_mask  # use the last successful mask
            else:
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                return img, None
        else:
            face = faces[0]
            shape = self.predictor(img, face)

            # Get points for mouth
            mouth_points = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
            )

            # Calculate bounding box dimensions
            self.x, self.y, self.w, self.h = cv2.boundingRect(mouth_points)

            # Set kernel size as a fraction of bounding box size
            kernel_size = int(max(self.w, self.h) * self.args.mask_dilation)
            # if kernel_size % 2 == 0:  # Ensure kernel size is odd
            # kernel_size += 1

            # Create kernel
            self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Create binary mask for mouth
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, mouth_points, 255)

            self.last_mask = mask  # Update last_mask with the new mask

        # Dilate the mask
        dilated_mask = cv2.dilate(mask, self.kernel)

        # Calculate distance transform of dilated mask
        dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

        # Normalize distance transform
        cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

        # Convert normalized distance transform to binary mask and convert it to uint8
        _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
        masked_diff = masked_diff.astype(np.uint8)

        # make sure blur is an odd number
        blur = self.args.mask_feathering
        if blur % 2 == 0:
            blur += 1
        # Set blur size as a fraction of bounding box size
        blur = int(max(self.w, self.h) * blur)  # 10% of bounding box size
        if blur % 2 == 0:  # Ensure blur size is odd
            blur += 1
        masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

        # Convert numpy arrays to PIL Images
        input1 = Image.fromarray(img)
        input2 = Image.fromarray(original_img)

        # Convert mask to single channel where pixel values are from the alpha channel of the current mask
        mask = Image.fromarray(masked_diff)

        # Ensure images are the same size
        assert input1.size == input2.size == mask.size

        # Paste input1 onto input2 using the mask
        input2.paste(input1, (0, 0), mask)

        # Convert the final PIL Image back to a numpy array
        input2 = np.array(input2)

        # input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
        cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)

        return input2, mask

    def Create_Mask(self, img, original_img):

        # Convert color space from BGR to RGB if necessary
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

        if self.last_mask is not None:
            self.last_mask = np.array(self.last_mask)  # Convert PIL Image to numpy array
            self.last_mask = cv2.resize(self.last_mask, (img.shape[1], img.shape[0]))
            mask = self.last_mask  # use the last successful mask
            mask = Image.fromarray(mask)

        else:
            # Detect face
            faces = self.mouth_detector(img)
            if len(faces) == 0:
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                return img, None
            else:
                face = faces[0]
                shape = self.predictor(img, face)

                # Get points for mouth
                mouth_points = np.array(
                    [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
                )

                # Calculate bounding box dimensions
                self.x, self.y, self.w, self.h = cv2.boundingRect(mouth_points)

                # Set kernel size as a fraction of bounding box size
                kernel_size = int(max(self.w, self.h) * self.args.mask_dilation)
                # if kernel_size % 2 == 0:  # Ensure kernel size is odd
                # kernel_size += 1

                # Create kernel
                self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

                # Create binary mask for mouth
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, mouth_points, 255)

                # Dilate the mask
                dilated_mask = cv2.dilate(mask, self.kernel)

                # Calculate distance transform of dilated mask
                dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

                # Normalize distance transform
                cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

                # Convert normalized distance transform to binary mask and convert it to uint8
                _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
                masked_diff = masked_diff.astype(np.uint8)

                if not self.args.mask_feathering == 0:
                    blur = self.args.mask_feathering
                    # Set blur size as a fraction of bounding box size
                    blur = int(max(self.w, self.h) * blur)  # 10% of bounding box size
                    if blur % 2 == 0:  # Ensure blur size is odd
                        blur += 1
                    masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

                # Convert mask to single channel where pixel values are from the alpha channel of the current mask
                mask = Image.fromarray(masked_diff)

                self.last_mask = mask  # Update last_mask with the final mask after dilation and feathering

        # Convert numpy arrays to PIL Images
        input1 = Image.fromarray(img)
        input2 = Image.fromarray(original_img)

        # Resize mask to match image size
        # mask = Image.fromarray(mask)
        mask = mask.resize(input1.size)

        # Ensure images are the same size
        assert input1.size == input2.size == mask.size

        # Paste input1 onto input2 using the mask
        input2.paste(input1, (0, 0), mask)

        # Convert the final PIL Image back to a numpy array
        input2 = np.array(input2)

        # input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
        cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)

        return input2, mask

    def Shear_Video(self, video_path, audio_path):
        # trim video if it's longer than the audio
        video_length = get_input_length(video_path)
        audio_length = get_input_length(audio_path)
        
        temp_video_name = os.path.basename(video_path)
        
        if video_length > audio_length:
            trimmed_video_path = os.path.join(
                self.wav2lip_temp, "trimmed_" + temp_video_name
            )
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
                    devnull
                ):
                    ffmpeg_extract_subclip(
                        video_path, 0, audio_length, targetname=trimmed_video_path
                    )
            return trimmed_video_path
        return video_path
        
    def Change_Video_Fps(self, input_path, output_path, new_fps):
    # 加载视频文件
        clip = VideoFileClip(input_path)
        
        # 修改帧率
        new_clip = clip.set_fps(new_fps)
        
        # 保存新视频文件
        new_clip.write_videofile(output_path, codec='libx264')

    def Get_Smoothened_Boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T :]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes
                
    def Face_Detect(self, images, results_file="Easy_Wav2Lip/last_detected_face.pkl"):
        # If results file exists, load it and return
        if os.path.exists(results_file):
            print("Using face detection data from last input")
            with open(results_file, "rb") as f:
                return pickle.load(f)

        results = []
        pady1, pady2, padx1, padx2 = self.args.pads
        
        tqdm_partial = partial(tqdm, position=0, leave=True)
        for image, (rect) in tqdm_partial(
            zip(images, self.Face_Rect(images)),
            total=len(images),
            desc="detecting face in every frame",
            ncols=100,
        ):
            if rect is None:
                cv2.imwrite(
                    "temp/faulty_frame.jpg", image
                )  # check this frame where the face was not detected.
                raise ValueError(
                    "Face not detected! Ensure the video contains a face in all the frames."
                )

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])


        boxes = np.array(results)
        if str(self.args.nosmooth) == "False":
            boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [
            [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
            for image, (x1, y1, x2, y2) in zip(images, boxes)
        ]

        # Save results to file
        with open(results_file, "wb") as f:
            pickle.dump(results, f)

        return results

    def Datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        print("\r" + " " * 100, end="\r")
        
        if self.args.box[0] == -1:
            if not self.args.static:
                face_det_results = self.Face_Detect(frames, self.detected_face_pkl)  # BGR2RGB for CNN face detection
            else:
                face_det_results = self.Face_Detect([frames[0]], self.detected_face_pkl)
        else:
            print("Using the specified bounding box instead of face detection...")
            y1, y2, x1, x2 = self.args.box
            face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if self.args.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.args.img_size, self.args.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.args.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.args.img_size // 2 :] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
                mel_batch = np.reshape(
                    mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
                )

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.args.img_size // 2 :] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )

            yield img_batch, mel_batch, frame_batch, coords_batch

    def Perform_Inference(self, video_path, audio_path, outfile_path):
        
        self.args.face = video_path
        self.args.audio = audio_path
        self.args.outfile = outfile_path
        
        #判断检测人脸的pkl是否存在
        if os.path.exists(self.detected_face_pkl):
            os.remove(self.detected_face_pkl)
        
        self.args.img_size = 96
        mel_step_size = 16
        if os.path.isfile(self.args.face) and self.args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
            self.args.static = True

        if not os.path.isfile(self.args.face):
            raise ValueError("--face argument must be a valid path to video/image file")

        elif self.args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
            full_frames = [cv2.imread(self.args.face)]
            fps = self.args.fps

        else:
            if self.args.fullres != 1:
                print("Resizing video...")
            video_stream = cv2.VideoCapture(self.args.face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break

                if self.args.fullres != 1:
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    frame = cv2.resize(
                        frame, (int(self.args.out_height * aspect_ratio), self.args.out_height)
                    )

                if self.args.rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = self.args.crop
                if x2 == -1:
                    x2 = frame.shape[1]
                if y2 == -1:
                    y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)

        print("analysing audio...")
        wav = audio.load_wav(self.args.audio, 16000)
        mel = audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                "Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again"
            )

        mel_chunks = []

        mel_idx_multiplier = 80.0 / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        full_frames = full_frames[: len(mel_chunks)]
        if str(self.args.preview_settings) == "True":
            full_frames = [full_frames[0]]
            mel_chunks = [mel_chunks[0]]
        print(str(len(full_frames)) + " frames to process")
        batch_size = self.args.wav2lip_batch_size
        if str(self.args.preview_settings) == "True":
            gen = self.Datagen(full_frames, mel_chunks)
        else:
            gen = self.Datagen(full_frames.copy(), mel_chunks)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(
                gen,
                total=int(np.ceil(float(len(mel_chunks)) / batch_size)),
                desc="Processing Wav2Lip",
                ncols=100,
            )
        ):
            if i == 0:
                if not self.args.quality == "Fast":
                    print(
                        f"mask size: {self.args.mask_dilation}, feathering: {self.args.mask_feathering}"
                    )
                    if not self.args.quality == "Improved":
                        print("Loading", self.args.sr_model)
                        run_params = load_sr()

                print("Starting...")
                frame_h, frame_w = full_frames[0].shape[:-1]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(os.path.join(self.wav2lip_temp,"result.mp4"), fourcc, fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            for p, f, c in zip(pred, frames, coords):
                # cv2.imwrite('temp/f.jpg', f)

                y1, y2, x1, x2 = c

                if (
                    str(self.args.debug_mask) == "True"
                ):  # makes the background black & white so you can see the mask better
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                cf = f[y1:y2, x1:x2]

                if self.args.quality == "Enhanced":
                    p = upscale(p, run_params)

                if self.args.quality in ["Enhanced", "Improved"]:
                    if str(self.args.mouth_tracking) == "True":
                        p, _ = self.Create_Tracked_Mask(p, cf)
                    else:
                        p, _ = self.Create_Mask(p, cf)

                f[y1:y2, x1:x2] = p
                out.write(f)

        # Close the window(s) when done
        cv2.destroyAllWindows()

        out.release()

        if str(self.args.preview_settings) == "False":
            print("converting to final video")

            subprocess.check_call([
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                os.path.join(self.wav2lip_temp,"result.mp4"),
                "-i",
                self.args.audio,
                "-c:v",
                "libx264",
                self.args.outfile
            ])
        print("Done!")

if __name__ == "__main__":
        
    
    PM = Wav2Lip_Model()
    PM.Initialize_Parames("Data/Hui")
    PM.Initialize_Models()
    
    shear_video = PM.Shear_Video(
        '1.mp4',
        '6.WAV'
    )
    PM.Perform_Inference(
        shear_video,
        '6.WAV',
        'output.mp4'
    )
