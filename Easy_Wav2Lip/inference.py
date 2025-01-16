print("\rloading torch       ", end="")
import torch

print("\rloading numpy       ", end="")
import numpy as np

print("\rloading Image       ", end="")
from PIL import Image

print("\rloading argparse    ", end="")
import argparse

print("\rloading configparser", end="")
import configparser

print("\rloading math        ", end="")
import math

print("\rloading os          ", end="")
import os

print("\rloading subprocess  ", end="")
import subprocess

print("\rloading pickle      ", end="")
import pickle

print("\rloading cv2         ", end="")
import cv2

print("\rloading audio       ", end="")
import audio

print("\rloading RetinaFace ", end="")
from batch_face import RetinaFace

print("\rloading re          ", end="")
import re

print("\rloading partial     ", end="")
from functools import partial

print("\rloading tqdm        ", end="")
from tqdm import tqdm

print("\rloading warnings    ", end="")
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.transforms.functional_tensor"
)
print("\rloading upscale     ", end="")
from enhance import upscale

print("\rloading load_sr     ", end="")
from enhance import load_sr

print("\rloading load_model  ", end="")
from easy_functions import load_model, g_colab

print("\rimports loaded!     ")

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
gpu_id = 0 if torch.cuda.is_available() else -1

if device == 'cpu':
    print('Warning: No GPU detected so inference will be done on the CPU which is VERY SLOW!')
parser = argparse.ArgumentParser(
    description="Inference code to lip-sync videos in the wild using Wav2Lip models"
)

parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Name of saved checkpoint to load weights from",
    required=True,
)

parser.add_argument(
    "--segmentation_path",
    type=str,
    default="checkpoints/face_segmentation.pth",
    help="Name of saved checkpoint of segmentation network",
    required=False,
)

parser.add_argument(
    "--face",
    type=str,
    help="Filepath of video/image that contains faces to use",
    required=True,
)
parser.add_argument(
    "--audio",
    type=str,
    help="Filepath of video/audio file to use as raw audio source",
    required=True,
)
parser.add_argument(
    "--outfile",
    type=str,
    help="Video path to save result. See default for an e.g.",
    default="results/result_voice.mp4",
)

parser.add_argument(
    "--static",
    type=bool,
    help="If True, then use only first video frame for inference",
    default=False,
)
parser.add_argument(
    "--fps",
    type=float,
    help="Can be specified only if input is a static image (default: 25)",
    default=25.0,
    required=False,
)

parser.add_argument(
    "--pads",
    nargs="+",
    type=int,
    default=[0, 10, 0, 0],
    help="Padding (top, bottom, left, right). Please adjust to include chin at least",
)

parser.add_argument(
    "--wav2lip_batch_size", type=int, help="Batch size for Wav2Lip model(s)", default=1
)

parser.add_argument(
    "--out_height",
    default=480,
    type=int,
    help="Output video height. Best results are obtained at 480 or 720",
)

parser.add_argument(
    "--crop",
    nargs="+",
    type=int,
    default=[0, -1, 0, -1],
    help="Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. "
    "Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width",
)

parser.add_argument(
    "--box",
    nargs="+",
    type=int,
    default=[-1, -1, -1, -1],
    help="Specify a constant bounding box for the face. Use only as a last resort if the face is not detected."
    "Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).",
)

parser.add_argument(
    "--rotate",
    default=False,
    action="store_true",
    help="Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg."
    "Use if you get a flipped result, despite feeding a normal looking video",
)

parser.add_argument(
    "--nosmooth",
    type=str,
    default=False,
    help="Prevent smoothing face detections over a short temporal window",
)

parser.add_argument(
    "--no_seg",
    default=False,
    action="store_true",
    help="Prevent using face segmentation",
)

parser.add_argument(
    "--no_sr", default=False, action="store_true", help="Prevent using super resolution"
)

parser.add_argument(
    "--sr_model",
    type=str,
    default="gfpgan",
    help="Name of upscaler - gfpgan or RestoreFormer",
    required=False,
)

parser.add_argument(
    "--fullres",
    default=3,
    type=int,
    help="used only to determine if full res is used so that no resizing needs to be done if so",
)

parser.add_argument(
    "--debug_mask",
    type=str,
    default=False,
    help="Makes background grayscale to see the mask better",
)

parser.add_argument(
    "--preview_settings", type=str, default=False, help="Processes only one frame"
)

parser.add_argument(
    "--mouth_tracking",
    type=str,
    default=False,
    help="Tracks the mouth in every frame for the mask",
)

parser.add_argument(
    "--mask_dilation",
    default=150,
    type=float,
    help="size of mask around mouth",
    required=False,
)

parser.add_argument(
    "--mask_feathering",
    default=151,
    type=int,
    help="amount of feathering of mask around mouth",
    required=False,
)

parser.add_argument(
    "--quality",
    type=str,
    help="Choose between Fast, Improved and Enhanced",
    default="Fast",
)

with open(os.path.join("checkpoints", "predictor.pkl"), "rb") as f:
    predictor = pickle.load(f)

with open(os.path.join("checkpoints", "mouth_detector.pkl"), "rb") as f:
    mouth_detector = pickle.load(f)

# creating variables to prevent failing when a face isn't detected
kernel = last_mask = x = y = w = h = None

g_colab = g_colab()

if not g_colab:
  # Load the config file
  config = configparser.ConfigParser()
  config.read('config.ini')

  # Get the value of the "preview_window" variable
  preview_window = config.get('OPTIONS', 'preview_window')

all_mouth_landmarks = []

model = detector = detector_model = None

def do_load(checkpoint_path):
    global model, detector, detector_model
    model = load_model(checkpoint_path)
    detector = RetinaFace(
        gpu_id=gpu_id, model_path="checkpoints/mobilenet.pth", network="mobilenet"
    )
    detector_model = detector.model

def face_rect(images):
    face_batch_size = 8
    num_batches = math.ceil(len(images) / face_batch_size)
    prev_ret = None
    for i in range(num_batches):
        batch = images[i * face_batch_size : (i + 1) * face_batch_size]
        all_faces = detector(batch)  # return faces list of all images
        for faces in all_faces:
            if faces:
                box, landmarks, score = faces[0]
                prev_ret = tuple(map(int, box))
            yield prev_ret

def create_tracked_mask(img, original_img):
    global kernel, last_mask, x, y, w, h  # Add last_mask to global variables

    # Convert color space from BGR to RGB if necessary
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

    # Detect face
    faces = mouth_detector(img)
    if len(faces) == 0:
        if last_mask is not None:
            last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
            mask = last_mask  # use the last successful mask
        else:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img, None
    else:
        face = faces[0]
        shape = predictor(img, face)

        # Get points for mouth
        mouth_points = np.array(
            [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
        )

        # Calculate bounding box dimensions
        x, y, w, h = cv2.boundingRect(mouth_points)

        # Set kernel size as a fraction of bounding box size
        kernel_size = int(max(w, h) * args.mask_dilation)
        # if kernel_size % 2 == 0:  # Ensure kernel size is odd
        # kernel_size += 1

        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Create binary mask for mouth
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, mouth_points, 255)

        last_mask = mask  # Update last_mask with the new mask

    # Dilate the mask
    dilated_mask = cv2.dilate(mask, kernel)

    # Calculate distance transform of dilated mask
    dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

    # Normalize distance transform
    cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

    # Convert normalized distance transform to binary mask and convert it to uint8
    _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
    masked_diff = masked_diff.astype(np.uint8)

    # make sure blur is an odd number
    blur = args.mask_feathering
    if blur % 2 == 0:
        blur += 1
    # Set blur size as a fraction of bounding box size
    blur = int(max(w, h) * blur)  # 10% of bounding box size
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


def create_mask(img, original_img):
    global kernel, last_mask, x, y, w, h # Add last_mask to global variables

    # Convert color space from BGR to RGB if necessary
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

    if last_mask is not None:
        last_mask = np.array(last_mask)  # Convert PIL Image to numpy array
        last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
        mask = last_mask  # use the last successful mask
        mask = Image.fromarray(mask)

    else:
        # Detect face
        faces = mouth_detector(img)
        if len(faces) == 0:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img, None
        else:
            face = faces[0]
            shape = predictor(img, face)

            # Get points for mouth
            mouth_points = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
            )

            # Calculate bounding box dimensions
            x, y, w, h = cv2.boundingRect(mouth_points)

            # Set kernel size as a fraction of bounding box size
            kernel_size = int(max(w, h) * args.mask_dilation)
            # if kernel_size % 2 == 0:  # Ensure kernel size is odd
            # kernel_size += 1

            # Create kernel
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Create binary mask for mouth
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, mouth_points, 255)

            # Dilate the mask
            dilated_mask = cv2.dilate(mask, kernel)

            # Calculate distance transform of dilated mask
            dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

            # Normalize distance transform
            cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

            # Convert normalized distance transform to binary mask and convert it to uint8
            _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
            masked_diff = masked_diff.astype(np.uint8)

            if not args.mask_feathering == 0:
                blur = args.mask_feathering
                # Set blur size as a fraction of bounding box size
                blur = int(max(w, h) * blur)  # 10% of bounding box size
                if blur % 2 == 0:  # Ensure blur size is odd
                    blur += 1
                masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

            # Convert mask to single channel where pixel values are from the alpha channel of the current mask
            mask = Image.fromarray(masked_diff)

            last_mask = mask  # Update last_mask with the final mask after dilation and feathering

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


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T :]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes
            
def face_detect(images, results_file="last_detected_face.pkl"):
    # If results file exists, load it and return
    if os.path.exists(results_file):
        print("Using face detection data from last input")
        with open(results_file, "rb") as f:
            return pickle.load(f)

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    
    tqdm_partial = partial(tqdm, position=0, leave=True)
    for image, (rect) in tqdm_partial(
        zip(images, face_rect(images)),
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
    if str(args.nosmooth) == "False":
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [
        [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
        for image, (x1, y1, x2, y2) in zip(images, boxes)
    ]

    # Save results to file
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    return results


def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    print("\r" + " " * 100, end="\r")
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print("Using the specified bounding box instead of face detection...")
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2 :] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2 :] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )

        yield img_batch, mel_batch, frame_batch, coords_batch


mel_step_size = 16

def _load(checkpoint_path):
    if device != "cpu":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
    return checkpoint


def main():
    args.img_size = 96
    frame_number = 11

    if os.path.isfile(args.face) and args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
        args.static = True

    if not os.path.isfile(args.face):
        raise ValueError("--face argument must be a valid path to video/image file")

    elif args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        if args.fullres != 1:
            print("Resizing video...")
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            if args.fullres != 1:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(
                    frame, (int(args.out_height * aspect_ratio), args.out_height)
                )

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    if not args.audio.endswith(".wav"):
        print("Converting audio to .wav")
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                args.audio,
                "temp/temp.wav",
            ]
        )
        args.audio = "temp/temp.wav"

    print("analysing audio...")
    wav = audio.load_wav(args.audio, 16000)
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
    if str(args.preview_settings) == "True":
        full_frames = [full_frames[0]]
        mel_chunks = [mel_chunks[0]]
    print(str(len(full_frames)) + " frames to process")
    batch_size = args.wav2lip_batch_size
    if str(args.preview_settings) == "True":
        gen = datagen(full_frames, mel_chunks)
    else:
        gen = datagen(full_frames.copy(), mel_chunks)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(
        tqdm(
            gen,
            total=int(np.ceil(float(len(mel_chunks)) / batch_size)),
            desc="Processing Wav2Lip",
            ncols=100,
        )
    ):
        if i == 0:
            if not args.quality == "Fast":
                print(
                    f"mask size: {args.mask_dilation}, feathering: {args.mask_feathering}"
                )
                if not args.quality == "Improved":
                    print("Loading", args.sr_model)
                    run_params = load_sr()

            print("Starting...")
            frame_h, frame_w = full_frames[0].shape[:-1]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter("temp/result.mp4", fourcc, fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for p, f, c in zip(pred, frames, coords):
            # cv2.imwrite('temp/f.jpg', f)

            y1, y2, x1, x2 = c

            if (
                str(args.debug_mask) == "True"
            ):  # makes the background black & white so you can see the mask better
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            cf = f[y1:y2, x1:x2]

            if args.quality == "Enhanced":
                p = upscale(p, run_params)

            if args.quality in ["Enhanced", "Improved"]:
                if str(args.mouth_tracking) == "True":
                    p, last_mask = create_tracked_mask(p, cf)
                else:
                    p, last_mask = create_mask(p, cf)

            f[y1:y2, x1:x2] = p

            g_colab = True

            if not g_colab:
                # Display the frame
                if preview_window == "Face":
                    cv2.imshow("face preview - press Q to abort", p)
                elif preview_window == "Full":
                    cv2.imshow("full preview - press Q to abort", f)
                elif preview_window == "Both":
                    cv2.imshow("face preview - press Q to abort", p)
                    cv2.imshow("full preview - press Q to abort", f)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    exit()  # Exit the loop when 'Q' is pressed

            if str(args.preview_settings) == "True":
                cv2.imwrite("temp/preview.jpg", f)
                if not g_colab:
                    cv2.imshow("preview - press Q to close", f)
                    if cv2.waitKey(-1) & 0xFF == ord('q'):
                        exit()  # Exit the loop when 'Q' is pressed

            else:
                out.write(f)

    # Close the window(s) when done
    cv2.destroyAllWindows()

    out.release()

    if str(args.preview_settings) == "False":
        print("converting to final video")

        subprocess.check_call([
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            "temp/result.mp4",
            "-i",
            args.audio,
            "-c:v",
            "libx264",
            args.outfile
        ])

if __name__ == "__main__":
    args = parser.parse_args()
    do_load(args.checkpoint_path)
    main()
