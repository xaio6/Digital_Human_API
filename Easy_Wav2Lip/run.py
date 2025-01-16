import os
import sys
import re
import argparse
from easy_functions import (format_time,
                            get_input_length,
                            get_video_details,
                            show_video,
                            g_colab)
import contextlib
import shutil
import subprocess
import time
from IPython.display import Audio, Image, clear_output, display
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import configparser

parser = argparse.ArgumentParser(description='Easy-Wav2Lip main run file')

parser.add_argument('-video_file', type=str, 
                    help='Input video file path', required=False, default="1111.mp4")
parser.add_argument('-vocal_file', type=str, 
                    help='Input audio file path', required=False, default="1.WAV")
parser.add_argument('-output_file', type=str, 
                    help='Output video file path', required=False, default=False)
args = parser.parse_args()

# retrieve variables from config.ini
config = configparser.ConfigParser()

config.read('config.ini')
if args.video_file:
    video_file = args.video_file
else:
    video_file = config['OPTIONS']['video_file']

if args.vocal_file:
    vocal_file = args.vocal_file
else:
    vocal_file = config['OPTIONS']['vocal_file']
quality = config['OPTIONS']['quality']
output_height = config['OPTIONS']['output_height']
wav2lip_version = config['OPTIONS']['wav2lip_version']
use_previous_tracking_data = config['OPTIONS']['use_previous_tracking_data']
nosmooth = config.getboolean('OPTIONS', 'nosmooth')
U = config.getint('PADDING', 'U')
D = config.getint('PADDING', 'D')
L = config.getint('PADDING', 'L')
R = config.getint('PADDING', 'R')
size = config.getfloat('MASK', 'size')
feathering = config.getint('MASK', 'feathering')
mouth_tracking = config.getboolean('MASK', 'mouth_tracking')
debug_mask = config.getboolean('MASK', 'debug_mask')
batch_process = config.getboolean('OTHER', 'batch_process')
output_suffix = config['OTHER']['output_suffix']
include_settings_in_suffix = config.getboolean('OTHER', 'include_settings_in_suffix')

if g_colab():
    preview_input = config.getboolean("OTHER", "preview_input")
else:
    preview_input = False
    
preview_settings = config.getboolean("OTHER", "preview_settings")
frame_to_preview = config.getint("OTHER", "frame_to_preview")

working_directory = os.getcwd()


start_time = time.time()

video_file = video_file.strip('"')
vocal_file = vocal_file.strip('"')

# check video_file exists
if video_file == "":
    sys.exit(f"video_file cannot be blank")

if os.path.isdir(video_file):
    sys.exit(f"{video_file} is a directory, you need to point to a file")

if not os.path.exists(video_file):
    sys.exit(f"Could not find file: {video_file}")

if wav2lip_version == "Wav2Lip_GAN":
    checkpoint_path = os.path.join(working_directory, "checkpoints", "Wav2Lip_GAN.pth")
else:
    checkpoint_path = os.path.join(working_directory, "checkpoints", "Wav2Lip.pth")

if feathering == 3:
    feathering = 5
if feathering == 2:
    feathering = 3

resolution_scale = 1
res_custom = False
if output_height == "half resolution":
    resolution_scale = 2
elif output_height == "full resolution":
    resolution_scale = 1
else:
    res_custom = True
    resolution_scale = 3

in_width, in_height, in_fps, in_length = get_video_details(video_file)
out_height = round(in_height / resolution_scale)

if res_custom:
    out_height = int(output_height)
fps_for_static_image = 30


if output_suffix == "" and not include_settings_in_suffix:
    sys.exit(
        "Current suffix settings will overwrite your input video! Please add a suffix or tick include_settings_in_suffix"
    )

frame_to_preview = max(frame_to_preview - 1, 0)

if include_settings_in_suffix:
    if wav2lip_version == "Wav2Lip_GAN":
        output_suffix = f"{output_suffix}_GAN"
    output_suffix = f"{output_suffix}_{quality}"
    if output_height != "full resolution":
        output_suffix = f"{output_suffix}_{out_height}"
    if nosmooth:
        output_suffix = f"{output_suffix}_nosmooth1"
    else:
        output_suffix = f"{output_suffix}_nosmooth0"
    if U != 0 or D != 0 or L != 0 or R != 0:
        output_suffix = f"{output_suffix}_pads-"
        if U != 0:
            output_suffix = f"{output_suffix}U{U}"
        if D != 0:
            output_suffix = f"{output_suffix}D{D}"
        if L != 0:
            output_suffix = f"{output_suffix}L{L}"
        if R != 0:
            output_suffix = f"{output_suffix}R{R}"
    if quality != "fast":
        output_suffix = f"{output_suffix}_mask-S{size}F{feathering}"
        if mouth_tracking:
            output_suffix = f"{output_suffix}_mt"
        if debug_mask:
            output_suffix = f"{output_suffix}_debug"
if preview_settings:
    output_suffix = f"{output_suffix}_preview"


rescaleFactor = str(round(1 // resolution_scale))
pad_up = str(round(U * resolution_scale))
pad_down = str(round(D * resolution_scale))
pad_left = str(round(L * resolution_scale))
pad_right = str(round(R * resolution_scale))
################################################################################


######################### reconstruct input paths ##############################
# Extract each part of the path
folder, filename_with_extension = os.path.split(video_file)
filename, file_type = os.path.splitext(filename_with_extension)

# Extract filenumber if it exists
filenumber_match = re.search(r"\d+$", filename)
if filenumber_match:  # if there is a filenumber - extract it
    filenumber = str(filenumber_match.group())
    filenamenonumber = re.sub(r"\d+$", "", filename)
else:  # if there is no filenumber - make it blank
    filenumber = ""
    filenamenonumber = filename

# if vocal_file is blank - use the video as audio
if vocal_file == "":
    vocal_file = video_file
# if not, check that the vocal_file file exists
else:
    if not os.path.exists(vocal_file):
        sys.exit(f"Could not find file: {vocal_file}")
    if os.path.isdir(vocal_file):
        sys.exit(f"{vocal_file} is a directory, you need to point to a file")

# Extract each part of the path
audio_folder, audio_filename_with_extension = os.path.split(vocal_file)
audio_filename, audio_file_type = os.path.splitext(audio_filename_with_extension)

# Extract filenumber if it exists
audio_filenumber_match = re.search(r"\d+$", audio_filename)
if audio_filenumber_match:  # if there is a filenumber - extract it
    audio_filenumber = str(audio_filenumber_match.group())
    audio_filenamenonumber = re.sub(r"\d+$", "", audio_filename)
else:  # if there is no filenumber - make it blank
    audio_filenumber = ""
    audio_filenamenonumber = audio_filename
################################################################################

# set process_failed to False so that it may be set to True if one or more processings fail
process_failed = False


temp_output = os.path.join(working_directory, "temp", "output.mp4")
temp_folder = os.path.join(working_directory, "temp")

last_input_video = None
last_input_audio = None

# --------------------------Batch processing loop-------------------------------!
while True:

    # construct input_video
    input_video = os.path.join(folder, filenamenonumber + str(filenumber) + file_type)
    input_videofile = os.path.basename(input_video)

    # construct input_audio
    input_audio = os.path.join(
        audio_folder, audio_filenamenonumber + str(audio_filenumber) + audio_file_type
    )
    input_audiofile = os.path.basename(input_audio)

    # see if filenames are different:
    if filenamenonumber + str(filenumber) != audio_filenamenonumber + str(
        audio_filenumber
    ):
        output_filename = (
            filenamenonumber
            + str(filenumber)
            + "_"
            + audio_filenamenonumber
            + str(audio_filenumber)
        )
    else:
        output_filename = filenamenonumber + str(filenumber)

    # construct output_video
    output_video = os.path.join(folder, output_filename + output_suffix + ".mp4")
    output_video = os.path.normpath(output_video)
    output_videofile = os.path.basename(output_video)

    # remove last outputs
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp", exist_ok=True)

    # preview inputs (if enabled)
    if preview_input:
        print("input video:")
        # show_video(input_video)
        if vocal_file != "":
            print("input audio:")
            display(Audio(input_audio))
        else:
            print("using", input_videofile, "for audio")
        print("You may want to check now that they're the correct files!")

    last_input_video = input_video
    last_input_audio = input_audio
    shutil.copy(input_video, temp_folder)
    shutil.copy(input_audio, temp_folder)

    # rename temp file to include padding or else changing padding does nothing
    temp_input_video = os.path.join(temp_folder, input_videofile)
    renamed_temp_input_video = os.path.join(
        temp_folder, str(U) + str(D) + str(L) + str(R) + input_videofile
    )
    shutil.copy(temp_input_video, renamed_temp_input_video)
    temp_input_video = renamed_temp_input_video
    temp_input_videofile = os.path.basename(renamed_temp_input_video)
    temp_input_audio = os.path.join(temp_folder, input_audiofile)

    # trim video if it's longer than the audio
    video_length = get_input_length(temp_input_video)
    audio_length = get_input_length(temp_input_audio)

    # if preview_settings:
    #     batch_process = False

    #     preview_length_seconds = 1
    #     converted_preview_frame = frame_to_preview / in_fps
    #     preview_start_time = min(
    #         converted_preview_frame, video_length - preview_length_seconds
    #     )

    #     preview_video_path = os.path.join(
    #         temp_folder,
    #         "preview_"
    #         + str(preview_start_time)
    #         + "_"
    #         + str(U)
    #         + str(D)
    #         + str(L)
    #         + str(R)
    #         + input_videofile,
    #     )
    #     preview_audio_path = os.path.join(temp_folder, "preview_" + input_audiofile)

    #     subprocess.call(
    #         [
    #             "ffmpeg",
    #             "-loglevel",
    #             "error",
    #             "-i",
    #             temp_input_video,
    #             "-ss",
    #             str(preview_start_time),
    #             "-to",
    #             str(preview_start_time + preview_length_seconds),
    #             "-c",
    #             "copy",
    #             preview_video_path,
    #         ]
    #     )
    #     subprocess.call(
    #         [
    #             "ffmpeg",
    #             "-loglevel",
    #             "error",
    #             "-i",
    #             temp_input_audio,
    #             "-ss",
    #             str(preview_start_time),
    #             "-to",
    #             str(preview_start_time + 1),
    #             "-c",
    #             "copy",
    #             preview_audio_path,
    #         ]
    #     )
    #     temp_input_video = preview_video_path
    #     temp_input_audio = preview_audio_path

    if video_length > audio_length:
        trimmed_video_path = os.path.join(
            temp_folder, "trimmed_" + temp_input_videofile
        )
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
                devnull
            ):
                ffmpeg_extract_subclip(
                    temp_input_video, 0, audio_length, targetname=trimmed_video_path
                )
        temp_input_video = trimmed_video_path
        
    # check if face detection has already happened on this clip
    last_detected_face = os.path.join(working_directory, "last_detected_face.pkl")
    if os.path.isfile("last_file.txt"):
        with open("last_file.txt", "r") as file:
            last_file = file.readline()
        if last_file != temp_input_video or use_previous_tracking_data == "False":
            if os.path.isfile(last_detected_face):
                os.remove(last_detected_face)

    # ----------------------------Process the inputs!-----------------------------!
    print(
        f"Processing{' preview of' if preview_settings else ''} "
        f"{input_videofile} using {input_audiofile} for audio"
    )

    # execute Wav2Lip & upscaler

    cmd = [
        sys.executable,
        "inference.py",
        "--face",
        temp_input_video,
        "--audio",
        temp_input_audio,
        "--outfile",
        temp_output,
        "--pads",
        str(pad_up),
        str(pad_down),
        str(pad_left),
        str(pad_right),
        "--checkpoint_path",
        checkpoint_path,
        "--out_height",
        str(out_height),
        "--fullres",
        str(resolution_scale),
        "--quality",
        quality,
        "--mask_dilation",
        str(size),
        "--mask_feathering",
        str(feathering),
        "--nosmooth",
        str(nosmooth),
        "--debug_mask",
        str(debug_mask),
        "--preview_settings",
        str(preview_settings),
        "--mouth_tracking",
        str(mouth_tracking),
    ]

    # Run the command
    subprocess.run(cmd)

    if preview_settings:
        if os.path.isfile(os.path.join(temp_folder, "preview.jpg")):
            print(f"preview successful! Check out temp/preview.jpg")
            with open("last_file.txt", "w") as f:
                f.write(temp_input_video)
            # end processing timer and format the time it took
            end_time = time.time()
            elapsed_time = end_time - start_time
            formatted_setup_time = format_time(elapsed_time)
            print(f"Execution time: {formatted_setup_time}")
            break

        else:
            print(f"Processing failed! :( see line above 👆")
            print("Consider searching the issues tab on the github:")
            print("https://github.com/anothermartz/Easy-Wav2Lip/issues")
            exit()

    # rename temp file and move to correct directory
    if os.path.isfile(temp_output):
        if os.path.isfile(output_video):
            os.remove(output_video)
        shutil.copy(temp_output, output_video)
        # show output video
        with open("last_file.txt", "w") as f:
            f.write(temp_input_video)
        print(f"{output_filename} successfully lip synced! It will be found here:")
        print(output_video)

        # end processing timer and format the time it took
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_setup_time = format_time(elapsed_time)
        print(f"Execution time: {formatted_setup_time}")

    else:
        print(f"Processing failed! :( see line above 👆")
        print("Consider searching the issues tab on the github:")
        print("https://github.com/anothermartz/Easy-Wav2Lip/issues")
        process_failed = True

    if batch_process == False:
        if process_failed:
            exit()
        else:
            break

    elif filenumber == "" and audio_filenumber == "":
        print("Files not set for batch processing")
        break

    # -----------------------------Batch Processing!------------------------------!
    if filenumber != "":  # if video has a filenumber
        match = re.search(r"\d+", filenumber)
        # add 1 to video filenumber
        filenumber = (
            f"{filenumber[:match.start()]}{int(match.group())+1:0{len(match.group())}d}"
        )

    if audio_filenumber != "":  # if audio has a filenumber
        match = re.search(r"\d+", audio_filenumber)
        # add 1 to audio filenumber
        audio_filenumber = f"{audio_filenumber[:match.start()]}{int(match.group())+1:0{len(match.group())}d}"

    # construct input_video
    input_video = os.path.join(folder, filenamenonumber + str(filenumber) + file_type)
    input_videofile = os.path.basename(input_video)
    # construct input_audio
    input_audio = os.path.join(
        audio_folder, audio_filenamenonumber + str(audio_filenumber) + audio_file_type
    )
    input_audiofile = os.path.basename(input_audio)

    # now check which input files exist and what to do for each scenario

    # both +1 files exist - continue processing
    if os.path.exists(input_video) and os.path.exists(input_audio):
        continue

    # video +1 only - continue with last audio file
    if os.path.exists(input_video) and input_video != last_input_video:
        if audio_filenumber != "":  # if audio has a filenumber
            match = re.search(r"\d+", audio_filenumber)
            # take 1 from audio filenumber
            audio_filenumber = f"{audio_filenumber[:match.start()]}{int(match.group())-1:0{len(match.group())}d}"
        continue

    # audio +1 only - continue with last video file
    if os.path.exists(input_audio) and input_audio != last_input_audio:
        if filenumber != "":  # if video has a filenumber
            match = re.search(r"\d+", filenumber)
            # take 1 from video filenumber
            filenumber = f"{filenumber[:match.start()]}{int(match.group())-1:0{len(match.group())}d}"
        continue

    # neither +1 files exist or current files already processed - finish processing
    print("Finished all sequentially numbered files")
    if process_failed:
        sys.exit("Processing failed on at least one video")
    else:
        break
