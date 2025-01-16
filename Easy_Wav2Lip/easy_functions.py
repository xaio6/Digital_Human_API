import torch
import subprocess
import json
import os
import dlib
import gdown
import re
import pickle
import re
from models import Wav2Lip
from base64 import b64encode
from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir
from IPython.display import HTML, display

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def get_video_details(filename):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        filename,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    info = json.loads(result.stdout)

    # Get video stream
    video_stream = next(
        stream for stream in info["streams"] if stream["codec_type"] == "video"
    )

    # Get resolution
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    resolution = width * height

    # Get fps
    fps = eval(video_stream["avg_frame_rate"])

    # Get length
    length = float(info["format"]["duration"])

    return width, height, fps, length


def show_video(file_path):
    """Function to display video in Colab"""
    mp4 = open(file_path, "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    width, _, _, _ = get_video_details(file_path)
    display(
        HTML(
            """
  <video controls width=%d>
      <source src="%s" type="video/mp4">
  </video>
  """
            % (min(width, 1280), data_url)
        )
    )


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def _load(checkpoint_path):
    if device != "cpu":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
    return checkpoint


def load_model(path):
    # If results file exists, load it and return
    working_directory = os.getcwd()
    folder, filename_with_extension = os.path.split(path)
    filename, file_type = os.path.splitext(filename_with_extension)
    results_file = os.path.join(folder, filename + ".pk1")
    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            return pickle.load(f)
    model = Wav2Lip()
    print("Loading {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    # Save results to file
    with open(results_file, "wb") as f:
        pickle.dump(model.eval(), f)
    # os.remove(path)
    return model.eval()

def get_input_length(filename):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # 使用正则表达式提取时长
    duration = re.search(rb'\d+\.\d+', result.stdout)
    if duration:
        return float(duration.group(0))
    else:
        raise ValueError("无法从ffprobe输出中提取时长")


def is_url(string):
    url_regex = re.compile(r"^(https?|ftp)://[^\s/$.?#].[^\s]*$")
    return bool(url_regex.match(string))


def load_predictor():
    checkpoint = os.path.join(
        "checkpoints", "shape_predictor_68_face_landmarks_GTX.dat"
    )
    predictor = dlib.shape_predictor(checkpoint)
    mouth_detector = dlib.get_frontal_face_detector()

    # Serialize the variables
    with open(os.path.join("checkpoints", "predictor.pkl"), "wb") as f:
        pickle.dump(predictor, f)

    with open(os.path.join("checkpoints", "mouth_detector.pkl"), "wb") as f:
        pickle.dump(mouth_detector, f)

    # delete the .dat file as it is no longer needed
    # os.remove(output)


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def g_colab():
    try:
        import google.colab

        return True
    except ImportError:
        return False
