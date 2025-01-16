import warnings
from gfpgan import GFPGANer

warnings.filterwarnings("ignore")


def load_sr():
    run_params = GFPGANer(
        model_path="Easy_Wav2Lip/checkpoints/GFPGANv1.4.pth",
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    return run_params


def upscale(image, properties):
    _, _, output = properties.enhance(
        image, has_aligned=False, only_center_face=False, paste_back=True
    )
    return output
