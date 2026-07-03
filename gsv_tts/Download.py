import os
import time
import logging
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path


base_url = None
modelscope_base_url = "https://modelscope.cn/models/chinokiki/GPTSoVITS-RT/resolve/master/%s"
huggingface_base_url = "https://huggingface.co/cnmds/GPTSoVITS-RT/resolve/main/%s?download=true"
hf_mirror_base_url = "https://hf-mirror.com/cnmds/GPTSoVITS-RT/resolve/main/%s?download=true"

# Default GPT/SoVITS model files (not in pretrained_models zip)
_DEFAULT_MODEL_FILES = [
    "s1v3.ckpt",
    "s2Gv2ProPlus.pth",
]


def download_file(url, filename):
    logging.info(f"Downloading model from {url}")

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        logging.error(f"ERROR: Download of {filename} incomplete or something went wrong. Expected {total_size_in_bytes} bytes, got {progress_bar.n} bytes.")
    else:
        logging.info(f"Download complete: {filename}")


def unzip_file(zip_filepath, extract_to):
    logging.info(f"Extracting {zip_filepath}...")
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logging.info(f"Extraction complete, files located at: {extract_to}")


def check_latency(url, timeout=3):
    try:
        start_time = time.time()
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        
        if response.status_code == 405:
            response = requests.get(url, timeout=timeout, stream=True)
            response.close()
            
        end_time = time.time()
        
        if 200 <= response.status_code < 400:
            latency = (end_time - start_time) * 1000
            return True, latency
        else:
            return False, float('inf')
            
    except requests.RequestException:
        return False, float('inf')


def get_base_url():
    ms_url = "https://www.modelscope.cn"
    hf_url = "https://huggingface.co"
    hfm_url = "https://hf-mirror.com"

    ms_ok, ms_latency = check_latency(ms_url, timeout=5)
    hf_ok, hf_latency = check_latency(hf_url, timeout=5)
    hfm_ok, hfm_latency = check_latency(hfm_url, timeout=5)

    # ModelScope is best for China — prefer it if available
    if ms_ok and (not hf_ok or ms_latency < hf_latency):
        logging.info("Selected ModelScope.")
        return modelscope_base_url

    # hf-mirror is fast in China, slower than ModelScope but faster than raw HF
    if hfm_ok and (not hf_ok or hfm_latency < hf_latency):
        logging.info("Selected HF-Mirror (hf-mirror.com).")
        return hf_mirror_base_url

    if hf_ok:
        logging.info("Selected Hugging Face.")
        return huggingface_base_url

    logging.error("All sources unreachable. Defaulting to HF-Mirror.")
    return hf_mirror_base_url


def download_model(filename, dir, download_url=None):
    if download_url is None:
        global base_url
        if base_url is None:
            base_url = get_base_url()
            
        download_url = base_url
        
    url = download_url % (filename)
    zip_filename = Path(dir) / filename

    download_file(url, zip_filename)
    unzip_file(zip_filename, os.path.dirname(zip_filename))
    os.remove(zip_filename)


def check_pretrained_models(models_dir):
    model_list = [
        Path(models_dir) / "chinese-hubert-base",
        Path(models_dir) / "g2p",
        Path(models_dir) / "sv",
    ]

    is_download = False
    for model_path in model_list:
        if not os.path.exists(model_path):
            is_download = True
            break
    
    if is_download:
        global base_url
        if base_url is None:
            base_url = get_base_url()

        os.makedirs(models_dir, exist_ok=True)

        if base_url == modelscope_base_url:
            download_model(
                download_url=base_url,
                filename="pretrained_models5.zip",
                dir=models_dir,
            )

        elif base_url == huggingface_base_url:
            download_model(
                download_url=base_url,
                filename="pretrained_models6.zip",
                dir=models_dir,
            )

            download_model(
                download_url="https://github.com/chinokikiss/GSV-TTS-Lite/releases/download/g2p/%s",
                filename="g2p.zip",
                dir=models_dir,
            )

        else:
            # hf-mirror or any other: download same zip as HF
            download_model(
                download_url=base_url,
                filename="pretrained_models6.zip",
                dir=models_dir,
            )

            download_model(
                download_url="https://github.com/chinokikiss/GSV-TTS-Lite/releases/download/g2p/%s",
                filename="g2p.zip",
                dir=models_dir,
            )


def ensure_default_models(models_dir):
    """Download default GPT and SoVITS model files if not present.

    Downloads s1v3.ckpt (~300MB) and s2Gv2ProPlus.pth (~500MB) using the
    automatically selected mirror (ModelScope > hf-mirror > HuggingFace).

    These are single .ckpt/.pth files (not zips), downloaded directly to models_dir.
    Existing files are skipped.

    Args:
        models_dir: Directory to place the model files (typically ~/.cache/gsv/).
    """
    global base_url
    if base_url is None:
        base_url = get_base_url()

    os.makedirs(models_dir, exist_ok=True)

    for filename in _DEFAULT_MODEL_FILES:
        filepath = Path(models_dir) / filename
        if filepath.exists():
            logging.info(f"Default model already exists: {filename}")
            continue

        # Try primary URL (auto-selected mirror)
        url = base_url % filename
        logging.info(f"Downloading default model: {filename}")
        try:
            download_file(url, filepath)
        except Exception as e:
            logging.warning(f"Primary download failed for {filename}: {e}")

            # Fallback chain
            fallbacks = []
            if base_url != hf_mirror_base_url:
                fallbacks.append(("hf-mirror.com", hf_mirror_base_url))
            if base_url != huggingface_base_url:
                fallbacks.append(("Hugging Face", huggingface_base_url))
            if base_url != modelscope_base_url:
                fallbacks.append(("ModelScope", modelscope_base_url))

            for name, fb_url in fallbacks:
                try:
                    logging.info(f"Trying {name} fallback: {filename}")
                    download_file(fb_url % filename, filepath)
                    logging.info(f"Downloaded via {name}")
                    break
                except Exception as e2:
                    logging.warning(f"{name} fallback failed: {e2}")
            else:
                raise RuntimeError(
                    f"Failed to download {filename} from all sources. "
                    f"Please download manually and place it in {models_dir}"
                )


cnroberta_int8_modelscope_base_url = "https://modelscope.cn/models/ltyytn/cnroberta_int8_dynamic/resolve/master/%s"
cnroberta_int8_huggingface_base_url = "https://huggingface.co/cnmds/GPTSoVITS-RT/resolve/main/int8/cnroberta/%s?download=true"

def download_cnroberta_int8(dir, download_url=None):
    """下载 CNRoberta INT8 Dynamic ONNX 模型"""
    if download_url is None:
        global base_url
        if base_url is None:
            base_url = get_base_url()
        
        if base_url == modelscope_base_url:
            download_url = cnroberta_int8_modelscope_base_url
        else:
            download_url = cnroberta_int8_huggingface_base_url
    
    os.makedirs(dir, exist_ok=True)
    
    files_to_download = [
        "config.json",
        "tokenizer.json",
        "cnroberta_int8_dynamic.onnx",
    ]
    
    for filename in files_to_download:
        url = download_url % filename
        filepath = Path(dir) / filename
        
        if os.path.exists(filepath):
            logging.info(f"文件已存在，跳过: {filepath}")
            continue
        
        logging.info(f"正在下载: {filename}")
        download_file(url, filepath)
    
    logging.info(f"CNRoberta INT8 ONNX 模型下载完成: {dir}")