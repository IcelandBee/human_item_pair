import os
import json
import time
import math
import base64
import io
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from openai import OpenAI


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =========================
# Config
# =========================

INPUT_FOLDER = "/mnt/DATA_71/public/data/testing_sets/real_data_testset/1024/human-item/gen"
REF_FOLDER = "/mnt/DATA_71/public/data/testing_sets/real_data_testset/1024/human-item/ref"

OUTPUT_JSON_PATH = "/mnt/DATA_71/public/data/testing_sets/real_data_testset/1024/human-item/human-item.json"

BASE_URL = "http://10.154.39.57:8001/v1"
API_KEY = "123456"
MODEL_NAME = "gemma-4-31B-it"

WORKERS = 12
MAX_RETRIES = 3
MAX_TOKENS = 128
TEMPERATURE = 0.0
MAX_PIXELS = 768 * 768

SKIP_FAILED = True
LIMIT = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# 避免代理影响内网服务访问
os.environ["NO_PROXY"] = "10.154.39.57,localhost,127.0.0.1"
os.environ["no_proxy"] = "10.154.39.57,localhost,127.0.0.1"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# virtual clothing
# SYSTEM_PROMPT = """
# You are a professional prompt engineer specialized in generating precise prompts for virtual clothing image editing tasks.
#
# You will be given two images:
#
# Image 1: the original person image.
# Image 2: the reference clothing image.
#
# Your task is to examine image 1 and image 2, then generate a clear image-editing prompt that describes how to dress the person in image 1 using the clothing shown in image 2.
#
# First, examine image 1 and identify the main person and the clothes currently worn by that person.
# Then examine image 2 and identify the reference clothing item or clothing set.
# Then decide whether the clothing in image 2 should replace one or more clothing items worn by the person in image 1, or whether it should be worn additionally over the existing clothing.
#
# Based on this analysis, generate exactly one final editing prompt.
#
# Use one of the following formats:
#
# output:
#
# Replace the [clothes description in image 1 to be replaced] worn by the person in image 1 with the [clothes description in image 2] in image 2, while making minimal changes and preserving the original pose of the person.
#
#
# Replace:
# - [clothes description in image 1 to be replaced] with the actual clothing item or clothing items in image 1 that should be replaced.
# - [clothes description in image 2] with the actual clothing item or clothing items shown in image 2.
# - [description of the clothing in image 1] with the original clothing that remains visible underneath.
#
# No brackets are needed in the final output.
#
# Rules:
# 1. You must describe each clothing item clearly and unambiguously in about 1-10 words, for example: suit, thermal shirt, sports coat, navy blue track jacket, grey t-shirt, etc.
# 2. Output only the final prompt sentence, with no explanation and no extra text.
# 3. If multiple clothing items are changed, replaced, or added, you must describe all relevant clothing items, for example: purple crop top and denim jacket; navy jacket and grey hoodie; black coat, white shirt, and blue jeans.
# 4. Do not mention image 3, edited result, generated result, or target image, because only image 1 and image 2 are provided.
# """.strip()
#
#
# USER_TEXT = """
# Analyse image 1 and image 2.
#
# Image 1 is the original person image.
# Image 2 is the reference clothing image.
#
# Generate one virtual clothing editing prompt that instructs how to dress the person in image 1 using the clothing from image 2.
#
# Output only one final prompt sentence using the required format.
# """.strip()

# person-texture
# SYSTEM_PROMPT = """
# You are a professional prompt engineer specialized in generating precise prompts for texture transfer image editing tasks.
#
# You will be given two images:
#
# Image 1: the original person image.
# Image 2: the reference texture image.
#
# Your task is to examine image 1 and generate exactly one texture-transfer editing prompt.
#
# First, examine image 1 and identify the main person.
# Then identify the largest visibly exposed garment worn by that person.
# You only need to analyse image 1 to determine the garment description.
# Image 2 should only be treated as the source of the texture.
#
# Based on this analysis, generate exactly one final editing prompt using the following format:
#
# Transfer the textures from image 2 to the [garment description] of the person in image 1. Preserve the garment’s shape and fit. Edit only the [garment description] region, keeping all other areas unchanged.
#
# Replace:
# - [garment description] with the actual description of the largest visibly exposed garment in image 1, for example: white t-shirt, black hoodie, blue denim jacket.
#
# No brackets are needed in the final output.
#
# Rules:
# 1. Only identify and describe the largest visibly exposed garment worn by the main person in image 1.
# 2. The garment description must be clear and unambiguous, in about 1-10 words, for example: white t-shirt, black hoodie, blue denim jacket.
# 3. Output only the final prompt sentence, with no explanation and no extra text.
# 4. Keep the output sentence strictly in the required format, changing only the garment description.
# 5. Do not describe or analyse the texture content in image 2.
# """.strip()
#
#
# USER_TEXT = """
# Analyse image 1 and image 2.
#
# Image 1 is the original person image.
# Image 2 is the reference texture image.
#
# Identify the main person in image 1, then locate the largest visibly exposed garment worn by that person.
#
# Generate one texture-transfer editing prompt in the required format.
#
# Output only one final prompt sentence.
# """.strip()

# object holding / hand-object interaction
SYSTEM_PROMPT = """
You are a professional prompt engineer specialized in generating precise prompts for hand-object interaction image editing tasks.

You will be given two images:

Image 1: the original person image.
Image 2: the reference object image.

Your task is to examine image 2 only, identify the main object shown in image 2, and generate a clear image-editing prompt that instructs the person in image 1 to naturally interact with that object using their hand or hands.

You do not need to analyse the person, pose, clothing, background, or scene in image 1. Image 1 should only be referred to as the original person image.

Based on this analysis, generate exactly one final editing prompt using the following format:

Let the person in image 1 [hand-object action phrase] [object description] shown in image 2 in a realistic and physically coherent way, preserving object integrity and overall image consistency, while making only the minimal necessary changes and keeping everything else in image 1 unchanged.

Replace:
- [hand-object action phrase] with the most natural hand-related action for the object shown in image 2, such as hold, carry, grip, hold by the handle, hold in both hands, or carry over the shoulder.
- [object description] with the main object shown in image 2.

No brackets are needed in the final output.

Rules:
1. You must identify only the main object shown in image 2.
2. The object description must be clear and unambiguous, using about 3-8 words, for example: a fresh green broccoli, a red leather handbag, a white ceramic mug, a bouquet of red roses.
3. Choose a natural hand-related action phrase based on the object type, for example: hold a broccoli, carry a handbag, grip a tennis racket, hold a mug by its handle, hold a bouquet of flowers.
4. Output only the final prompt sentence, with no explanation and no extra text.
5. Keep the output sentence strictly in the required format, changing only the hand-object action phrase and the object description.
6. Do not describe image 1 except by using the phrase “the person in image 1”.
""".strip()


USER_TEXT = """
Analyse image 2 only.

Image 1 is the original person image.
Image 2 is the reference object image.

Identify the main object shown in image 2 and choose the most natural hand-related action for interacting with that object.

Generate one hand-object interaction editing prompt using the required format.

Output only one final prompt sentence.
""".strip()


def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMG_EXTS


def load_image_rgb(path: str, max_pixels: int = 0) -> Image.Image:
    img = Image.open(path).convert("RGB")

    if max_pixels and max_pixels > 0:
        w, h = img.size
        if w * h > max_pixels:
            scale = math.sqrt(max_pixels / float(w * h))
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    return img


def pil_to_data_url(img: Image.Image, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=image_format)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def clean_text(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()

    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    cleaned = cleaned.strip('"').replace("\\", "")
    cleaned = cleaned.replace("\n", " ").replace('\\"', '"').strip()

    return cleaned


def build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def infer_one(
    client: OpenAI,
    model_name: str,
    image1_path: str,
    image2_path: str,
    max_retries: int = 3,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> str:
    img1 = load_image_rgb(image1_path, max_pixels=MAX_PIXELS)
    img2 = load_image_rgb(image2_path, max_pixels=MAX_PIXELS)

    img1_url = pil_to_data_url(img1, image_format="PNG")
    img2_url = pil_to_data_url(img2, image_format="PNG")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_TEXT},
                {"type": "image_url", "image_url": {"url": img1_url}},
                {"type": "image_url", "image_url": {"url": img2_url}},
            ],
        },
    ]

    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            text = resp.choices[0].message.content
            return clean_text(text)

        except Exception as e:
            last_err = e
            logging.warning(
                f"infer failed attempt {attempt}/{max_retries}: {repr(e)}"
            )
            if attempt < max_retries:
                time.sleep(1.0 * attempt)

    raise RuntimeError(f"infer failed after {max_retries} retries: {repr(last_err)}")


def build_samples(
    input_folder: str,
    ref_folder: str,
) -> List[Dict[str, str]]:
    input_folder = Path(input_folder)
    ref_folder = Path(ref_folder)

    image_files = [
        p.name for p in input_folder.iterdir()
        if p.is_file() and is_image_file(p.name)
    ]
    image_files.sort()

    if len(image_files) == 0:
        raise ValueError(f"输入目录没有图像: {input_folder}")

    samples = []

    for filename in image_files:
        original_image_path = input_folder / filename
        reference_image_path = ref_folder / filename

        if not reference_image_path.exists():
            logging.warning(f"跳过 {filename}：参考图不存在: {reference_image_path}")
            continue

        samples.append({
            "filename": filename,
            "cond_1": str(original_image_path),
            "cond_2": str(reference_image_path),
        })

    return samples


def main():
    output_json_path = Path(OUTPUT_JSON_PATH)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"INPUT_FOLDER={INPUT_FOLDER}")
    logging.info(f"REF_FOLDER={REF_FOLDER}")
    logging.info(f"OUTPUT_JSON_PATH={OUTPUT_JSON_PATH}")
    logging.info(f"BASE_URL={BASE_URL}")
    logging.info(f"MODEL_NAME={MODEL_NAME}")
    logging.info(f"WORKERS={WORKERS}")

    samples = build_samples(
        input_folder=INPUT_FOLDER,
        ref_folder=REF_FOLDER,
    )

    if LIMIT is not None:
        samples = samples[:LIMIT]

    logging.info(f"待处理样本数: {len(samples)}")

    client = build_client(BASE_URL, API_KEY)

    results = []
    failed = []

    def process_one(sample: Dict[str, str]) -> Tuple[bool, Dict[str, Any]]:
        filename = sample["filename"]

        try:
            generated_text = infer_one(
                client=client,
                model_name=MODEL_NAME,
                image1_path=sample["cond_1"],
                image2_path=sample["cond_2"],
                max_retries=MAX_RETRIES,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            result = {
                "cond_1": sample["cond_1"],
                "cond_2": sample["cond_2"],
                "prompt": generated_text,
            }

            return True, result

        except Exception as e:
            err = {
                "filename": filename,
                "cond_1": sample["cond_1"],
                "cond_2": sample["cond_2"],
                "error": repr(e),
            }
            return False, err

    total = len(samples)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(process_one, sample) for sample in samples]

        for idx, future in enumerate(as_completed(futures), start=1):
            ok, data = future.result()

            if ok:
                results.append(data)
                print(f"{Path(data['cond_1']).name} -> {data['prompt']}", flush=True)
            else:
                failed.append(data)
                logging.error(f"处理失败 {data['filename']}: {data['error']}")

                if not SKIP_FAILED:
                    raise RuntimeError(f"处理失败 {data['filename']}: {data['error']}")

            if idx % 20 == 0 or idx == total:
                logging.info(
                    f"已完成: {idx}/{total}, success={len(results)}, failed={len(failed)}"
                )

    results.sort(key=lambda x: Path(x["cond_1"]).name)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    if failed:
        failed_path = output_json_path.with_suffix(".failed.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=4)
        logging.warning(f"失败样本已保存到: {failed_path}")

    logging.info(f"处理完成，结果已保存到: {output_json_path}")
    logging.info(f"success={len(results)}, failed={len(failed)}")


if __name__ == "__main__":
    main()