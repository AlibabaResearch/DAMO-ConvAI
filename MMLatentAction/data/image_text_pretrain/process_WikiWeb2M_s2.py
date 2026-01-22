import copy
import os
import json
import random
import urllib.parse
import urllib.request
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging

from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("download.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置路径
BASE_DIR = Path("aburns4/WikiWeb2M")
JSON_FILES = ["pretrain-train.json", "pretrain-val.json", "pretrain-test.json"]
IMAGE_DIR = BASE_DIR / "images-86k"
TMP_DIR = Path("tmp_download")
TMP_LIST_FILE = TMP_DIR / "WikiWeb2M_image_list_AllToDownloaded.json.tmp"

# 确保目录存在
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

# 1. 读取所有 JSON，提取并去重 URL
def extract_urls():
    urls = set()
    for fname in JSON_FILES:
        path = BASE_DIR / fname
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue
        logger.info(f"Reading {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            sec_urls = item.get("section_image_url", [])
            for url in sec_urls:
                url = url.strip()  # 去除空格（如你示例中的末尾空格）
                if url and url.startswith("http"):
                    urls.add(url)
    logger.info(f"Total unique URLs extracted: {len(urls)}")
    return sorted(urls)  # 排序便于调试/复现

# 2. 保存断点列表
def save_url_list(urls):
    with open(TMP_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump(urls, f, indent=2)
    logger.info(f"URL list saved to {TMP_LIST_FILE}")

# 3. 将 URL 转为安全文件名（尽量保留可读性，避免路径过长）
def url_to_filename(url):
    # 剔除 scheme 和 domain，只保留 path + query（但太长会 hash）
    # 策略：用 SHA256 生成唯一标识 + 前缀可读部分（最多 50 chars）
    parsed = urllib.parse.urlparse(url)
    base = parsed.netloc + parsed.path + parsed.query
    # 可读前缀：取 path 最后部分（文件名），最多 50 字
    basename = os.path.basename(parsed.path)
    if not basename:
        basename = "image"
    safe_basename = "".join(c if c.isalnum() or c in "._-" else "_" for c in basename)[:50]
    # 用 SHA256 防冲突
    hash_part = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"{safe_basename}_{hash_part}"

# 4. 下载单个图片（带重试）
from PIL import Image
import requests
import io
import random
import time
import logging


def to_rgb(pil_image: Image.Image, max_size: int = 560) -> Image.Image:
    # Step 1: Convert to RGB with white background for RGBA
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        rgb_image = white_background
    else:
        rgb_image = pil_image.convert("RGB")

    # Step 2: Resize so that the longer side is at most `max_size`, preserving aspect ratio
    w, h = rgb_image.size
    if max(w, h) > max_size:
        # Compute new size preserving aspect ratio
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        rgb_image = rgb_image.resize((new_w, new_h), Image.Resampling.LANCZOS)  # or BICUBIC/BILINEAR

    return rgb_image


#
#
# def download_image(url, retries=3, delay=1):
#     file_name=url.replace("/","--")
#     filepath = IMAGE_DIR / file_name
#
#     print(url)
#     print(filepath)
#
#
#
#     # 断点续传：已存在则跳过
#     if filepath.exists():
#         return url, True, "already exists"
#
#     for attempt in range(1, retries + 1):
#         try:
#             with requests.get(url, stream=True) as response:
#                 response.raise_for_status()
#                 with io.BytesIO(response.content) as bio:
#                     img = copy.deepcopy(Image.open(bio))
#                 img=to_rgb(img)
#             # 保存为原格式（如 JPEG/PNG）
#             img.save(filepath, quality=95, format="JPEG")  # quality 对 JPEG 有效，PNG 无视
#             return url, True, "success"
#
#         except Exception as e:
#             if filepath.exists():
#                 filepath.unlink(missing_ok=True)
#             logger.warning(f"Attempt {attempt}/{retries} failed for {url}: {type(e).__name__}: {e}")
#             if attempt < retries:
#                 time.sleep(delay * (2 ** (attempt - 1)) + random.uniform(0.5, 1.0))
#
#     return url, False, f"Failed after {retries} attempts"




import asyncio
import time
import random
import io
from pathlib import Path
from PIL import Image
import logging



# 🔁 同步包装器：调用异步 playwright 函数
def download_image(url, retries=1, delay=1):
    file_name = url.replace("/", "--").replace(":", "_")
    filepath = IMAGE_DIR / file_name

    print(f"[URL] {url}")
    print(f"[Save to] {filepath}")

    # 断点续传：已存在则跳过
    if filepath.exists():
        return url, True, "already exists"

    for attempt in range(1, retries + 1):
        try:
            # 运行异步下载逻辑
            success = asyncio.run(_download_with_playwright(url, filepath))
            if success:
                return url, True, "success"
            else:
                raise RuntimeError("Image download failed (empty or invalid response)")

        except Exception as e:
            if filepath.exists():
                filepath.unlink(missing_ok=True)
            logger.warning(f"Attempt {attempt}/{retries} failed for {url}: {type(e).__name__}: {e}")
            if attempt < retries:
                backoff = delay * (2 ** (attempt - 1)) + random.uniform(0.5, 1.0)
                time.sleep(backoff)

    return url, False, f"Failed after {retries} attempts"


async def _download_with_playwright(url: str, filepath: Path) -> bool:
    from playwright.async_api import async_playwright, Route, Request, Response
    import re

    # 规范 URL（去空格）
    url = url.strip()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--disable-plugins",
            ]
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # bypass_csp=True,  # 按需开启（某些站需）
            # ignore_https_errors=True,  # 按需（测试环境）
        )
        page = await context.new_page()

        image_bytes_fut = asyncio.Future()

        # 🧩 拦截逻辑：匹配原始 url 或其重定向目标
        # 注意：Playwright route 对重定向后的最终请求依然触发
        async def intercept_route(route: Route):
            req: Request = route.request
            # 关键：匹配原始 url 本身，或最终响应 URL（应对重定向）
            if (
                    req.url == url or
                    (hasattr(req, '_redirected_from') and req._redirected_from and req._redirected_from.url == url)
            ):
                try:
                    # 继续请求，但不修改
                    response = await route.fetch()
                    if response.ok and "image" in (response.headers.get("content-type") or "").lower():
                        body = await response.body()
                        if body and len(body) > 0:
                            image_bytes_fut.set_result(body)
                        else:
                            image_bytes_fut.set_exception(ValueError("Empty image body"))
                    else:
                        image_bytes_fut.set_exception(
                            ValueError(f"Non-image or non-200 response: {response.status} {response.url}")
                        )
                    await route.continue_()
                except Exception as e:
                    image_bytes_fut.set_exception(e)
                    await route.continue_()
            else:
                await route.continue_()

        # 启用路由拦截（仅针对 http/https）
        await context.route("**/*", intercept_route)

        try:
            # 🌐 导航一个空白页后，手动触发 GET 请求（避免渲染整页）
            # 使用 page.goto + wait_until='networkidle' 最可靠
            await page.goto("https://www.google.com/", wait_until="networkidle", timeout=10000)

            # ✅ 关键：用 page.request.get 发起原生 HTTP 请求（绕过页面 JS/CORS）
            # Playwright 1.22+ 支持 context.request（更轻量）
            # fallback to page.goto if needed
            try:
                resp = await context.request.get(
                    url,
                    headers={
                        "Referer": "https://www.google.com/",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    },
                    max_redirects=5,
                    timeout=15000,
                )
                if resp.ok and "image" in (resp.headers.get("content-type") or "").lower():
                    body = await resp.body()
                    if body:
                        image_bytes_fut.set_result(body)
                    else:
                        raise ValueError("Empty response body")
                else:
                    raise ValueError(f"Non-image response: {resp.status} {resp.url}")
            except Exception as e:
                # 若 context.request 失败（如旧版 Playwright），fallback 到 route + navigate
                logger.debug(f"context.request failed, falling back to route + navigate: {e}")
                await page.goto(url, wait_until="networkidle", timeout=15000)

            # 等待 image_bytes 准备好（最多 15s）
            try:
                image_data = await asyncio.wait_for(image_bytes_fut, timeout=15.0)
            except asyncio.TimeoutError:
                raise TimeoutError("Image response not captured within timeout")

            # ✅ 安全解码：确保 BytesIO 在 PIL 使用期间不被 GC
            with io.BytesIO(image_data) as bio:
                # 强制加载到内存，避免 lazy load 关闭问题
                img = Image.open(bio)
                img.load()  # 👈 关键：强制读取像素
                img = to_rgb(img)

            # 保存（根据扩展名自动选格式）
            suffix = filepath.suffix.lower()
            save_kwargs = {}
            if suffix in {".jpg", ".jpeg"}:
                save_kwargs.update({"format": "JPEG", "quality": 95})
            elif suffix == ".png":
                save_kwargs.update({"format": "PNG"})
            else:
                save_kwargs.update({"format": "JPEG", "quality": 95})  # fallback

            img.save(filepath, **save_kwargs)
            return True

        finally:
            await page.close()
            await context.close()
            await browser.close()






# 5. 主下载函数
def main():
    # Step 1: 提取 URL 列表
    if TMP_LIST_FILE.exists():
        logger.info(f"Resuming from existing list: {TMP_LIST_FILE}")
        with open(TMP_LIST_FILE, 'r', encoding='utf-8') as f:
            urls = json.load(f)
        remain_urls=[]
        for url in urls:
            file_name = url.replace("/", "--").replace(":", "_")
            filepath = IMAGE_DIR / file_name

            if not filepath.exists():
                remain_urls.append(url)

        with open(TMP_LIST_FILE, "w", encoding="utf-8") as f:
            json.dump(remain_urls, f, indent=2)

        logger.info(f"Remain: {len(remain_urls)}")

        urls = remain_urls

    else:
        urls = extract_urls()
        save_url_list(urls)

    # Step 2: 过滤已下载（按文件存在判断）
    to_download = urls


    # Step 3: 多线程下载
    success, fail = 0, 0
    with ThreadPoolExecutor(max_workers=16) as executor:  # 可调
        future_to_url = {executor.submit(download_image, url): url for url in to_download}
        for i, future in enumerate(tqdm(as_completed(future_to_url), total=len(to_download), desc="Downloading", unit="img"), 1):
            url, ok, msg = future.result()
            if ok:
                success += 1
                logger.info(f"[{i}/{len(to_download)}] ✅ {url} → {msg}")
            else:
                fail += 1
                logger.error(f"[{i}/{len(to_download)}] ❌ {url} → {msg}")

    logger.info(f"✅ Done. Success: {success}, Fail: {fail} / Total Requested: {len(to_download)}")

if __name__ == "__main__":
    main()