import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm


def process_MM_data():
    # 1: Process image caption dataset
    image_path_1 = "laion/conceptual-captions-12m-webdataset/images"
    main_json_path_1 = "laion/conceptual-captions-12m-webdataset/image_caption-100000.json"
    with open(main_json_path_1, "r", encoding="utf-8") as f:
        main_json_1 = json.load(f)

    ImageCaption_MM_instances = []
    for instance in main_json_1:
        image_path = "{}/{}".format(image_path_1, instance["image_path"])

        MM_instance = {
            "id": instance["id"],
            "images": [image_path],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Please observe the image and generate a caption relevant to it."
                },
                {
                    "role": "assistant",
                    "content": instance["txt"]
                }
            ],
            "instance_type": "image_caption",
        }
        ImageCaption_MM_instances.append(MM_instance)

    random.seed(42)
    random.shuffle(ImageCaption_MM_instances)
    ImageCaption_MM_instances=ImageCaption_MM_instances[:20000]


    # 2: Process MM News dataset
    image_path_2 = "N24News/images"
    main_json_path_2 = "N24News/news/nytimes.json"
    with open(main_json_path_2, "r", encoding="utf-8") as f:
        main_json_2 = json.load(f)

    MMNews_MM_instances = []
    for instance in main_json_2:
        image_path = "{}/{}.jpg".format(image_path_2, instance["image_id"])
        MM_instance = {
            "id": instance["article_id"],
            "images": [image_path],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Complete the news."
                },
                {
                    "role": "assistant",
                    "content": instance["article"]
                }
            ],
            "instance_type": "mm_news",

        }
        MMNews_MM_instances.append(MM_instance)

    random.seed(42)
    random.shuffle(MMNews_MM_instances)
    MMNews_MM_instances=MMNews_MM_instances[:50000]



    # 3: Process MM WikiPages dataset
    image_path_3 = "aburns4/WikiWeb2M/images-86k"
    main_json_path_3 = "aburns4/WikiWeb2M/pretrain-test.json"
    with open(main_json_path_3, "r", encoding="utf-8") as f:
        main_json_3 = json.load(f)


    MMWiki_MM_instances = []
    for instance in main_json_3:
        try:
            section_image_url = instance["section_image_url"][0]
            section_image_url = section_image_url.replace("/", "--").replace(":", "_")
        except:
            continue
        image_path = "{}/{}".format(image_path_3, section_image_url)
        if os.path.exists(image_path):
            section_text = "\n\n".join(instance["section_text"])
            MM_instance = {
                "id": instance["page_url"],
                "images": [image_path],
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": "Complete the wiki pages."
                    },
                    {
                        "role": "assistant",
                        "content": section_text
                    }
                ],
                "instance_type": "mm_wiki_pages",

            }
            MMWiki_MM_instances.append(MM_instance)

    random.seed(42)
    random.shuffle(MMWiki_MM_instances)
    MMWiki_MM_instances=MMWiki_MM_instances[:50000]


    # 4: Process MM Conversation dataset
    image_path_4 = "lmms-lab/LLaVA-NeXT-Data/images"
    main_json_path_4 = "lmms-lab/LLaVA-NeXT-Data/conversations-100000.json"
    with open(main_json_path_4, "r", encoding="utf-8") as f:
        main_json_4 = json.load(f)

    MMConv_MM_instances = []
    for instance in main_json_4:
        image_path = "{}/{}".format(image_path_4, instance["image_path"])

        conversations = instance["conversations"]
        conv_messages = []
        for conv_item in conversations:
            if conv_item["from"] == "human":
                message = {
                    "role": "user",
                    "content": conv_item["value"]
                }
            elif conv_item["from"] == "gpt":
                message = {
                    "role": "assistant",
                    "content": conv_item["value"]
                }
            conv_messages.append(message)

        sys_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages = sys_messages + conv_messages

        MM_instance = {
            "id": instance["id"],
            "images": [image_path],
            "messages": messages,
            "instance_type": "mm_conversation",
        }
        MMConv_MM_instances.append(MM_instance)
    # MMConv_MM_instances=MMConv_MM_instances[:100000]


    print(f"The number of ImageCaption_MM_instances: {len(ImageCaption_MM_instances)}")
    print(f"The number of MMNews_MM_instances: {len(MMNews_MM_instances)}")
    print(f"The number of MMWiki_MM_instances: {len(MMWiki_MM_instances)}")
    print(f"The number of MMConv_MM_instances: {len(MMConv_MM_instances)}")

    all_MM_instances = ImageCaption_MM_instances + MMNews_MM_instances + MMWiki_MM_instances + MMConv_MM_instances
    random.shuffle(all_MM_instances)

    target_dir = TARGET_DIR
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    target_path = f"{target_dir}/MM_pretrain.json"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(all_MM_instances, f, indent=2)


import os
import json
import zstandard as zstd
from pathlib import Path
from typing import Iterator, Optional, List


def stream_slimpajama_test(
        data_dir: str = "cerebras/SlimPajama-627B/test",
        max_samples: Optional[int] = None,
) -> Iterator[dict]:
    """
    流式读取 SlimPajama test split（.jsonl.zst 分块压缩）

    Args:
        data_dir: test 子目录路径（包含 chunk-*.jsonl.zst）
        max_samples: 最多返回多少条样本（None = 全读）
        skip_samples: 跳过前 N 条
        shuffle_chunks: 是否打乱 chunk 顺序（增加样本多样性）
        chunk_limit: 仅读前 K 个 chunk（调试用）
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Test dir not found: {data_path}")

    zst_files = []
    for chunk_id in range(1, 6):  # chunk1 to chunk5
        chunk_dir = data_path / f"chunk{chunk_id}"
        if chunk_dir.is_dir():
            zst_files.extend(chunk_dir.glob("*.jsonl.zst"))

    zst_files = sorted(zst_files)  # 保证可复现性
    if not zst_files:
        raise FileNotFoundError(f"No .jsonl.zst files found in {data_path}/chunk1~chunk5")

    sample_count = 0

    # 遍历每个 chunk 文件
    for zst_file in zst_files:
        try:
            dctx = zstd.ZstdDecompressor()
            with open(zst_file, "rb") as f:
                with dctx.stream_reader(f) as reader:
                    # 按行读取解压后内容（UTF-8）
                    buffer = b""
                    while True:
                        chunk = reader.read(8192)  # 8KB buffer
                        if not chunk:
                            break
                        buffer += chunk

                        # 按换行符分割（处理跨 chunk 边界的行）
                        lines = buffer.split(b"\n")
                        buffer = lines[-1]  # 保留不完整行

                        for line_bytes in lines[:-1]:
                            if not line_bytes.strip():
                                continue

                            # 解码并解析 JSON
                            try:
                                line = line_bytes.decode("utf-8", errors="ignore").strip()
                                if not line:
                                    continue
                                instance = json.loads(line)

                                # 构造你的 Text_instance
                                yield {
                                    "id": f"General_{sample_count}",
                                    "messages": [
                                        {"role": "system", "content": ""},
                                        {"role": "user", "content": "The following is a text."},
                                        {"role": "assistant", "content": instance["text"]}
                                    ],
                                    "instance_type": "General",
                                }

                                sample_count += 1
                                if max_samples is not None and sample_count >= max_samples:
                                    return  # 提前退出所有循环

                            except Exception as e:
                                # 容错：跳过损坏行，记录警告（可选写入日志）
                                print(f"⚠️ Skip invalid line in {zst_file.name}: {e}")
                                continue

            # 处理 buffer 中剩余的最后一行（可能不完整，丢弃）
            if buffer.strip():
                print(f"⚠️ Incomplete last line in {zst_file.name}, skipped.")

        except Exception as e:
            print(f"❌ Error processing {zst_file}: {e}")
            continue


def process_TextOnly_data():
    main_json_path_1 = "nvidia/HelpSteer3"
    HelpSteer3_data = load_dataset(main_json_path_1, split=f"train")

    HelpSteer3_Text_instances = []
    for index, instance in enumerate(HelpSteer3_data):
        overall_preference = instance["overall_preference"]
        if overall_preference >= 0:
            chosen_response = instance["response2"]
        else:
            chosen_response = instance["response1"]

        conv_messages = instance["context"] + [{
            "role": "assistant",
            "content": chosen_response
        }]

        sys_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages = sys_messages + conv_messages

        Text_instance = {
            "id": f"HelpSteer3_{index}",
            "messages": messages,
            "instance_type": "HelpSteer3",
        }
        HelpSteer3_Text_instances.append(Text_instance)


    General_Text_instances = []
    max_samples = 300000
    for inst in tqdm(stream_slimpajama_test(max_samples=max_samples), total=max_samples):
        General_Text_instances.append(inst)

    ################################################################
    print(f"The number of HelpSteer3_Text_instances: {len(HelpSteer3_Text_instances)}")
    print(f"The number of General_Text_instances: {len(General_Text_instances)}")

    all_Text_instances = HelpSteer3_Text_instances +General_Text_instances
    random.shuffle(all_Text_instances)

    target_dir = TARGET_DIR
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    target_path = f"{target_dir}/TextOnly_pretrain.json"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(all_Text_instances, f, indent=2)


if __name__ == '__main__':
    TARGET_DIR = "v12"
    process_MM_data()
    process_TextOnly_data()