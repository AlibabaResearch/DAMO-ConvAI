import os
import json
import glob
from tqdm import tqdm

img_folder = "YanqiDai/MMRole_dataset/images"


# TODO: fist ID, then OOD

profile_folder = "YanqiDai/MMRole_dataset/profiles/in-distribution/detailed_profiles"
input_dir = "YanqiDai/MMRole_dataset/dialogues/in-distribution/comment"
tag = "YanqiDai/MMRole_dataset"
saved_conv_dir = f"{tag}"
saved_conv_path = f"{saved_conv_dir}/conversations-train-comment.json"
json_files = glob.glob(os.path.join(input_dir, "*.json"))



# profile_folder = "YanqiDai/MMRole_dataset/profiles/out-of-distribution/detailed_profiles"
# input_json = "YanqiDai/MMRole_dataset/dialogues/out-of-distribution/comment.json"
# tag = "YanqiDai/MMRole_dataset"
# saved_conv_dir = f"{tag}"
# saved_conv_path = f"{saved_conv_dir}/conversations-OODtest-comment.json"
# json_files = [input_json]


os.makedirs(saved_conv_dir, exist_ok=True)

# 获取所有 JSON 文件
print(json_files)

all_conversations = []

for file_path in tqdm(json_files, desc="Processing JSON files"):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for instance in data:
        original_id = instance["id"]
        image_path = instance["image"]  # 保持原始 image 路径不变

        # 构造完整的图像路径
        full_img_path = os.path.join(img_folder, image_path)

        # 检查图像是否存在
        if not os.path.exists(full_img_path):
            print(full_img_path)
            continue  # 跳过不存在图像的样本

        new_convs = []
        original_roles = []  # 新增：记录每轮原始角色
        assistant_responses = []

        conversations=instance["conversations"]
        if len(conversations)==0:
            continue

        character_role= None
        for turn in conversations:
            try:
                orig_role = turn["role"]
            except:
                orig_role = turn["from"]

            original_roles.append(orig_role)

            if orig_role == "human":
                new_role = "user"
            else:
                new_role = "assistant"
                assistant_responses.append(turn["value"].strip())
                character_role = orig_role

            new_convs.append({
                "role": new_role,
                "content": turn["value"]
            })

        full_text = " ".join(assistant_responses) + " "

        character_profile=None
        if character_role is not None:
            _character_role=character_role.replace(" ","_")
            profile_path = os.path.join(profile_folder, f"{_character_role}.json")
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    character_profile = json.load(f)

                new_entry = {
                    "id": original_id,
                    "image_path": image_path,
                    "conversations": new_convs,
                    "original_roles": original_roles,  # 新增字段
                    "character_role": character_role,
                    "character_profile": {
                        character_role: character_profile,
                    },
                    "text": full_text,
                }

                all_conversations.append(new_entry)
            except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                print(f"Warning: Failed to load character profile from {profile_path}: {e}")
                character_profile = None  # 显式保留 None 或根据需求设默认值




# 保存为单个 JSON 文件
with open(saved_conv_path, 'w', encoding='utf-8') as out_f:
    json.dump(all_conversations, out_f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(all_conversations)} conversations to {saved_conv_path}")