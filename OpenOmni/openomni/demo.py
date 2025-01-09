# 保存临时音频、图像文件的路径
TEMP_FILES_PATH = "assets"
from openomni.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from openomni.conversation import conv_templates, SeparatorStyle
from openomni.mm_utils import process_images
from openomni.model.builder import load_pretrained_qwen_model
from openomni.utils import disable_torch_init
from torch.utils.data import Dataset, DataLoader
from openomni.flow_inference import AudioDecoder

import gradio as gr
from gradio_multimodalchatbot import MultimodalChatbot
from gradio.data_classes import FileData
import sounddevice as sd
import time
import numpy as np
import wave
from PIL import Image
import os
import base64

def ctc_postprocess(tokens, blank):
    _toks = tokens.squeeze(0).tolist()
    print(_toks,len(_toks))
    # deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    deduplicated_toks = [v for i, v in enumerate(_toks)]
    hyp = [v for v in deduplicated_toks if v != blank]
    hyp = " ".join(list(map(str, hyp)))
    return hyp


disable_torch_init()

model_path=""
voice_config_path=""
flow_ckpt_path=""
hift_ckpt_path=""

assert len(model_path)>0 and len(voice_config_path)>0 and len(flow_ckpt_path)>0 and len(hift_ckpt_path)>0, "set model path first, refer to inference.py for detail args"

audio_decoder = AudioDecoder(config_path=voice_config_path, 
                            flow_ckpt_path=flow_ckpt_path,
                            hift_ckpt_path=hift_ckpt_path,
                            device='cuda')

model_path = os.path.expanduser(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_qwen_model(model_path, "")
tokenizer.add_tokens(["<image>"], special_tokens=True)
tokenizer.add_tokens(["<speech>"], special_tokens=True)
tokenizer.chat_template="{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

image_token_index = tokenizer.convert_tokens_to_ids("<image>")
speech_token_index = tokenizer.convert_tokens_to_ids("<speech>")


model.config.image_grid_pinpoints = [
    [
        672,
        672
    ]
]

if not os.path.exists(TEMP_FILES_PATH):
    os.makedirs(TEMP_FILES_PATH)

last_input_text = ""
last_input_audio = None
last_upload_image = None
temp_files = []

def get_demo_audio():
    # 生成一个简单的音频响应（例如，生成一个1秒的正弦波）
    fs = 44100  # 采样率
    duration = 1  # 持续时间
    frequency = 440  # 频率
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    audio_response = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio_response, fs

def save_audio(audio, filename):
    # 将音频数据保存到一个WAV文件中
    filename = os.path.join(TEMP_FILES_PATH, filename)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16位音频
        wf.setframerate(audio[0])  # 采样率
        wf.writeframes(audio[1].tobytes())  # 写入音频数据
    temp_files.append(filename)
    return filename

def save_image(image_array, filename):
    # 将 gr.Image 返回的 ndarray 保存到文件中
    filename = os.path.join(TEMP_FILES_PATH, filename)
    image = Image.fromarray(image_array)
    image.save(filename)    
    temp_files.append(filename)
    return filename

def clean_temp_files():
    for file in temp_files:
        print(f"removing {file}")
        os.remove(file)
    temp_files.clear()

def get_user_msg(input_text, input_audio, upload_image):
    user_msg = {"text": input_text, "files": []}

    if input_audio is not None:
        audio_file = save_audio(input_audio, f"audio_{time.time()}.wav")
        print(f"audio_file={audio_file}")
        user_msg["files"].append( {"file": FileData(path=audio_file)} )

    if upload_image is not None:
        image_file = save_image(upload_image, f"image_{time.time()}.jpg")
        print(f"image_file={image_file}")
        user_msg["files"].append( {"file": FileData(path=image_file)} ) 

    return user_msg

def get_bot_msg(text_response, audio_response, fs):
    bot_msg = {"text": text_response, "files": []}

    if audio_response is not None:
        audio_file = save_audio((fs, audio_response), f"resp_audio_{time.time()}.wav")
        print(f"resp_audio_file={audio_file}")
        bot_msg["files"].append( {"file": FileData(path=audio_file)} )

    return bot_msg

# 处理点击 chat 按钮的响应
def chat_response(input_text, input_audio, upload_image, temperature, top_p, max_output_tokens, history):
    # 保存最近一次输入的数据（用于 regenerate）
    global last_input_text, last_input_audio, last_upload_image
    last_input_text = input_text
    last_input_audio = input_audio
    last_upload_image = upload_image

    # 保存对话记录
    usr_msg = get_user_msg(input_text, input_audio, upload_image)

    system_message= "You are a helpful language, vision and speech assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language or speech."

    speech_file = "./assets/question.wav"
    image_file="./assets/example.png"

    prompt=[]
    image=[]
    speechs=[]
    for (usr,bot) in history:
        usr_turn={'role':'user','content':usr['text']}
        bot_turn={'role':'assistant','content': bot['text']}
        if len(usr['files'])>0:
            for i in usr['files']:
                if i['file'].endswith('.jpg'):
                    images.append(i['file'])
                elif i['file'].endswith('.wav'):
                    speechs.append(i['file'])
                    usr_turn['content']="<speech>\n Please answer the questions in the user's input speech"
        prompt.append(usr_turn)
        prompt.append(bot_turn)
    usr_turn={'role':'user','content':usr_msg['text']}
    if len(usr_msg['files'])>0:
        for i in usr_msg['files']:
            if i['file'].endswith('.jpg'):
                images.append(i['file'])
            elif i['file'].endswith('.wav'):
                speechs.append(i['file'])
                usr_turn['content']="<speech>\n Please answer the questions in the user's input speech"
    prompt.append(usr_turn)

    if len(images)>0:
        prompt[0]['content']='<image> \n'+prompt[0]['content']
        image_file=images[-1]

    if len(speechs)==0:
        speechs=[speech_file]
    
    ##############################################################################################!!!
    # 生成文本回复和音频回复的部分
    input_id = tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}]+prompt,
    add_generation_prompt=True)
    # input_id += encode_id
    for idx, encode_id in enumerate(input_id):
        if encode_id == image_token_index:
            input_id[idx] = IMAGE_TOKEN_INDEX
        if encode_id == speech_token_index:
            input_id[idx] = SPEECH_TOKEN_INDEX

    input_ids = torch.tensor([input_id], dtype=torch.long)
    input_ids = input_ids.to(device='cuda', non_blocking=True)

    image = Image.open(image_file).convert('RGB')
    image_tensor = process_images(
        [image], image_processor, model.config)[0]
    speech_tensors=[]
    speech_lengths=[]
    for speech_file in speechs:
        speech = whisper.load_audio(speech_file)
        speech = whisper.pad_or_trim(speech)
        speech_tensor = whisper.log_mel_spectrogram(speech, n_mels=args.mel_size).permute(1, 0)
        speech_tensors.append(speech_tensor)
        speech_lengths.append(speech_tensor.shape[0])
    speech_tensors = torch.stack(speech_tensors).to(dtype=torch.float16, device=torch.cuda.current_device(), non_blocking=True)
    speech_lengths = torch.LongTensor(speech_lengths).to(device=torch.cuda.current_device(), non_blocking=True)
    with torch.inference_mode():
        time1=time.time()
        outputs = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            speech=speech_tensors,
            speech_lengths=speech_lengths,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            streaming_unit_gen=False,
            faster_infer=False
        )
        time2=time.time()
        output_ids, output_units = outputs

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if args.s2s:
        if model.config.speech_generator_type=="ar":
            output_units=output_units
        elif model.config..speech_generator_type=="ctc":
            output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)
    
    tts_speechs=[]

    audio_tokens=[int(x) for x in output_units.split(' ')]

    tts_token = torch.tensor(audio_tokens, device='cuda').unsqueeze(0)

    tts_speech = audio_decoder.offline_inference(tts_token)

    tts_speechs.append(tts_speech.squeeze())

    tts_speech = torch.cat(tts_speechs, dim=-1).cpu()

    ##############################################################################################!!!

    bot_msg = get_bot_msg(text_response, ts_speech.unsqueeze(0), 22050)

    history.append([usr_msg, bot_msg])

  
    return history, history, "", None, None

# 处理点击 regenerate 按钮的响应
def regenerate_response(history):
    # 删除 history 中的最后一项
    if len(history) > 0:
        history.pop()
        return chat_response(last_input_text, last_input_audio, last_upload_image, temperature, top_p, max_output_tokens, history)
    else:
        return history, history, "", None, None


# 处理点击 clear 按钮的响应
def clean_history():
    global last_input_text, last_input_audio, last_upload_image
    last_input_text = ""
    last_input_audio = None
    last_upload_image = None

    clean_temp_files()

    return [], "", None, None

with gr.Blocks() as demo:
    # 添加Logo和标题
    with gr.Row():
        with open(".assets/logo.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            gr.Markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{encoded_string}" alt="Logo" width="100" height="100">
                <h1>OpenOmni</h1>
            </div>
            """)
        
    # 布局调整，使界面更加方正
    with gr.Row():
        with gr.Column(scale=1):
            # 图片上传
            upload_image = gr.Image(height=224, width=224, label="Image")
            # 模型参数设置
            temperature = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.5, label="Temperature", interactive=True)
            top_p = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.5, label="Top P", interactive=True)
            max_output_tokens = gr.Slider(minimum=512, maximum=16384, step=256, value=2048, label="Max Output Tokens", interactive=True)
        with gr.Column(scale=2):
            # 对话记录
            chatbot = MultimodalChatbot(height=600)
            # 文本输入 & 音频输入
            input_text = gr.Textbox(lines=5, label="Input Text")
            input_audio = gr.Audio(label="Audio")
            # 按钮
            with gr.Row():
                chat = gr.Button("Chat")
                regenerate = gr.Button("Regenerate")
                clear = gr.Button("Clear")

    chat.click(
        fn=chat_response,
        inputs=[input_text, input_audio, upload_image, temperature, top_p, max_output_tokens, chatbot],
        outputs=[chatbot, chatbot, input_text, input_audio, upload_image]
    )

    regenerate.click(
        fn=regenerate_response,
        inputs=[chatbot],
        outputs=[chatbot, chatbot, input_text, input_audio, upload_image]
    )

    clear.click(
        fn=clean_history,
        inputs=[],
        outputs=[chatbot, input_text, input_audio, upload_image]
    )

demo.launch()
