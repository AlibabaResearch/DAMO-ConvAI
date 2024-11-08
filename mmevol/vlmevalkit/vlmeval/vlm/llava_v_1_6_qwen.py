import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoConfig

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,DEFAULT_IMAGE_PATCH_TOKEN)
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.model import *

from ..smp import *
from ..utils import DATASET_TYPE
from .base import BaseModel

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_mosaic_coordinate(mosaic_index, xc, yc, w, h, input_h, input_w, pix_line=0):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        h_scale,w_scale=1. * yc / h, 1. * xc / w
        min_scale = min(h_scale,w_scale)
        if h_scale<w_scale:
            gap_line=min(pix_line,int(random.uniform(0,xc-int(min_scale*w))))
            x1=xc-int(min_scale*w)-gap_line
            x2=x1+int(min_scale*w)
            y1=0
            y2=yc
        else:
            gap_line=min(pix_line,int(random.uniform(0,yc-int(min_scale*h))))
            y1=yc-int(min_scale*h)-gap_line
            y2=y1+int(min_scale*h)
            x1=0
            x2=xc
    # index1 to top right part of image
    elif mosaic_index == 1:
        h_scale,w_scale=1. * yc / h, 1. * (input_w * 2-xc) / w
        min_scale = min(h_scale,w_scale)
        if h_scale<w_scale:
            gap_line=min(pix_line,int(random.uniform(0,(input_w * 2-xc)-int(min_scale*w))))
            x1=xc+gap_line
            x2=x1+int(min_scale*w)
            y1=0
            y2=yc
        else:
            gap_line=min(pix_line,int(random.uniform(0,yc-int(min_scale*h))))
            y1=yc-int(min_scale*h)-gap_line
            y2=y1+int(min_scale*h)
            x1=xc
            x2=input_w * 2
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        h_scale,w_scale=1. * (input_h * 2-yc) / h, 1. * xc / w
        min_scale = min(h_scale,w_scale)
        if h_scale<w_scale:
            gap_line=min(pix_line,int(random.uniform(0,xc-int(min_scale*w))))
            x1=xc-int(min_scale*w)-gap_line
            x2=x1+int(min_scale*w)
            y1=yc
            y2=input_h * 2
        else:
            gap_line=min(pix_line,int(random.uniform(0,(input_h * 2-yc)-int(min_scale*h))))
            y1=yc+gap_line
            y2=y1+int(min_scale*h)
            x1=0
            x2=xc
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        h_scale,w_scale=1. * (input_h * 2-yc) / h, 1. * (input_w * 2-xc) / w
        min_scale = min(h_scale,w_scale)
        if h_scale<w_scale:
            gap_line=min(pix_line,int(random.uniform(0,(input_w * 2-xc)-int(min_scale*w))))
            x1=xc+gap_line
            x2=x1+int(min_scale*w)
            y1=yc
            y2=input_h * 2
        else:
            gap_line=min(pix_line,int(random.uniform(0,(input_h * 2-yc)-int(min_scale*h))))
            y1=yc+gap_line
            y2=y1+int(min_scale*h)
            x1=xc
            x2=input_w * 2

    return (x1, y1, x2, y2), min_scale

def concat_images_horizontally_with_margin(images, margin=10):
    """
    Concatenates images horizontally with a specified margin between images,
    padding with black if heights are not the same, and saves the result to a file.

    Parameters:
    - image_filenames: List of strings, where each string is the filepath to an image.
    - output_filename: String, the filename to save the concatenated image.
    - margin: Integer, the width of the black margin to insert between images.

    Returns:
    - None. The result is saved to 'output_filename'.
    """
    # images = [Image.open(filename) for filename in image_filenames]
    max_height = max(image.height for image in images)
    total_width = sum(image.width for image in images) + margin * (len(images) - 1)
    # Create a new image with a black background
    new_image = Image.new('RGB', (total_width, max_height), (0, 0, 0))
    
    x_offset = 0
    for image in images:
        # Calculate padding to center the image vertically
        y_offset = (max_height - image.height) // 2
        new_image.paste(image, (x_offset, y_offset))
        x_offset += image.width + margin  # Add margin after each image except the last one
    return new_image

class LLaVA_Qwen2_V(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='/mnt/workspace/checkpoints/Lin-Chen/open-llava-next-llama3-8b', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.ckpt = model_path

        print(f'load from {self.model_path}')
        # model_path = os.path.expanduser(model_path)
        # model_name = get_model_name_from_path(model_path)
        # tokenizer, model, image_processor, context_len = load_pretrained_model(
        # model_path, None, model_name)
        device_map="auto"
        kwargs = {"device_map": torch.cuda.current_device(), **kwargs}
        kwargs['torch_dtype'] = torch.float16
        # cfg_pretrained = AutoConfig.from_pretrained(model_path)
        # print(cfg_pretrained.vocab_size)
        model = LlavaQwen2ForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            # config=cfg_pretrained,
            # trust_remote_code=True,
            **kwargs
        )   
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        self.model=model
        self.model.eval()

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)

        self.image_processor = vision_tower.image_processor

        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048
        # self.image_processor = image_processor
        self.context_len=context_len
        self.square_eval=True
        if self.square_eval:
            model.config.image_grid_pinpoints = [
                [
                    672,
                    672
                ]
            ]
        self.kwargs = kwargs
        self.tokenizer = tokenizer
        torch.cuda.empty_cache()
        self.num_beams=3
        self.temperature=0.0

        self.sep="<|im_end|>"
        self.roles=("<|im_start|>user\n","<|im_start|>assistant\n")
        self.constant_system_prompt="<|im_start|>system You are a helpful assistant."
        self.options_system_prompt = self.constant_system_prompt+('Carefully read the following question and select the letter corresponding '
                                      'to the correct answer. Highlight the applicable choices without giving '
                                      'explanations.')
        self.wo_options_system_prompt = self.constant_system_prompt+'Carefully read the following question Answer the question directly.'
        self.detail_system_prompt = self.constant_system_prompt+'Answer this question in detail.'
        self.vqa_prompt = self.constant_system_prompt+'Answer the question using a single word or phrase.'
        

    def use_custom_prompt(self, dataset):
        if listinstr(['multi-choice', 'VQA'], DATASET_TYPE(dataset)):
            return True
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            return True
        return False

    def build_prompt(self, line, dataset=None):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt = ''

        question = line['question']
        if DATASET_TYPE(dataset) == 'multi-choice':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                system_prompt = self.options_system_prompt +  "\nPlease just indicate your choice."
            else:
                system_prompt = self.wo_options_system_prompt

            if 'MMMU' in dataset: # Corner Case
                prompt = system_prompt + prompt
                system_prompt = ''

        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question'] + " Yes or No?"
            prompt = question
        
        elif dataset is not None and listinstr(['OCRBench'], dataset):
            system_prompt = self.vqa_prompt
            question = line['question']
            prompt = question
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['LLaVABench'], dataset):
                system_prompt = self.vqa_prompt
                prompt = question
            elif listinstr(['MMVet'], dataset):
                system_prompt = self.detail_system_prompt
                prompt = question
            else:
                system_prompt = self.vqa_prompt
                prompt = question

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def generate_inner(self, message, dataset=None):
        if DATASET_TYPE(dataset) == 'multi-choice':
            max_new_tokens = 200+1024
        elif DATASET_TYPE(dataset) == 'Y/N':
            max_new_tokens = 3+1024
        else:
            max_new_tokens = 1024

        '''
        nums_beams = 3
        '''
        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        content = []
        if len(message)<3:
            message=[dict(type='text',value=self.constant_system_prompt)]+message
        image_files=[]
        add_signal=False
        sys_prompt=message[0]['value']
        qs=''
        ro=0
        for i in message[1:]:
            if i['type']=='text':
                if self.model.config.mm_use_im_start_end:
                    i['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                        DEFAULT_IM_END_TOKEN + '\n' + i['value']
                else:
                    i['value'] = DEFAULT_IMAGE_TOKEN + '\n' + i['value']
                qs+=i['value']
                ro+=1
            elif i['type']=='image':
                image_files.append(i['value'])

        assert ro==1

        # image_files=image_files[:1]

        # if len(image_files)>1:
        MMMU_flag='mmu' in dataset.lower()

        # sp_token=['<|top_left_subfigure|>',
        #     '<|top_right_subfigure|>','<|lower_left_subfigure|>','<|lower_right_subfigure|>',
        #     '','','']

        sp_token=['the first image', 'the second image','the third image','the fourth image', 'the fifth image']

        # sp_token=[' (top left subfigure) ',
        #     ' (top right subfigure) ',' (lower left subfigure) ',' (lower right subfigure) ',
        #     '','','']
        # if MMMU_flag:
        if MMMU_flag:
            for i,j in zip(['<image 1>', '<image 2>','<image 3>',
            '<image 4>','<image 5>'], sp_token):
                qs=qs.replace(i,j)
        else:
            qs=qs
            # qs=qs+sp_token[0]
                    

        # # if len(message)==3:
        # #     assert message[1]['type']=="image"
        # #     sys_prompt,image_file,qs=message[0]['value'],message[1]['value'],message[2]['value']
        # # else:
        # #     sys_prompt,image_file,qs=self.constant_system_prompt,message[0]['value'],message[1]['value']

        # if self.model.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
        #         DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        # print(sys_prompt,self.sep,self.roles[0],qs,self.sep,self.roles[1])

        if "Please answer yes or no." in qs:
            qs=qs.replace("Please answer yes or no.","\nAnswer the question using a single word or phrase. Check carefully to make sure your answer is correct.")
        else:
            qs=qs+"\nAnswer the question using a single word or phrase. Check carefully to make sure your answer is correct."

        prompt = sys_prompt+self.sep+self.roles[0]+qs+self.sep+self.roles[1]

        # print(prompt)

        images=[]
        image_sizes=[]
        for image_file in image_files:
            image = Image.open(image_file).convert('RGB')
            images.append(image)
            image_sizes.append(image.size)
        
        # if MMMU_flag:
        #     input_h, input_w = 336*3, 336*3
        #     point_bias=0.0
        #     base_line_width=5

        #     def w_h(img):
        #         width,height=img.size
        #         return height/width

        #     image_ratio=np.mean([w_h(img) for img in images])

        #     input_h=int(image_ratio*input_w)
        #     # yc, xc = s, s  # mosaic center x, y
        #     yc = int(random.uniform((1-point_bias) * input_h, (1+point_bias) * input_h))
        #     xc = int(random.uniform((1-point_bias) * input_w, (1+point_bias) * input_w))
        #     min_y1, max_y2, min_x1, max_x2 = float('inf'), -float('inf'), float('inf'), -float('inf')
        #     mosaic_img =Image.new("RGB", (input_w * 2, input_h * 2), (0,0,0))
        #     fake_img=Image.new("RGB", (input_w * 2, input_h * 2), (0,0,0))

        #     # images=images[:4]
        #     # if len(images)==2:
        #     #     images.append(fake_img.copy())
        #     #     images.append(fake_img.copy())
        #     # elif len(images)==3:
        #     #     images.append(fake_img.copy())

        #     bias=[[-1,-1],[1,-1],[-1,1],[1,1]]
        #     # random.shuffle(sub_data)
        #     for i_mosaic,j in enumerate(images[:4]):
        #         img=j
        #         w0,h0 = img.size # orig hw
        #         # generate output mosaic image
        #         bia=bias[i_mosaic]
        #         (l_x1, l_y1, l_x2, l_y2), scale = get_mosaic_coordinate(
        #                     i_mosaic, xc+bia[0]*base_line_width, yc+bia[1]*base_line_width, 
        #                     w0, h0, input_h, input_w
        #                 )
        #         # print(l_x1,l_y1,l_x2,l_y2)

        #         img = img.resize((int(w0 * scale), int(h0 * scale)), Image.BILINEAR)
        #         draw = ImageDraw.Draw(img)
        #         font = ImageFont.load_default()
        #         # text_width, text_height = draw.textsize(sp_token[i_mosaic], font=font)
        #         w1, h1 = img.size 
        #         # t_x = (w1 - text_width) / 2
        #         # t_y = (h1 - text_height) / 2

        #         # 在图像上添加绿色文本
        #         draw.text((10, 10), sp_token[i_mosaic], font=font, fill=(0, 255, 0))  # 绿色文字

        #         f_w, f_h = min(l_x2 - l_x1, w1), min(l_y2 - l_y1, h1)
                
        #         # Paste the image slice into the mosaic
        #         mosaic_img.paste(img, (l_x1, l_y1, l_x1 + f_w, l_y1 + f_h))
                
        #         # Update bounding coordinates
        #         min_y1, max_y2 = min(min_y1, l_y1), max(max_y2, l_y1 + f_h)
        #         min_x1, max_x2 = min(min_x1, l_x1), max(max_x2, l_x1 + f_w)

        #     # Crop the final mosaic to remove the empty margins
        #     mosaic_img = mosaic_img.crop((min_x1, min_y1, max_x2, max_y2))

        #     if len(images)==5:
        #         tmp_image=images[-1]
        #         width1,height1=mosaic_img.size
        #         width2,height2=tmp_image.size
        #         new_width=width1+width2
        #         new_height=max(height1,height2)
        #         new_image=Image.new("RGB",(new_width+10,new_height),(0,0,0))
        #         draw = ImageDraw.Draw(tmp_image)
        #         font = ImageFont.load_default()
        #         # text_width, text_height = draw.textsize(sp_token[-1], font=font)
        #         # t_x = (width2 - text_width) / 2
        #         # t_y = (height2 - text_height) / 2

        #         # 在图像上添加绿色文本
        #         draw.text((10, 10), sp_token[-1], font=font, fill=(0, 255, 0))  # 绿色文字

        #         new_image.paste(mosaic_img,(0,0))
        #         new_image.paste(tmp_image,(width1+10,0))

        #         mosaic_img=new_image

        #     images=[mosaic_img]
        #     image_sizes=[mosaic_img.size]

        if len(images)>1:
            # MMMU or BLINK
            images=[concat_images_horizontally_with_margin(images)]
            image_sizes=[images[0].size]

        images=[images[0]]
        image_sizes=[image_sizes[0]]


        image_tensors = process_images(
            images, self.image_processor, self.model.config)

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')[None,]
        
        input_ids = input_ids.to(device=torch.cuda.current_device(), non_blocking=True)

        # image_tensor=image_tensor[None,]
        image_tensors=image_tensors.to(
                    dtype=torch.float16, device=torch.cuda.current_device(), non_blocking=True)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=0,
                num_beams=self.num_beams,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True)
        res= self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)[0].strip()
        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]
        # print(f"content: {prompt}, res: {res}")
        return res
