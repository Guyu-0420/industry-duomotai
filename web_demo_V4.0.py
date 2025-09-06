#!/usr/bin/env python
# encoding: utf-8
import torch
import argparse
from transformers import AutoModel, AutoTokenizer
import gradio as gr
from PIL import Image
from decord import VideoReader, cpu
import io
import os
import copy
import requests
import base64
import json
import traceback
import re
import modelscope_studio as mgr


# README, How to run demo on different devices

# For Nvidia GPUs.
# python web_demo_2.6.py --device cuda

# For Mac with MPS (Apple silicon or AMD GPUs).
# PYTORCH_ENABLE_MPS_FALLBACK=1 python web_demo_2.6.py --device mps

# Argparser
parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--device', type=str, default='cuda', help='cuda or mps')
parser.add_argument('--multi-gpus', action='store_true', default=False, help='use multi-gpus')
args = parser.parse_args()
device = args.device
assert device in ['cuda', 'mps']


def call_silicon_flow_api(text="who are you", image_data_list=None, custom_rules=None):
    """
    调用硅基流动 API 进行分析。
    :param text: 用户输入的文本内容
    :param image_data: 图片或视频的二进制数据（可选）
    :param custom_rules: 自定义规则（可选）
    :return: API 返回的分析结果
    """
    # API 的 URL 和认证信息
    url = "https://api.siliconflow.cn/v1/chat/completions"
    api_key = "sk-nywsipqlhdqhhbosxzchjjtkvnozcgqvsdijqnvuyjwxnbyz"
    # 请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # 请求体
    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",  # 模型可更换
        "messages": [
            {
                "role": "user",
                "content": []
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5
    }
    print("Debug: API Request Payload =", payload)
    # 添加文本内容
    if text:
        payload["messages"][0]["content"].append({"type": "text", "text": text}) # 加入提示信息
        print(payload["messages"][0]["content"])
        
        if image_data_list:
            for image_data in image_data_list:
                payload["messages"][0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                })

    # 如果有自定义规则，则添加到请求体中
    if custom_rules:
        payload["custom_rules"] = custom_rules

    # 发送 POST 请求
    try:
        response = requests.post(url, json=payload, headers=headers) 
        
        # 检查响应状态码
        if response.status_code == 200:
            return response.json()                  
        else:
            # 提取错误信息
            try:
                error_info = response.json()
                print(f"API 调用失败，状态码：{response.status_code}")
                print(f"错误信息：{error_info}")
                return {"error": error_info}  # 返回错误信息
            except json.JSONDecodeError:
                print(f"API 调用失败，状态码：{response.status_code}")
                print(f"原始错误信息：{response.text}")
                return {"error": {"code": response.status_code, "message": response.text}}
    except requests.exceptions.RequestException as e:
        print("API 请求失败:", e)
        return {"error": {"code": -1, "message": str(e)}}



ERROR_MSG = "出现错误, 请重试"
model_name = '多场景通用大模型V4.0-演示版'
MAX_NUM_FRAMES = 64
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm', '.m4v'}
predefine_msgs = '''
        【任务描述】
        根据图片内容进行安全隐患分析，并按照以下格式输出结果。

        首先，请识别图片中的主要场景类型。如果符合下列任一场景，则使用对应的指令进行分析：
        "工厂": "请检查是否佩戴了安全帽，以及是否存在其他工厂常见隐患。",
        "办公室": "请检查是否存在打瞌睡，跷二郎腿，以及其他办公场景常见隐患。",
        "家中": "请检查是否存在老人摔倒，烟雾缭绕等家中常见隐患。",
        "野生动物园": "请检查是否出现参观人员不按要求进入野生动物活动区域等隐患。",
        "交通规则": "请检查是否大货车、大巴车等在限制路段出现的情况",
        "街道": "请检查是否有翻越围栏，闯红绿灯等安全隐患",
        "人员筛查": "请检查是否存在穿着白色衣服的女人",
        "商场": "请检查是否有携带刀具或其他危险用具等安全隐患。",
        "通用": "请检查是否存在常见的安全隐患。"

        【示例输出】（输出结果不包含任务描述和自定义规则，仅按示例进行输出）
        若存在安全隐患，则输出：
        "是否存在安全隐患: 是\n不安全生产因素: \n1. 老人摔倒在地\n2. ...\n建议措施:\n1. 扶起老人，观察是否有受伤情况\n2. 检查家里是否有不平的地面、地面上有水等容易摔倒的隐患，及时消除\n3. ..."
        (安全隐患和措施一一对应 都不超过三条)
        若不存在安全隐患，则输出：
        "是否存在安全隐患: 否\n当前场景不存在安全隐患" 

        【自定义规则】
        此外还需考虑以下规则（如果有）：'''


def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

def is_image(filename):
    return get_file_extension(filename) in IMAGE_EXTENSIONS

def is_video(filename):
    return get_file_extension(filename) in VIDEO_EXTENSIONS


form_radio = {
    'choices': ['Beam Search', 'Sampling'],
    #'value': 'Beam Search',
    'value': 'Sampling',
    'interactive': True,
    'label': 'Decode Type'
}


def create_component(params, comp='Slider'):
    if comp == 'Slider':
        return gr.Slider(
            minimum=params['minimum'],
            maximum=params['maximum'],
            value=params['value'],
            step=params['step'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Radio':
        return gr.Radio(
            choices=params['choices'],
            value=params['value'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Button':
        return gr.Button(
            value=params['value'],
            interactive=True
        )
    elif comp == 'UploadButton':
        return gr.UploadButton(
            label=params.get('label','上传文件'),
            file_types=params.get('file_types',[]),
            file_count=params.get('file_count','single')
        )


def create_multimodal_input(upload_image_disabled=False, upload_video_disabled=False):
    return mgr.MultimodalInput(upload_image_button_props={'label': '图片输入源', 'disabled': upload_image_disabled, 'file_count': 'multiple', 'visible':False}, 
                                        upload_video_button_props={'label': '视频输入源', 'disabled': upload_video_disabled, 'file_count': 'single', 'visible':False},
                                        submit_button_props={'label': '提交'})



def chat(img_list, msgs, ctx, params=None, vision_hidden_states=None):
    try:        
        # 提取图片数据（如果有）
        image_data_list = []
        if img_list:  # img_list 是一个包含多张图片路径或数据的列表
            print("encode image for api=",img_list)
            for img in img_list:
                # 调用 encode_image_for_api 处理每张图片
                image_data = encode_image_for_api(img)
                image_data_list.append(image_data)

        user_input_text = ""
        for msg in msgs:
            if msg["role"] == "user":
                for item in msg["content"]:
                    if isinstance(item, str):  # 如果是纯文本
                        user_input_text += item + " "
        final_prompt = f"{predefine_msgs}\n\n{user_input_text}"
        print("提示工程=",final_prompt)

        print("=== 唯一一次调用：使用 msgs 对话历史 ===")
        # 调用硅基流动 API
        if len(msgs[0]['content']) > 1:
            msg_input = predefine_msgs + msgs[0]['content'][1]
        else:
            msg_input = final_prompt + "请输出图片中的安全隐患"

        # msg_input = "危险区域必须设置明显警示标志"  # 调试 发现之前user_text没有录入
        api_result = call_silicon_flow_api(msg_input, image_data_list)
        if api_result is None:
            return -1, ERROR_MSG, None, None

        # 解析 API 返回的结果
        answer = api_result.get("choices", [{}])[0].get("message", {}).get("content", ERROR_MSG)
        return 0, answer, None, None
    except Exception as e:
        print('调用硅基流动 API 时出现错误:', e)
        traceback.print_exc()
        #print("Debug: User Text =",msgs)
        #print("Debug: Image Data (first 100 bytes) =", image_data[:100] if image_data else None)
        return -1, ERROR_MSG, None, None   


def convert_to_jpeg(image_path):
    """
    将图片转换为 JPEG 格式并保存为临时文件。
    :param image_path: 原始图片文件路径
    :return: 转换后的 JPEG 文件路径
    """
    try:
        # 打开图片
        img = Image.open(image_path)
        
        # 转换为 RGB 模式（如果图片不是 RGB 模式）
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # 构造新的文件路径（以 .jpeg 结尾）
        temp_jpeg_path = os.path.splitext(image_path)[0] + ".jpeg"
        
        # 保存为 JPEG 格式
        img.save(temp_jpeg_path, format="JPEG")
        
        # print(f"Debug: Converted image to JPEG and saved at {temp_jpeg_path}")
        return temp_jpeg_path
    except Exception as e:
        # print(f"Error converting image to JPEG: {e}")
        traceback.print_exc()
        return None

def encode_image(image):
    if not isinstance(image, Image.Image):
        if hasattr(image, 'path'):
            image = Image.open(image.path).convert("RGB")
        else:
            image = Image.open(image.file.path).convert("RGB")
    # resize to max_size
    max_size = 448
    if max(image.size) > max_size:
        w,h = image.size
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    return image

# def encode_image_for_api(image_file):
#     print("我是=", image_file) 
#     if hasattr(image_file, "path") and isinstance(image_file.path, str):
#         image_file = image_file.path
#     """
#     将图片文件转换为Base64 格式传递给 API
#     """
#     # # 如果图片不是 JPEG 格式，先转换为 JPEG
#     # if not image_file.lower().endswith((".jpg", ".jpeg")):
#     #     image_file = convert_to_jpeg(image_file)
#     #     if not image_file:
#     #         raise ValueError("Failed to convert image to JPEG format.")
                
#     with open(image_file, "rb") as f:
#         image_data = f.read()
#     image_data = base64.b64encode(image_data).decode("utf-8") 
#     return image_data

def encode_image_for_api(image_file):
    print("我是=", image_file) 
    if hasattr(image_file, "path") and isinstance(image_file.path, str):
        image_file = image_file.path
    """
    将图片文件转换为Base64 格式传递给 API
    """
    # 打开图片
    with Image.open(image_file) as img:
        # 如果是 RGBA 模式，转为 RGB（避免保存为 JPEG 出错）
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 调整图片大小为 360x480
        img = img.resize((360, 480), Image.Resampling.LANCZOS)

        # 将图片保存到内存中（BytesIO）
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format if img.format else 'JPEG')  # 保留原格式或使用 JPEG
        image_data = img_byte_arr.getvalue()

    # 编码为 Base64
    encoded_str = base64.b64encode(image_data).decode("utf-8")
    return encoded_str

def encode_video_for_api(video_file):
    """
    提取视频的第一帧并转换为Base64 格式传递给 API
    """
    def get_first_frame(video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        frame = vr[0].asnumpy()
        return Image.fromarray(frame.astype('uint8'))
    
    first_frame = get_first_frame(video_file)
    buffered = io.BytesIO()
    first_frame.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def check_mm_type(mm_file):
    if hasattr(mm_file, 'path'):
        path = mm_file.path
    else:
        path = mm_file.file.path
    if is_image(path):
        return "image"
    if is_video(path):
        return "video"
    return None


def encode_mm_file(mm_file):
    if check_mm_type(mm_file) == 'image':
        return [encode_image_for_api(mm_file)]  # 确保返回字节对象
    if check_mm_type(mm_file) == 'video':
        return [encode_video_for_api(mm_file)]  # 确保返回字节对象
    return None

def make_text(text):
    #return {"type": "text", "pairs": text} # # For remote call
    return text

def encode_message(_question):
    print("woshiencode-message = ", _question)
    files = _question.files
    question = _question.text
    pattern = r"\[mm_media\]\d+\[/mm_media\]"
    matches = re.split(pattern, question)
    message = []
    if len(matches) != len(files) + 1:
        gr.Warning("Number of Images not match the placeholder in text, please refresh the page to restart!")
    assert len(matches) == len(files) + 1

    text = matches[0].strip()
    if text:
        message.append(make_text(text))
    for i in range(len(files)):
        message += encode_mm_file(files[i])
        text = matches[i + 1].strip()
        if text:
            message.append(make_text(text))
            print("Debug: Encoded Message =", message)
    return message


def check_has_videos(_question):
    images_cnt = 0
    videos_cnt = 0
    for file in _question.files:
        if check_mm_type(file) == "image":
            images_cnt += 1 
        else:
            videos_cnt += 1
    return images_cnt, videos_cnt 


def count_video_frames(_context):
    num_frames = 0
    for message in _context:
        for item in message["content"]:
            #if item["type"] == "image": # For remote call
            if isinstance(item, Image.Image):
                num_frames += 1
    return num_frames



def respond(_question, _chat_bot, _app_cfg, params_form, file_state, image_display_area):
    print("respond里面的question文本 =", _question.text)
    print("respond里面的文件路径 =", [getattr(file, 'path', getattr(file, 'name', None)) for file in _question.files])
    # user_image = _question.files[0].path # 获取文件路径
    uploaded_images = []  # file_state是左边的文件储存地
    if isinstance(file_state, list):
        for file_path in file_state:
            uploaded_images.append(file_path)

    _context = _app_cfg['ctx'].copy()
    _context.append({'role': 'user', 'content': encode_message(_question)})
    # print(" _context =", _context)

    images_cnt = _app_cfg['images_cnt']
    videos_cnt = _app_cfg['videos_cnt']
    files_cnts = check_has_videos(_question)
    # print("Debug: _question.text =", _question.text)
    # print("Debug: _question.files =", [file.name if hasattr(file, 'name') else file for file in _question.files])
    
    if files_cnts[1] + videos_cnt > 1 or (files_cnts[1] + videos_cnt == 1 and files_cnts[0] + images_cnt > 0):
        gr.Warning("Only supports single video file input right now!")
        return _question, _chat_bot, _app_cfg

    if params_form == 'Beam Search':
        params = {
            'sampling': False,
            'num_beams': 3,
            'repetition_penalty': 1.2,
            "max_new_tokens": 2048
        }
    else:
        params = {
            'sampling': True,
            'top_p': 0.8,
            'top_k': 100,
            'temperature': 0.7,
            'repetition_penalty': 1.05,
            "max_new_tokens": 2048
        }
    
    if files_cnts[1] + videos_cnt > 0:
        params["max_inp_length"] = 4352 # 4096+256
        params["use_image_id"] = False
        params["max_slice_nums"] = 1 if count_video_frames(_context) > 16 else 2

    code, _answer, _, sts = chat(uploaded_images, _context, None, params)

    # 将结果转换为 Gallery 组件所需的格式
    gallery_output = [(img, "") for img in uploaded_images]  # 每个图片路径加上空标题
    print("Debug: Gallery output =", gallery_output)

    images_cnt += files_cnts[0]
    videos_cnt += files_cnts[1]
    _context.append({"role": "assistant", "content": [make_text(_answer)]}) 
    # 只上传文字，不上传图片 图片自动上传到左侧区域
    # image_display_area = [uploaded_images]
    image_display_area = gallery_output
    _question_text = _question.text
    _chat_bot.append((_question_text, _answer))
    # _chat_bot.append((_question, _answer))

    if code == 0:
        _app_cfg['ctx']=_context
        _app_cfg['sts']=sts
    _app_cfg['images_cnt'] = images_cnt
    _app_cfg['videos_cnt'] = videos_cnt

    upload_image_disabled = videos_cnt > 0
    upload_video_disabled = videos_cnt > 0 or images_cnt > 0

    # 清空 txt_message.files
    txt_message = create_multimodal_input(upload_image_disabled, upload_video_disabled)
    txt_message.files = []  # 显式清空文件路径
    print("txt = ",txt_message.files)

    return txt_message, _chat_bot, _app_cfg, image_display_area

# 重新生成（目前不调用）
def fewshot_add_demonstration(_image, _user_message, _assistant_message, _chat_bot, _app_cfg):
    ctx = _app_cfg["ctx"]
    message_item = []
    if _image is not None:
        image = Image.open(_image).convert("RGB")
        ctx.append({"role": "user", "content": [encode_image(image), make_text(_user_message)]})
        message_item.append({"text": "[mm_media]1[/mm_media]" + _user_message, "files": [_image]})
    else:
        if _user_message:
            ctx.append({"role": "user", "content": [make_text(_user_message)]})
            message_item.append({"text": _user_message, "files": []})
        else:
            message_item.append(None)
    if _assistant_message:
        ctx.append({"role": "assistant", "content": [make_text(_assistant_message)]})
        message_item.append({"text": _assistant_message, "files": []})
    else:
        message_item.append(None)

    _chat_bot.append(message_item)
    return None, "", "", _chat_bot, _app_cfg

# 重新生成（目前不调用）
def fewshot_respond(_image, _user_message, _chat_bot, _app_cfg, params_form):
    user_message_contents = []
    _context = _app_cfg["ctx"].copy()
    if _image:
        image = Image.open(_image).convert("RGB")
        user_message_contents += [encode_image(image)]
    if _user_message:
        user_message_contents += [make_text(_user_message)]
    if user_message_contents:
        _context.append({"role": "user", "content": user_message_contents})

    if params_form == 'Beam Search':
        params = {
            'sampling': False,
            'num_beams': 3,
            'repetition_penalty': 1.2,
            "max_new_tokens": 2048
        }
    else:
        params = {
            'sampling': True,
            'top_p': 0.8,
            'top_k': 100,
            'temperature': 0.7,
            'repetition_penalty': 1.05,
            "max_new_tokens": 2048
        }
    
    code, _answer, _, sts = chat("", _context, None, params)

    _context.append({"role": "assistant", "content": [make_text(_answer)]})

    if _image:
        _chat_bot.append([
            {"text": "[mm_media]1[/mm_media]" + _user_message, "files": [_image]},
            {"text": _answer, "files": []}        
        ])
    else:
        _chat_bot.append([
            {"text": _user_message, "files": [_image]},
            {"text": _answer, "files": []}        
        ])
    if code == 0:
        _app_cfg['ctx']=_context
        _app_cfg['sts']=sts
    return None, '', '', _chat_bot, _app_cfg

# 重新生成（目前不调用）
def regenerate_button_clicked(_question, _image, _user_message, _assistant_message, _chat_bot, _app_cfg, params_form):
    if len(_chat_bot) <= 1 or not _chat_bot[-1][1]:
        gr.Warning('No question for regeneration.')
        return '', _image, _user_message, _assistant_message, _chat_bot, _app_cfg
    if _app_cfg["chat_type"] == "Chat":
        images_cnt = _app_cfg['images_cnt']
        videos_cnt = _app_cfg['videos_cnt']
        _question = _chat_bot[-1][0]
        _chat_bot = _chat_bot[:-1]
        _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
        files_cnts = check_has_videos(_question)
        images_cnt -= files_cnts[0]
        videos_cnt -= files_cnts[1]
        _app_cfg['images_cnt'] = images_cnt
        _app_cfg['videos_cnt'] = videos_cnt
        upload_image_disabled = videos_cnt > 0
        upload_video_disabled = videos_cnt > 0 or images_cnt > 0
        _question, _chat_bot, _app_cfg = respond(_question, _chat_bot, _app_cfg, params_form)
        return _question, _image, _user_message, _assistant_message, _chat_bot, _app_cfg
    else: 
        last_message = _chat_bot[-1][0]
        last_image = None
        last_user_message = ''
        if last_message.text:
            last_user_message = last_message.text
        if last_message.files:
            last_image = last_message.files[0].file.path
        _chat_bot = _chat_bot[:-1]
        _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
        _image, _user_message, _assistant_message, _chat_bot, _app_cfg = fewshot_respond(last_image, last_user_message, _chat_bot, _app_cfg, params_form)
        return _question, _image, _user_message, _assistant_message, _chat_bot, _app_cfg


def flushed():
    return gr.update(interactive=True)


def clear(txt_message, chat_bot, app_session, image_display_area):
    txt_message.files.clear()
    txt_message.text = ''
    chat_bot = copy.deepcopy(init_conversation)
    app_session['sts'] = None
    app_session['ctx'] = []
    app_session['images_cnt'] = 0
    app_session['videos_cnt'] = 0
    image_display_area = []  # 将 Gallery 内容设置为空列表
    return create_multimodal_input(), chat_bot, app_session, None, '', '', image_display_area
    

def select_chat_type(_tab, _app_cfg):
    _app_cfg["chat_type"] = _tab
    return _app_cfg

def save_uploaded_files(files, folder):
    saved_paths = []
    for file in files:
        # 获取文件名
        file_name = os.path.basename(file.name)
        # 构造保存路径
        save_path = os.path.join(folder, file_name)
        # 将文件保存到本地
        with open(save_path, "wb") as f:
            f.write(file.read())
        saved_paths.append(save_path)
        # print(f"File saved to: {save_path}")
    return saved_paths

init_conversation = [
    [
        None,
        {
            # The first message of bot closes the typewriter.
            "text": """
            尊敬的用户，欢迎您使用多场景智能异常识别系统V4.0！我们致力于为您提供高效、智能的异常行为监控与分析解决方案。

            本模型具备强大的通用异常行为识别能力，能够自动检测和分析各种场景中的异常情况。同时，您还可以通过下方的自定义规则输入窗口，根据具体的业务需求设定个性化的检测规则，实现更加精准的风险控制和人员管理。

            为了帮助您快速上手并充分利用本模型的各项功能，请参考以下步骤进行操作：
            1. 上传数据：点击“图片输入源”或“视频输入源”，选择需要分析的文件。
            2. 自定义规则：在下方的自定义规则输入窗口中，输入您的定制化规则。例如，您可以设置特定区域的人员限制、设备运行状态等。（注意：若没有自定义规则，不用输入文字，直接点击“提交”即可）
            3. 查看结果：点击“提交”，系统将自动处理并生成分析报告。

            如有任何疑问或需要进一步的帮助，请随时查阅“使用说明”或联系我们的技术支持团队。感谢您的信任与支持！
            """,
            "flushing": False
        }
    ],
]


css = """
video { height: auto !important; }
.example label { font-size: 16px;}
"""

introduction = """

## 模型V4.0特点:
1. 通用异常行为识别：覆盖多种常见场景的异常识别。通过对图片场景分类，输出特定场景的异常情况。
2. 对于自定义规则异常行为识别：用户可以根据具体业务需求，定制个性化规则。
3. 点击 `常见示例` 按钮，可查看部分场景的模型测试结果。

"""

with gr.Blocks(css=css) as demo:
    with gr.Tab(model_name):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown(value=introduction)
                params_form = create_component(form_radio, comp='Radio')
                # 组件不可见
                params_form.visible = False

                # preview_area = gr.Gallery(label="已上传文件", columns=2, height=400, object_fit="contain")

                regenerate = create_component({'value': '重新生成'}, comp='Button')
                # 组件不可见
                regenerate.visible = False
                # 显示图片
                image_display_area = gr.Gallery(label="已上传图片", columns=2, height=300, object_fit="contain", interactive=False) # 禁用交互性

                upload_image_btn = create_component(
                    {'label': '图片输入源', 'file_types': ['image'], 'file_count': 'multiple'}, comp='UploadButton'
                )
                upload_video_btn = create_component(
                    {'label': '视频输入源', 'file_types': ['video'], 'file_count': 'single'}, comp='UploadButton'
                )

                clear_button = create_component({'value': '清除历史'}, comp='Button')

            with gr.Column(scale=3, min_width=500):
                app_session = gr.State({'sts':None,'ctx':[], 'images_cnt': 0, 'videos_cnt': 0, 'chat_type': 'Chat'})
                chat_bot = mgr.Chatbot(label=f"{model_name}", value=copy.deepcopy(init_conversation), height=600, flushing=False, bubble_full_width=False)
                
                with gr.Tab("自定义规则输入窗口") as chat_tab:
                    txt_message = create_multimodal_input()
                    print("woshi tab lide txt =", txt_message)
                    
                    chat_tab_label = gr.Textbox(value="Chat", interactive=False, visible=False)

                    def handle_uploaded_files(files):
                        file_paths = [file.name for file in files]  # 获取文件路径列表
                        print("Debug: Uploaded files =", file_paths)
                        return file_paths, file_paths  # 同时更新 preview_area 和 file_state

                    file_state = gr.State([])  # 用于存储文件路径的状态变量
                        # 绑定上传按钮事件
                    upload_image_btn.upload(
                        handle_uploaded_files,
                        inputs=[upload_image_btn],
                        outputs=[image_display_area, file_state]
                    )

                    txt_message.submit(
                        respond,
                        [txt_message, chat_bot, app_session, params_form, file_state, image_display_area], 
                        [txt_message, chat_bot, app_session, image_display_area]
                    )


                with gr.Tab("Few Shot", visible=False) as fewshot_tab:
                    fewshot_tab_label = gr.Textbox(value="Few Shot", interactive=False, visible=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(type="filepath", sources=["upload"])
                        with gr.Column(scale=3):
                            user_message = gr.Textbox(label="User")
                            assistant_message = gr.Textbox(label="Assistant")
                            with gr.Row():
                                add_demonstration_button = gr.Button("Add Example")
                                generate_button = gr.Button(value="Generate", variant="primary")
                    add_demonstration_button.click(
                        fewshot_add_demonstration,
                        [image_input, user_message, assistant_message, chat_bot, app_session],
                        [image_input, user_message, assistant_message, chat_bot, app_session]
                    )
                    generate_button.click(
                        fewshot_respond,
                        [image_input, user_message, chat_bot, app_session, params_form],
                        [image_input, user_message, assistant_message, chat_bot, app_session]
                    )

                chat_tab.select(
                    select_chat_type,
                    [chat_tab_label, app_session],
                    [app_session]
                )
                chat_tab.select( # do clear
                    clear,
                    [txt_message, chat_bot, app_session],
                    [txt_message, chat_bot, app_session, image_input, user_message, assistant_message]
                )
                fewshot_tab.select(
                    select_chat_type,
                    [fewshot_tab_label, app_session],
                    [app_session]
                )
                fewshot_tab.select( # do clear
                    clear,
                    [txt_message, chat_bot, app_session],
                    [txt_message, chat_bot, app_session, image_input, user_message, assistant_message]
                )
                chat_bot.flushed(
                    flushed,
                    outputs=[txt_message]
                )
                regenerate.click(
                    regenerate_button_clicked,
                    [txt_message, image_input, user_message, assistant_message, chat_bot, app_session, params_form],
                    [txt_message, image_input, user_message, assistant_message, chat_bot, app_session]
                )
                clear_button.click(
                    clear,
                    [txt_message, chat_bot, app_session, image_display_area],
                    [txt_message, chat_bot, app_session, image_input, user_message, assistant_message, image_display_area]
                    # [txt_message, chat_bot, app_session, preview_area],
                    # [txt_message, chat_bot, app_session, image_input, user_message, assistant_message, preview_area]
                )
        
    with gr.Tab("常见示例"):
        with gr.Column():
            # 示例 1: 工人佩戴安全帽
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    **一、工业场景异常识别示例**
                    
                    可以识别工人是否正确佩戴安全帽，当前场景
                    是否存在火灾风险以及燃油泄漏情况等工业场
                    景中的异常情况。
                    """)
                with gr.Column(scale=1):
                    gr.Image(value="/home/server/Pictures/Screenshots/无自定义规则-安全帽.png", label="工地场景示例", interactive=False, width=400)

            # 示例 2: 办公室场景
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    **二、办公场所异常识别示例**
                    
                    可以识别是否有人工作时打瞌睡，办公室文件
                    柜放置是否稳定等办公场景中的异常情况。
                    """)
                with gr.Column(scale=1):
                    gr.Image(value="/home/server/Pictures/Screenshots/办公室-坐姿不端-无自定义规则.png", label="办公室违规行为示例", interactive=False, width=400)

            # 示例 3: 街道危险行为
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    **三、街道场景异常识别示例**
                    
                    可以识别行人翻越围栏，电动车逆行，盲道被
                    占用以及街道地面凹陷等街道场景中常见的异
                    常情况。
                    """)
                with gr.Column(scale=1):
                    gr.Image(value="/home/server/Pictures/Screenshots/无自定义规则-翻越围栏.png", label="街道危险行为示例", interactive=False, width=400)

            # 示例 4: 交通安全隐患示例
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    **四、道路交通异常识别示例**
                    
                    可以识别车辆或行人闯红灯，高速路上机动车
                    占用应急车道，大货车在非规定时间上路等异
                    情况。
                    """)
                with gr.Column(scale=1):
                    gr.Image(value="/home/server/Pictures/Screenshots/有规则-电动车违规驾驶.png", label="交通隐患示例", interactive=False, width=400)

            # 示例 5: 商场安全隐患示例
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    **五、商场安全隐患示例**

                    可以识别消防门前被杂物堵塞，公共场所  
                    有人携带刀具，扶手破损等公共场所常见
                    异常。
                    """)
                with gr.Column(scale=1):
                    gr.Image(value="/home/server/Pictures/Screenshots/有规则-堵塞安全通道.png", label="商场安全隐患示例", interactive=False, width=400)




# launch
demo.launch(share=False, debug=True, show_api=False, server_port=8780, server_name="0.0.0.0")