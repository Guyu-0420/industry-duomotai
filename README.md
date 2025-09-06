# Industry-多模态
## 安防大模型 V3.0  
web_demo_V3.0.py:通过api调用实现工业安全多模态分析  
Gradio 可视化 Demo（端口 8783），对外提供一个网页界面（0.0.0.0:8783）  
调用硅基流动 SiliconFlow 的 OpenAI 兼容接口（/v1/chat/completions）让模型（Pro/Qwen/Qwen2.5-VL-7B-Instruct）进行“安全隐患”分析，返回结构化中文结论  
代码内置中文系统提示（predefine_msgs），把模型角色设为“生产安全专家”，要求输出 “是否存在隐患/不安全因素/建议措施” 三段式清单。  

#### 多模态输入：  
图片：支持多张。会用 PIL 统一到 360×480、RGB，再转 base64 data URI 传给 API。<br>
视频：只支持单个，用 decord 读取第一帧转为图片再传。<br>
单视频与多图不能混用（代码里显式限制）。<br>
#### 环境配置：
没有对本地其他文件的要求，只有对python环境依赖库的要求<br>
需要安装的库有：
注意：wheel轮子要单独下载<br>
···
conda activate <你的环境>
export PYTHONNOUSERSITE=1
python -m pip install -U \
  "gradio==4.44.1" "gradio_client==1.3.0" "pydantic==2.10.6" "fastapi==0.115.11" "uvicorn==0.30.6" "anyio==4.1.0" \
  "Pillow>=10,<11" "decord==0.6.0" "requests==2.32.3" "numpy>=1.26,<2.0" "distro==1.7" "transformers>=4.41,<5" \
  torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
#安装 modelscope_studio 本地 whl
python -m pip install -U /home/server/project/MiniCPM-o/modelscope_studio-0.4.0.9-py3-none-any.whl
#配 API Key
export SILICONFLOW_API_KEY="你的key"
#运行
python /home/server/project/MiniCPM-o/web_demos/web_demo_V3.0.py --device cuda
#浏览器访问：http://<服务器IP>:8780
···
输出：
···
(web_demo) server@server-Precision-7920-Tower:~/Pictures$ python /home/server/project/MiniCPM-o/web_demos/web_demo_V3.0.py --device cuda
woshi tab lide txt = <modelscope_studio.components.MultimodalInput.ModelScopeMultimodalInput object at 0x73b000757430>
/home/server/anaconda3/envs/web_demo/lib/python3.10/site-packages/gradio/utils.py:1002: UserWarning: Expected 4 arguments for function <function clear at 0x73b000d54f70>, received 3.
  warnings.warn(
/home/server/anaconda3/envs/web_demo/lib/python3.10/site-packages/gradio/utils.py:1006: UserWarning: Expected at least 4 arguments for function <function clear at 0x73b000d54f70>, received 3.
  warnings.warn(
Running on local URL:  http://0.0.0.0:8783

To create a public link, set `share=True` in `launch()`. 
--------
  warnings.warn(
Debug: Uploaded files = ['/tmp/gradio/d9880485c1b4b1a0f685cd5929c26d58a8006a67eb6ff27ca5f297f0f858bcba/2.jpg']
respond里面的question文本 = 1
respond里面的文件路径 = []
woshiencode-message =  files=[] text='1'
encode image for api= ['/tmp/gradio/d9880485c1b4b1a0f685cd5929c26d58a8006a67eb6ff27ca5f297f0f858bcba/2.jpg']
我是= /tmp/gradio/d9880485c1b4b1a0f685cd5929c26d58a8006a67eb6ff27ca5f297f0f858bcba/2.jpg
Debug: API Request Payload = {'model': 'Qwen/Qwen2.5-VL-72B-Instruct', 'messages': [{'role': 'user', 'content': []}], 'stream': False, 'max_tokens': 512, 'temperature': 0.7, 'top_p': 0.7, 'top_k': 50, 'frequency_penalty': 0.5}
Debug: Gallery output = [('/tmp/gradio/d9880485c1b4b1a0f685cd5929c26d58a8006a67eb6ff27ca5f297f0f858bcba/2.jpg', '')]
txt =  []
···
虽然还是有很明显的问题，但至少可以使用了。注意pydantic的版本必须是2.10.6，改了就会报错

#### 本地文件准备：
1/常见事例里：<br>
/home/server/Pictures/text_web_demo_327/结果/1.png<br>
...<br>
/home/server/Pictures/text_web_demo_327/结果/9.png<br>

## 多场景通用大模型 V4.0 
web_demo_V4.0.py:通过api调用实现通用场景异常多模态分析<br>
启动一个 Gradio Web 页面（0.0.0.0:8780），可以上传多张图片或1个视频；<br>
可输入（或不输入）自定义规则；<br>
后端把图片/视频首帧转成 base64，连同一段预置提示词一起，调用 SiliconFlow 的 OpenAI 兼容接口（/v1/chat/completions，模型 Qwen2.5-VL），拿到结果并显示在聊天窗里。<br>
#### 环境安装教程：
同上文V3.0
#### 本地文件：
/home/server/Pictures/Screenshots/无自定义规则-安全帽.png<br>
...<br>
/home/server/Pictures/Screenshots/有规则-堵塞安全通道.png<br> 
