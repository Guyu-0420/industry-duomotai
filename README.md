# industry-duomotai
## 安防大模型  
web_demo_V3.0.py:通过api调用实现工业安全多模态分析  
Gradio 可视化 Demo（端口 8783），对外提供一个网页界面（0.0.0.0:8783）  
调用硅基流动 SiliconFlow 的 OpenAI 兼容接口（/v1/chat/completions）让模型（Pro/Qwen/Qwen2.5-VL-7B-Instruct）进行“安全隐患”分析，返回结构化中文结论  
代码内置中文系统提示（predefine_msgs），把模型角色设为“生产安全专家”，要求输出 “是否存在隐患/不安全因素/建议措施” 三段式清单。  

多模态输入：  
图片：支持多张。会用 PIL 统一到 360×480、RGB，再转 base64 data URI 传给 API。<br>
视频：只支持单个，用 decord 读取第一帧转为图片再传。<br>
单视频与多图不能混用（代码里显式限制）。<br>

没有对本地其他文件的要求，只有对python环境依赖库的要求<br>
需要安装的库有：

需要的本地文件：<br>
1/常见事例里：<br>
/home/server/Pictures/text_web_demo_327/结果/1.png<br>
...<br>
/home/server/Pictures/text_web_demo_327/结果/9.png<br>
