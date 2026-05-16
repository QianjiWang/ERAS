import os
from openai import OpenAI
from datetime import datetime
import json
import sys
import re
import _main_command_ctrl
"""补充blip模块--开始"""
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
# 使用local_files_only=True可以强制离线，避免连内网时代码不运行
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large",
                                                     cache_dir="./_huggingface_model/blip",
                                                     local_files_only=True)
processor_blip = BlipProcessor.from_pretrained(pretrained_model_name_or_path="Salesforce/blip-image-captioning-large",
                                          cache_dir="./_huggingface_model/blip",
                                          local_files_only=True)
raw_image = Image.open('./3_img2text/imgs/img0.jpg').convert('RGB')
# unconditional image captioning
inputs = processor_blip(raw_image, return_tensors="pt")
out = model_blip.generate(**inputs)
scene_description = "Here is the description of the scene: \n" + processor_blip.decode(out[0], skip_special_tokens=True)
print(scene_description)
"""补充blip模块--结束"""
conversation_file_folder = "./1_LLMs/conversation_history_1"
skill_listName_path = "./AssembleSkillList1.txt"
api_key_path = "./1_LLMs/api_key.txt"
api_key = open(api_key_path).read().strip()
MainCommandCtrl = _main_command_ctrl.MainCommandCtrl()

try:
    with open(skill_listName_path, 'r', encoding='utf-8') as file:
        skill_list_content = file.read()
        print(skill_list_content)
except FileNotFoundError:
    print(f"未能找到技能库名字列表的文件: {skill_listName_path}")
    sys.exit(1)  # 使用非零状态码表示错误退出
except Exception as e:
    print(f"读取文件时出错: {e}")
    sys.exit(1)  # 使用非零状态码表示错误退出
print("Skill List:\n", skill_list_content)

client = OpenAI(
    api_key=api_key, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 初始化对话
conversation_history = [
    {'role': 'system', 'content': 'You are a helpful assistant, and an extraordinary scientist/expert in robot learning.'}
]

# %%
# 语音输入模块
import whisper
import torch
import sounddevice as sd
import soundfile as sf
import wave
import keyboard  # 用于检测按键事件
import numpy as np
# def get_user_input():g
#     return input("User: ")

"""RNNoise降噪模块--开始"""
try:
    # 尝试导入常见的RNNoise Python绑定
    try:
        import pyrnnoise as rnnoise
        RNNOISE_AVAILABLE = True
        print("RNNoise库(pyrnnoise)已加载，将启用语音降噪功能")
    except ImportError:
        RNNOISE_AVAILABLE = False
        print("警告: RNNoise库未安装，将跳过降噪步骤")
        print("如需安装，请运行以下命令之一:")
        print("  pip install pyrnnoise")
        print("  或 pip install rnnoise-python")
        print("  或使用其他音频降噪方案")
except Exception as e:
    RNNOISE_AVAILABLE = False
    print(f"警告: RNNoise库加载失败 ({e})，将跳过降噪步骤")
"""RNNoise降噪模块--结束"""


# Whisper 模型初始化
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = whisper.load_model(name="small",
                           device=device,
                           download_root="./_whisper_model",
                           ).cuda()
model.eval()
print("Whisper 模型加载完成")

# 录音参数
ENGLISH = True
DURATION = 8
SAMPLE_RATE = 16000  # Whisper 推荐采样率
CHANNELS = 1         # 声道数
RECORDING_FILENAME = "./temp_recording.wav"

def record_audio(output_file):
    # 录制麦克风音频
    print(f"开始录音，请说话...(时长为{DURATION}秒)")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # 等待录音完成
    # 保存录音为 WAV 文件
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print("录音结束，正在保存...")
        sf.write(output_file, audio, SAMPLE_RATE)  # 使用 soundfile 保存
        print(f"录音已保存为: {output_file}")
    except Exception as e:
        print(f"保存录音文件时发生错误: {e}")

def denoise_audio(input_file, output_file=None):
    """使用RNNoise对音频进行降噪处理"""
    if not RNNOISE_AVAILABLE:
        print("RNNoise不可用，跳过降噪步骤")
        return input_file
    
    if output_file is None:
        output_file = input_file.replace('.wav', '_denoised.wav')
    
    try:
        print(f"正在对音频文件 {input_file} 进行降噪处理...")
        
        # 读取原始音频
        audio_data, sample_rate = sf.read(input_file)
        
        # 确保是单声道
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 确保采样率为48kHz（RNNoise要求）
        if sample_rate != 48000:
            # 重采样到48kHz
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=48000)
            sample_rate = 48000
        
        # 转换为float32格式并归一化到[-1, 1]
        audio_data = audio_data.astype(np.float32)
        
        # 初始化RNNoise处理器
        denoiser = rnnoise.RNNoise()
        
        # RNNoise处理需要分帧处理（每帧480个样本）
        frame_size = 480
        num_frames = len(audio_data) // frame_size
        denoised_audio = np.zeros(num_frames * frame_size, dtype=np.float32)
        
        print(f"正在处理 {num_frames} 个音频帧...")
        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = start_idx + frame_size
            frame = audio_data[start_idx:end_idx]
            denoised_frame = denoiser.process_frame(frame)
            denoised_audio[start_idx:end_idx] = denoised_frame
        
        # 如果采样率不是48kHz，需要重采样回原始采样率
        if SAMPLE_RATE != 48000:
            denoised_audio = librosa.resample(denoised_audio, orig_sr=48000, target_sr=SAMPLE_RATE)
        
        # 保存降噪后的音频
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        sf.write(output_file, denoised_audio, SAMPLE_RATE)
        print(f"降噪完成，已保存至: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"降噪处理失败: {e}")
        print("将使用原始音频进行转录")
        return input_file

def transcribe_audio(file_path, is_en=True):
    """调用 Whisper 模型将音频转文本"""
    print(f"正在处理音频文件 {file_path}...")
    if is_en is True:
        result = model.transcribe(file_path, language="en")
    else:
        result = model.transcribe(file_path, language="zh")
    print(f"转录结果: {result['text']}")
    return result["text"]

def get_user_input(is_en=True):
    print("按 's' 开始录音，按 'Esc' 结束对话...")
    while True:
        if keyboard.is_pressed('s'): break
        if keyboard.is_pressed('Esc'): return "exit"
    """用户输入：录音 + RNNoise降噪 + Whisper 转文本"""
    record_audio(RECORDING_FILENAME)  # 录音
    
    # 先进行降噪处理
    DENOISED_FILENAME = RECORDING_FILENAME.replace('.wav', '_denoised.wav')
    denoised_file = denoise_audio(RECORDING_FILENAME, DENOISED_FILENAME)
    
    # 使用降噪后的音频进行转录
    text = transcribe_audio(denoised_file, is_en=is_en)  # Whisper 转文本
    return text.strip()









# %%
# chat解析模块
def chat_with_model(conversation_history):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=conversation_history,
    )
    response = completion.choices[0].message.content.strip()
    return response

prompt1 = """
Imagine your are serving an Embodied AI Assembly System. Your job is to help me analyzing the assembly task through command, then output the code command that achieves the desired goal. 
Your task is to call specific functions (api) which are predefined. The requirement is to generate code commands concisely.

You can temporarily use the following functions:
initialize_assembly(string specific_assembly_skill)
assemble(string specific_assembly_skill)
finish_assembly(string specific_assembly_skill)

"""

prompt2 = """
All of your outputs need to be identified by one of the following tags:
<question> Always ask me a clarification questions if you are unsure </question>
<reason> Explain why you did something the way you did it </reason>
<command> Output code command that achieves the desired goal </command>

For example:
Me: I want to assemble that axis with 6 corners and that object with 2 shafts.
You: <command>initialize_assembly("hexagonal axis");assemble("hexagonal axis");finish_assembly("hexagonal axis");initialize_assembly("two-axis socket 1");assemble("two-axis socket 1");finish_assembly("two-axis socket 1");</command><reason>I deduct that your command is to assemble the "hexagonal axis" and "two-axis socket 1" which can be found in the skill list, so I call the correspond api.</reason>

Are you ready?
"""

def first_conversation():
    knowledge = """
Assembly Process Knowledge Base: 
1. Planetary Gear System Assembly
Prerequisite Check: Verify if the flanged shaft is properly assembled, if not mentioned, you need to assemble the flanged shaft first.
Sequence: First, assemble the flanged shaft, then install the center gear, finally assemble the planetary gears around it.
Critical Note: Ensure proper meshing alignment to avoid gear tooth interference.
2. Ball Bearing Installation
Prerequisite Check: Confirm the housing bore is clean and free of burrs.
Sequence: Press-fit the outer ring first, then align and insert the inner ring with rolling elements.
Critical Note: Avoid misalignment during press-fitting to prevent bearing seizure.
3.Shaft-Coupling Assembly
Prerequisite Check: Ensure both shaft ends have matching keyways and are deburred.
Sequence: Slide the coupling hub onto one shaft, align, then secure with set screws or keys.
Critical Note: Check concentricity post-assembly to minimize vibration.
"""
    user_message = knowledge + scene_description + prompt1 + skill_list_content + prompt2
    conversation_history.append({'role': 'user', 'content': user_message})
    response = chat_with_model(conversation_history)
    print(f"Assistant: {response}")
    conversation_history.append({'role': 'assistant', 'content': response})

def main_loop():
    print("开始对话，输入 'exit' 以结束对话。")
    for i in range(100):  # 最多100次对话
        user_message = get_user_input(is_en=ENGLISH)
        if user_message.lower() == 'exit':
            print("对话已结束。")
            break
        conversation_history.append({'role': 'user', 'content': user_message})
        response = chat_with_model(conversation_history)
        print(f"Assistant: {response}")
        conversation_history.append({'role': 'assistant', 'content': response})
        "提取命令并执行"
        command_content = extract_command(response)
        if command_content:
            execute_command(command_content)
        else:
            print("没有命令可以执行。")
    else:
        print("达到最大对话次数限制，对话自动结束。")

def extract_command(response):
    """
    从模型的回答中提取 <command> 标签内的命令
    """
    command_pattern = re.compile(r'<command>(.*?)</command>', re.DOTALL)
    command_match = command_pattern.search(response)
    if command_match:
        command_content = command_match.group(1).strip()
        return command_content
    return None

def execute_command(command_content):
    """
    解析并执行命令
    """
    # 假设命令是按行排列的，每行是一个函数调用
    commands = command_content.split(';')
    for command in commands:
        command = command.strip()
        if command.startswith("initialize_assembly"):
            skill_name = command.split('"')[1]
            MainCommandCtrl.initialize_assembly(skill_name)
        elif command.startswith("assemble"):
            skill_name = command.split('"')[1]
            MainCommandCtrl.assemble(skill_name)
        elif command.startswith("finish_assembly"):
            skill_name = command.split('"')[1]
            MainCommandCtrl.finish_assembly(skill_name)
        else:
            print(f"Unknown command: {command}")

def save_conversation_to_file(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 获取当前时间并格式化为字符串，用于文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_folder, f"conversation_{timestamp}.json")
    # 将对话历史保存为JSON文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)
    print(f"对话历史已保存至文件: {filename}")





if __name__ == "__main__":
    first_conversation()
    main_loop()
    save_conversation_to_file(output_folder=conversation_file_folder)