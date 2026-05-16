# ablation_study.py
import os
from openai import OpenAI
import json
import re
from datetime import datetime

# 常量定义
TEST_CASES = 20
USER_MESSAGES = [
    "I want to assemble this planet gear system.",
    "Help me assemble the planetary gear system.",
    "Assemble the planetary gear system here.",
    "This planet gear system is need to be assembled."
]
TARGET_SEQUENCE = [
    'initialize_assembly("flanged shaft")',
    'assemble("flanged shaft")',
    'finish_assembly("flanged shaft")',
    'initialize_assembly("center gear")',
    'assemble("center gear")',
    'finish_assembly("center gear")',
    'initialize_assembly("planetary gear")',
    'assemble("planetary gear")',
    'finish_assembly("planetary gear")'
]

# 消融实验配置
ABLATION_GROUPS = [
    {"name": "full", "exclude": []},
    {"name": "no_knowledge", "exclude": ["knowledge"]},
    {"name": "no_scene", "exclude": ["scene"]},
    {"name": "no_prompt1", "exclude": ["prompt1"]},
    {"name": "no_skill", "exclude": ["skill"]},
    {"name": "no_prompt2", "exclude": ["prompt2"]},
]

# 初始化基础组件
api_key = open("./1_LLMs/api_key.txt").read().strip()
client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def build_prompt(group_config):
    # 各模块内容
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
    """ if "knowledge" not in group_config["exclude"] else ""
    
    scene = "Here is the description of the scene: \n" + "there is a metal object that is sitting on a table"
    
    prompt1 = """
    Imagine your are serving an Embodied AI Assembly System. Your job is to help me analyzing the assembly task through command, then output the code command that achieves the desired goal. 
    Your task is to call specific functions (api) which are predefined. The requirement is to generate code commands concisely.

    You can temporarily use the following functions:
    initialize_assembly(string specific_assembly_skill)
    assemble(string specific_assembly_skill)
    finish_assembly(string specific_assembly_skill)
    """ if "prompt1" not in group_config["exclude"] else ""
    
    skill_list = open("./AssembleSkillList1.txt").read() if "skill" not in group_config["exclude"] else ""
    
    prompt2 = """
    All of your outputs need to be identified by one of the following tags:
    <question> Always ask me a clarification questions if you are unsure </question>
    <reason> Explain why you did something the way you did it </reason>
    <command> Output code command that achieves the desired goal </command>

    For example:
    Me: I want to assemble that axis with 6 corners and that object with 2 shafts.
    You: <command>initialize_assembly("hexagonal axis");assemble("hexagonal axis");finish_assembly("hexagonal axis");initialize_assembly("two-axis socket 1");assemble("two-axis socket 1");finish_assembly("two-axis socket 1");</command><reason>I deduct that your command is to assemble the "hexagonal axis" and "two-axis socket 1" which can be found in the skill list, so I call the correspond api.</reason>

    Are you ready?
    """ if "prompt2" not in group_config["exclude"] else ""

    return f"{knowledge}{scene}{prompt1}{skill_list}{prompt2}"

def check_success(response):
    commands = re.findall(r'<command>(.*?)</command>', response, re.DOTALL)
    if not commands:
        return 0.0  # 返回浮点数分数
    
    cmd_sequence = [c.strip() for c in commands[0].split(';') if c.strip()]
    target_idx = 0
    score = 0.0
    
    # 检查完整序列
    full_sequence_matched = False
    for cmd in cmd_sequence:
        if target_idx < len(TARGET_SEQUENCE) and cmd == TARGET_SEQUENCE[target_idx]:
            target_idx += 1
            if target_idx == len(TARGET_SEQUENCE):
                full_sequence_matched = True
                break
    
    # 如果完整匹配，得1分
    if full_sequence_matched:
        score = 1.0
    else:
        # 检查是否包含关键组装步骤
        key_assemblies = [
            'assemble("flanged shaft")',
            'assemble("center gear")',
            'assemble("planetary gear")'
        ]
        found_key_steps = 0
        for cmd in cmd_sequence:
            if cmd in key_assemblies:
                found_key_steps += 1
        
        # 如果找到所有关键步骤，得0.5分
        if found_key_steps == len(key_assemblies):
            score = 0.5
    
    return score

def run_experiment(group_config):
    results = {
        "score": 0.0,  # 改为浮点数
        "success_rate": 0.0,
        "partial_success_rate": 0.0,  # 新增部分成功率的统计
        "history": []
    }
    
    for i in range(TEST_CASES):
        # 初始化对话
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant...'},
            {'role': 'user', 'content': build_prompt(group_config)},
        ]
        
        # 获取用户输入
        user_msg = USER_MESSAGES[i % 4]
        
        # 进行对话
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=conversation + [{'role': 'user', 'content': user_msg}]
        ).choices[0].message.content
        
        # 记录结果
        score = check_success(response)
        print(f"本次结果:\t {score}分")
        results["score"] += score
        results["history"].append({
            "test_case": i+1,
            "user_input": user_msg,
            "response": response,
            "score": score  # 记录详细分数
        })
    
    results["success_rate"] = sum(1 for h in results["history"] if h["score"] == 1.0) / TEST_CASES
    results["partial_success_rate"] = sum(1 for h in results["history"] if h["score"] >= 0.5) / TEST_CASES
    print(f"本组完整成功率:\t {results['success_rate']*100:.1f}%")
    print(f"本组部分成功率(≥0.5):\t {results['partial_success_rate']*100:.1f}%")
    return results


result_path = "./1_LLMs/_conversation_project1/ablation_study1/ablation_results"
def save_results(group_name, results):
    filename = f"./1_LLMs/_conversation_project1/ablation_study1/ablation_results/{group_name}_{datetime.now().strftime('%Y%m%d%H%M')}.txt"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    for group in ABLATION_GROUPS:
        print(f"Running ablation group: {group['name']}")
        results = run_experiment(group)
        save_results(group["name"], results)
        print(f"Completed. Success rate: {results['success_rate']*100:.1f}%")

if __name__ == "__main__":
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    main()