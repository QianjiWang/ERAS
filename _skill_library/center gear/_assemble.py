# %%
#@markdown ### **Imports**
# file import
import pandas as pd
import assemble_env
import utils
# module import
from random import uniform,choice
import numpy as np
import time
import os
import torch
from torch import nn
from PIL import Image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import keyboard
from IPython.display import display, clear_output, Javascript
import gc
# 获取当前脚本所在的完整路径
current_file_path = __file__
# 获取当前脚本所在的文件夹目录
current_folder = os.path.dirname(current_file_path)
ckpt_path = os.path.join(current_folder, "_trained_models", "DP_model.ckpt")
MinMax_dir_path = os.path.join(current_folder, "_trained_models", "MinMax")

img_height = [256,256]
img_width = [256,256]
img_channel = 3

agent_obs_dim = 12
action_dim = 7

pred_horizon = 8
obs_horizon = 1
action_horizon = 4

# %%
def try_to_csv(file_path, df, info="", index=False, header=False, mode='w', isPrintInfo=False):
    # 检查文件夹是否存在，如果不存在则创建
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    while True:
        try:
            df.to_csv(file_path, index=index, header=header, mode=mode)
            break
        except Exception as e:
            if isPrintInfo: print("本次"+info+"数据写入csv失败,尝试重新写入...")
    
def try_read_csv(file_path, info="", header=None, isPrintInfo=False):  
    while True:  
        try:  
            # 检查文件是否为空  
            if os.path.getsize(file_path) == 0:  
                if isPrintInfo: print(f"{info}文件为空。")
                return pd.DataFrame()
            # 读取文件的前几行以检查是否有有效数据  
            with open(file_path, 'r') as f:  
                first_line = f.readline().strip()  
                if not first_line:  
                    if isPrintInfo: print(f"{info}文件没有有效数据。")
                    return pd.DataFrame()      
            csv = pd.read_csv(file_path, header=header)
            # print(csv)  
            return csv  
        except Exception as e:  
            if isPrintInfo:  
                print(f"本次{info}读取csv失败, 错误信息: {e}，尝试重新读取...")
                

def unnormalize_data_byValue(ndata, max, min, eps=1e-8):
    ndata = (ndata + 1) / 2 # 反归一化回原始范围
    range_ = max - min
    data = ndata * range_ + min
    return data

action_min = try_read_csv(MinMax_dir_path+"\\action.csv", info="action_MinMax", header=None).iloc[0:].to_numpy().reshape(2,-1)[0]
action_max = try_read_csv(MinMax_dir_path+"\\action.csv", info="action_MinMax", header=None).iloc[0:].to_numpy().reshape(2,-1)[1]
# print("action_min:", action_min)
# print("action_max:", action_max)

# %%
# device transfer
device = torch.device('cuda')


# %%
import DiffusionPolicy_Networks as nets
import DiffusionPolicy_Networks_VE1 as nets_VE1
vision_encoder_1 = nets_VE1.get_lightweight_vit('mobilevit')
vision_encoder_2 = nets_VE1.get_lightweight_vit('mobilevit')
vision_encoder_1 = nets.replace_bn_with_gn(vision_encoder_1)
vision_encoder_2 = nets.replace_bn_with_gn(vision_encoder_2)
print("视觉编码器1的形状:\n", vision_encoder_1)
print("视觉编码器2的形状:\n", vision_encoder_2)

# ResNet18 has output dim of 512
vision_feature_dim = 512
obs_dim = vision_feature_dim + 0

# create network object
noise_pred_net = nets.ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*2*obs_horizon
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'vision_encoder_1': vision_encoder_1,
    'vision_encoder_2': vision_encoder_2,
    'noise_pred_net': noise_pred_net
})

print("number of all parameters: {:e}".format(
    sum(p.numel() for p in nets['vision_encoder_1'].parameters())+
    sum(p.numel() for p in nets['vision_encoder_2'].parameters())+
    sum(p.numel() for p in nets['noise_pred_net'].parameters()))
)

# demo
with torch.no_grad():
    # example inputs
    image_1 = torch.zeros((1, obs_horizon,3,img_height[0],img_width[0]))
    image_2 = torch.zeros((1, obs_horizon,3,img_height[1],img_width[1]))
    # agent_obs = torch.zeros((1, obs_horizon, 6))
    # vision encoder
    image_features_1 = nets['vision_encoder_1'](
        image_1.flatten(end_dim=1))
    # (2,512)
    image_features_1 = image_features_1.reshape(*image_1.shape[:2],-1)
    # (1,2,512)
    image_features_2 = nets['vision_encoder_2'](image_2.flatten(end_dim=1))
    image_features_2 = image_features_2.reshape(*image_2.shape[:2],-1)
    # obs = torch.cat([image_features,agent_obs],dim=-1)
    obs = torch.cat([image_features_1, image_features_2],dim=-1)
    print("obs.shape:\t", obs.shape)
    print("obs.flatten(start_dim=1).shape:\t",obs.flatten(start_dim=1).shape)
    noised_action = torch.randn((1, pred_horizon, action_dim))
    diffusion_iter = torch.zeros((1,))

    # the noise prediction network
    # takes noisy action, diffusion iteration and observation as input
    # predicts the noise added to action
    noise = nets['noise_pred_net'](
        sample=noised_action,
        timestep=diffusion_iter,
        global_cond=obs.flatten(start_dim=1))

    # illustration of removing noise
    # the actual noise removal is performed by NoiseScheduler
    # and is dependent on the diffusion noise schedule
    denoised_action = noised_action - noise

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
# device = torch.device('cuda')
_ = nets.to(device)



#@markdown ### **load pretrained weights**
load_pretrained = True
if load_pretrained:
    if os.path.isfile(ckpt_path):
        # 加载检查点文件
        state_dict = torch.load(ckpt_path, map_location='cuda')
        # 从字典中提取模型状态
        model_state_dict = state_dict['model_state_dict']
        # 加载模型状态
        ema_nets = nets
        ema_nets.load_state_dict(model_state_dict)
        print('Pretrained weights loaded.')
    else:
        print("No pretrained weights found. Training from scratch.")
else:
    print("Skipped pretrained weight loading.")
    





# %%
#@markdown ### **Env**
#@markdown
env = assemble_env.AssembleEnv(img_obs_height=img_height, img_obs_width=img_width, img_obs_channel=img_channel,
                                    agent_obs_dim=agent_obs_dim, action_dim=action_dim,
                                    obs_horizon=obs_horizon, action_horizon=action_horizon, pred_horizon=pred_horizon)
RTDE_SOFT_F_THRESHOLD_env = env.rob.RTDE_SOFT_F_THRESHOLD
env.rob.RTDE_SOFT_F_THRESHOLD = 100

# %%
#@markdown ### **Initialize**
#@markdown
"设置被抓取的位姿"
pos_c = [244.5e-3, -692.03e-3, 42.28e-3, 0, -3.1415926, -1.5707963]
# env.rob.moveIK(pos_c[0:3], pos_c[3:6], wait=True)
pos_c_up = pos_c.copy()
pos_c_up[2] += 100e-3
"设置被装配的位姿"
pos_a = [-6.15e-3, -647.5e-3, 101.5e-3, 0, -3.1415, 0]
# env.rob.moveIK(pos_a[0:3], pos_a[3:6], wait=True)
pos_a_up = pos_a.copy()
pos_a_up[2] += 100e-3

env.rob.openRG2()
"catch"
# 被抓取位置上方
env.rob.moveIK(pos_c_up[0:3], pos_c_up[3:6], wait=True)
# 被抓取位置
env.rob.moveIK(pos_c[0:3], pos_c[3:6], wait=True)
# 抓取
env.rob.closeRG2()
#往下抵住
env.rob.moveIK_changePosOrt(d_pos=[0e-3,0e-3,-4e-3], d_ort=[0,0,0], wait=True)
# time.sleep(2)
# 被抓取位置上方
env.rob.moveIK(pos_c_up[0:3], pos_c_up[3:6], wait=True)
env.rob.moveIK_changePosOrt(d_pos=[0e-3,0e-3,50e-3], d_ort=[0,0,0], wait=True)


"go_to"
# 被装配位置上方
env.rob.moveIK(pos_a_up[0:3], pos_a_up[3:6], wait=True)

# env.rob.close()

env.rob.RTDE_SOFT_F_THRESHOLD = RTDE_SOFT_F_THRESHOLD_env



# %%
#@markdown ### **Inference**
#@markdown
while True:
    print("重置.......")
    
    time.sleep(0.5)
    isPrintStepInfo = False
    # isUseRandomEnv = True
    # env.reset(isAddError=False, is_drag_mode=False)
    # if isUseRandomEnv:
    #     env.reset(isAddError=True, is_drag_mode=False, error_scale_pos=4.5, error_scale_ort=2)
    # else:
    #     env.reset(isAddError=False, is_drag_mode=False)
    env.reset(isAddError=True, is_drag_mode=False, error_scale_pos=3, error_scale_ort=1.5)
        
    # print("按 s 开始本回合")
    # while True:
    #     env._get_obs(isPrintInfo=False)
    #     if keyboard.is_pressed("s"): break
    
    pred_steps = 1000
    for ps in range(pred_steps):
        print("按 x 进行停止运动")
        if keyboard.is_pressed("x"): break
        """--------------------------------------------------"""
        """核心: get action"""
        B = 1
        # stack the last obs_horizon number of observations
        img_obs_1_list = np.stack([x for x in env.img_obs_1_list])
        img_obs_2_list = np.stack([x for x in env.img_obs_2_list])

        # images are already normalized to [0,1]
        # print(img_obs_list.shape)
        nimage_obses_1 = img_obs_1_list.transpose((0, 3, 1, 2))  # Change the channel dimension to be the first one (2,img_height,img_width,3)->(2,3,img_height,img_width)
        nimage_obses_2 = img_obs_2_list.transpose((0, 3, 1, 2)) 
        
        # device transfer
        nimage_obses_1 = torch.from_numpy(nimage_obses_1).to(device, dtype=torch.float32)
        nimage_obses_2 = torch.from_numpy(nimage_obses_2).to(device, dtype=torch.float32)
        # (2,3,img_height,img_width)

        # infer action
        with torch.no_grad():
            # get image features
            image_features_1 = ema_nets['vision_encoder_1'](nimage_obses_1.flatten(end_dim=0))
            image_features_2 = ema_nets['vision_encoder_2'](nimage_obses_2.flatten(end_dim=0))
            # (2,512)

            # concat with low-dim observations
            obs_features = torch.cat([image_features_1, image_features_2], dim=-1)
            # print(obs_features)
            # print("两次视觉观测编码之差: ", torch.norm(obs_features[0] - obs_features[1], p=float('inf')))
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        
        action_pred = unnormalize_data_byValue(naction, max=action_max, min=action_min)

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end,:]
        clear_output(wait=True) # 清空当前输出
        print("DiffusionPolicy预测的动作为:")
        utils.print_array_with_precision(action, 3)
        # (action_horizon, action_dim)
        """--------------------------------------------------"""
        
        for i in range(len(action)):
            if keyboard.is_pressed("x"): break
            # "锁死转动"
            # action[i][3:6] = np.zeros(3)
            env.step(action=action[i], 
                    isPrintInfo=isPrintStepInfo)
            if env.done or env.assembleDepth>0.02:
                break
            
        if env.done or env.assembleDepth>0.02:
            break
        
    
    
    if env.success_flag is True:
        print("本回合成功")
        
    
            
            
    print("---------------------------------------------------------------------")
    print("---------------------------------------------------------------------")
    
    
    torch.cuda.empty_cache()
    gc.collect()
    # env.rob.close()
    break




# %%
#@markdown ### **Finish**
#@markdown
env.rob.openRG2()
# time.sleep(2)
# 位置上方
env.rob.moveIK_changePosOrt(d_pos=[0e-3,0e-3,100e-3], d_ort=[0,0,0], wait=True)
# env.rob.close()



# %%
#@markdown ### **Close**
#@markdown
env.rob.close()