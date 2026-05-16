# encoding: utf-8
import rtde_control
import rtde_receive
# import rtde_io
import urx
import numpy as np
import cv2
import math
import utils, ftsensor_read
import time


class UR5_Rtde(object):
    def __init__(self):
        UR5_IP = "192.168.1.21"
        self.rtde_c = rtde_control.RTDEControlInterface(UR5_IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(UR5_IP)
        # self.rtde_io = rtde_io.RTDEIOInterface(UR5_IP) # rtde有公有的寄存器被占用的问题


    def close(self):
        self.rtde_c.stopScript()
        print("RTDE Robot Connection End")
        
    def refresh(self):
        self.rtde_c.disconnect()
        self.rtde_c.reconnect()
        
    def disconnect(self):
        self.rtde_c.disconnect()

    
    def getJ(self):
        return self.rtde_r.getActualQ()

    def servoJ(self, joints_positions, speed=0.5, acc=0.25, time=1, lookahead_time=0.1, gain=1000):
        self.rtde_c.servoJ(joints_positions, speed, acc, time, lookahead_time, gain)
    
    def moveJ(self, joints_positions, speed=0.5, acc=0.25, threshold_time=10, asynchronous=True,
              isWait=True, isPrintLog=False):
        self.rtde_c.moveL_FK(joints_positions, speed, acc, asynchronous)
        cnt = 0
        start_time = time.time()
        while isWait:
            cnt += 1
            cur_j = self.getJ()
            diff = sum(abs(x-y) for x,y in zip(joints_positions, cur_j))
            if (diff < 0.01) or (time.time()-start_time > threshold_time):
                break
            if isPrintLog: print(f'cnt = {cnt}, now_joint = {cur_j}')

    def getPosRotvec(self):
        pos_rotvec = self.rtde_r.getActualTCPPose()
        # 获取位置和旋转向量
        return pos_rotvec    
        
    def getPosOrt(self):
        pos_rotvec = self.rtde_r.getActualTCPPose()
        rotvector = np.array(pos_rotvec[3:6]).reshape(3, 1)
        rot = cv2.Rodrigues(rotvector)[0]
        rpy = utils.rot2euler(rot).tolist()
        pos_ort = np.array(pos_rotvec[0:3] + rpy)
        # 获取位置和rpy角
        return pos_ort


    def setTCPOffset(self, tcp_offset):
        # 设置TCP
        self.rtde_c.setTcp(tcp_offset)

    def getTCPOffset(self):
        return self.rtde_c.getTCPOffset()

    def setPayload(self, mass, cog):
        # mass: Mass in kilograms
        # cog: Center of Gravity, a vector [CoGx, CoGy, CoGz] 
        #      specifying the displacement (in meters) from the toolmount.
        self.rtde_c.setTargetPayload(mass=mass, cog=cog)


    def setPosRotvec(self, pos_rotvec, speed=0.5, acc=0.25, asynchronous=True, 
                        isWait=True, isSoft=False):             
        self.rtde_c.moveL(pos_rotvec, speed, acc, asynchronous)
        cnt = 0
        while isWait:
            cnt += 1
            cur_pos_rotvec = self.getPosRotvec()
            diff1 = sum(abs(x-y) for x, y in zip(pos_rotvec[0:3], cur_pos_rotvec[0:3]))
            # 修正-3.14153和3.14152错误判断的情况
            diff2 = sum(min(abs(x-y), abs(x-y-2*math.pi), abs(x-y+2*math.pi)) 
                        for x, y in zip(pos_rotvec[3:6], cur_pos_rotvec[3:6]))
            print(f'cnt = {cnt}, now_pos = {cur_pos_rotvec}')
            if (diff1 + diff2 < 0.0005):
                break

        while isSoft:
            ft =self.getFTdata()
            for i in range(3):
                if abs(ft[i]) > 20 or abs(ft[i+3]) > 10:
                    self.rtde_c.stopL(acc=0.25, asynchronous=True)
                    break
            
        print(f'target pos = {pos_rotvec}\nnow_pos = {cur_pos_rotvec}')

    def setPosOrt(self, pos_ort, speed=0.5, acc=0.25, threshold_time=10, asynchronous=True, 
                    isWait=True, isSoft=False, soft_f_threshold=15, soft_return=0.0015, isPrintLog=False):
        rot= utils.rpy_to_rotation(pos_ort[3], pos_ort[4], pos_ort[5])
        vector= cv2.Rodrigues(rot)[0].tolist()
        # 二维[[1], [2], ..., [n]]转一维[1, 2, ..., n]
        rotvector = [i for item in vector for i in item]
        tcp_pos_rotvec = np.append(pos_ort[0:3], np.array(rotvector))
        self.rtde_c.moveL(tcp_pos_rotvec, speed, acc, asynchronous)

        cnt = 0
        start_time = time.time()
        while isWait:
            if isSoft:
                soft_flag = False
                ft =self.getFTdata()
                cur_pos_ort = self.getPosOrt()
                diff_pos = pos_ort[0:3] - cur_pos_ort[0:3]
                soft_pos_ort = np.copy(pos_ort)
                for i in range(3):
                    # if abs(ft[i]) > 25 or abs(ft[i+3]) > 15:
                    # if abs(ft[i]) > 25:
                    # if abs(ft[i]) > 15:
                    if abs(ft[i]) > soft_f_threshold:
                        "可在这儿设置回退幅度的大小！"
                        # soft_pos_ort[i] = cur_pos_ort[i] - 0.0015 * np.sign(diff_pos[i])
                        # soft_pos_ort[i] = cur_pos_ort[i] - 0.003 * np.sign(diff_pos[i])
                        soft_pos_ort[i] = cur_pos_ort[i] - soft_return * np.sign(diff_pos[i])
                        soft_flag = True
                self.setPosOrt(pos_ort=soft_pos_ort, speed=speed, acc=acc, 
                                threshold_time=threshold_time, asynchronous=asynchronous, 
                                isWait=isWait, isSoft=False, isPrintLog=False)
                if soft_flag is True:
                    print("此次ik力过大,F={}\tT={}".format(np.linalg.norm(ft[0:3]), np.linalg.norm(ft[3:6])))
                break
            cnt += 1
            cur_pos_rotvec = self.getPosRotvec()
            diff1 = sum(abs(x-y) for x, y in zip(tcp_pos_rotvec[0:3], cur_pos_rotvec[0:3]))
            # 修正-3.14153和3.14152错误判断的情况
            diff2 = sum(min(abs(x-y), abs(x-y-2*math.pi), abs(x-y+2*math.pi)) 
                        for x, y in zip(tcp_pos_rotvec[3:6], cur_pos_rotvec[3:6]))
            if isPrintLog: print(f'cnt = {cnt}, now_pos = {cur_pos_rotvec}')
            if (diff1 + diff2 < 0.0005) or (time.time()-start_time > threshold_time):
                # self.rtde_c.stopL(acc=0.25, asynchronous=True)
                # self.rtde_c.stopL()
                break
        
        if isPrintLog: print(f'target pos = {pos_ort}\nnow_pos = {self.getPosOrt()}')


    def getFTdata(self):
        return ftsensor_read.udp_get()
    


    

