import os
import subprocess

class MainCommandCtrl:
    def __init__(self):
        # 技能库的根路径
        self.skill_library_path = os.path.join(os.getcwd(), "_skill_library")

    def _execute_script(self, skill_name, script_name):
        """
        在当前工作目录下直接执行指定技能文件夹中的脚本
        :param skill_name: 技能名称，对应文件夹名称
        :param script_name: 脚本名称，比如 '_assemble.py'
        """
        # 构造目标脚本路径
        script_path = os.path.join(self.skill_library_path, skill_name, script_name)
        if not os.path.exists(script_path):
            print(f"Error: {script_name} not found in skill '{skill_name}' directory.")
            return

        try:
            # 使用 subprocess 运行脚本
            result = subprocess.run(
                ["python", script_path],  # 命令：使用当前 Python 解释器运行目标脚本
                cwd=os.getcwd(),  # 在当前工作目录运行
                capture_output=True,  # 捕获标准输出和错误
                text=True  # 将输出解码为字符串
            )
            # 打印脚本执行的输出结果
            if result.returncode == 0:
                print(f"Output of {script_name}:\n{result.stdout}")
            else:
                print(f"Error while executing {script_name}:\n{result.stderr}")
        except Exception as e:
            print(f"Error while executing {script_name}: {e}")

    def assemble(self, specific_assembly_skill):
        print(f"Assembling: {specific_assembly_skill}")
        self._execute_script(specific_assembly_skill, "_assemble.py")

    def initialize_assembly(self, specific_assembly_skill):
        print(f"Initializing assembly: {specific_assembly_skill}")
        self._execute_script(specific_assembly_skill, "_initialize_assembly.py")

    def finish_assembly(self, specific_assembly_skill):
        print(f"Finishing assembly: {specific_assembly_skill}")
        self._execute_script(specific_assembly_skill, "_finish_assembly.py")











# import os
# import importlib.util

# class MainCommandCtrl:
#     def __init__(self):
#         # 技能库的根路径
#         self.skill_library_path = os.path.join(os.getcwd(), "_skill_library") #"get current working directory"（获取当前工作目录）

#     def _load_and_execute(self, skill_name, script_name):
#         """
#         动态加载并执行对应技能文件夹中的指定脚本的 main 函数
#         :param skill_name: 技能名称，对应文件夹名称
#         :param script_name: 脚本名称，比如 'assemble.py'
#         """
#         # 构造目标脚本路径
#         script_path = os.path.join(self.skill_library_path, skill_name, script_name)
#         if not os.path.exists(script_path):
#             print(f"Error: {script_name} not found in skill '{skill_name}' directory.")
#             return
        
#         # 动态加载模块
#         spec = importlib.util.spec_from_file_location(script_name[:-3], script_path)  # 去掉 .py 后缀作为模块名
#         module = importlib.util.module_from_spec(spec)
#         try:
#             spec.loader.exec_module(module)  # 加载模块
#             if hasattr(module, "main"):
#                 module.main()  # 调用模块的 main 函数
#             else:
#                 print(f"Error: No 'main' function found in {script_name}.")
#         except Exception as e:
#             print(f"Error while executing {script_name}: {e}")

#     def assemble(self, specific_assembly_skill):
#         print(f"Assembling: {specific_assembly_skill}")
#         self._load_and_execute(specific_assembly_skill, "_assemble.py")

#     def initialize_assembly(self, specific_assembly_skill):
#         print(f"Initializing assembly: {specific_assembly_skill}")
#         self._load_and_execute(specific_assembly_skill, "_initialize_assembly.py")

#     def finish_assembly(self, specific_assembly_skill):
#         print(f"Finishing assembly: {specific_assembly_skill}")
#         self._load_and_execute(specific_assembly_skill, "_finish_assembly.py")
