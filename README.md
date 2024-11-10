# MICCAI-STS-2024
SemiTeethSegChallenge

环境搭建
1.win+R 打开终端界面（这里也可以直接打开Anaconda Prompt）
 在终端命令行中输入：conda env list
 既可查看本地已经存在几个虚拟环境。base是基础环境变量，安装了anaconda之后都会有一个base虚拟环境。我们首先进入base环境：
 在终端命令行中输入：conda activate base
 进入环境后我们在anaconda中为我们的detectron2项目创建虚拟环境：
 在终端命令行中输入：conda create -n detectron_env(自定义你的环境变量的名称，我这里取名为detectron_env) python=3.8
2.为了加速我们的训练，我们往往在训练过程中调用GPU，因此需要下载对应cuda版本的pytorch（这里的detectron2是基于pytorch来实现的）
在终端命令行中输入：conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
3.在终端命令行中输入：pip install cython 
在终端命令行中输入：pip install opencv-python
（在对应的环境变量下pip install，进入的方式是：activate 你的环境变量名称）
4.打开终端后cd到项目所在的主目录下，我的项目文件名称为DAE-Net：
在终端命令行中输入：cd DAE-Net
进入到包含项目代码的主目录中后
在终端命令行中输入：pip install -e .
等待所有依赖安装好

主干网络在\DAE-Net\detectron2\modeling\backbone\resnet.py中

运行训练程序：python main.py
测试程序：python predict.py

