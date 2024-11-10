环境配置：
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

训练 main.py
测试 predict.py

