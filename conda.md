# anaconda 命令
## 创建虚拟环境
打开 anaconda powershell prompt

查看已有环境
``
conda env list
``

创建环境
``
conda create -n env_name python==3.x
``

进入环境
``
conda activate env_name
``

退出环境
``
conda deactivate
``

删除环境(**必须退回base**)
``
conda remove --name env_name --all
``
## 在虚拟环境安装包
```
conda install package_name
```
使用conda forge下载
```
conda install -c conda-forge package_name
```
切换到清华源（不要主动做）
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
#删除源
conda config --remove channels channel_name
# 设置搜索时显示通道地址
conda config --set show_channel_urls yes
```
退回原来的源
```
conda config --remove-key channels
```
所有通道
```
conda config --show channels
```
使用conda install 出现The following packages will be DOWNGRADED:

conda 的求解器在提出这个降级计划时，已经考虑了你环境中所有其他由 conda 安装的包对 numpy 的依赖。如果降级 numpy 会破坏你环境中另一个重要的包（比如 pandas 或 scipy），conda 通常会：
- 尝试同时升级或降级那个包。
- 或者，如果找不到解决方案，它会直接报错，告诉你存在无法解决的依赖冲突。
既然 conda 只是提出了一个降级计划而不是报错，这说明它认为降级后，你的整个环境仍然是自洽和兼容的。

结论：这是一个良性的、必要的依赖调整。请放心输入 y 并回车，让 conda 完成它的工作。

conda不行再用pip
```
pip install package_name
pip install package_name==版本号
pip install -r requirements.txt
pip insatll ... --timeout 6000
```

下载pytorch
```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch#conda命令
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia #有CUDA 12.4以上

pip3 install torch torchvision torchaudio #没有独显
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 # 有CUDA12.8
```
查看当前环境库列表
```
conda list#包括pip和conda命令
conda list -n env_name
```
找包 输出信息能包含所有环境中的这个包
```
where package_name
```
查看包的详细信息
```
conda search package_name
conda list package_name
pip show package_name
```
删除包
```
# 删除单个包
conda remove package_name
# 删除多个包
conda remove package1 package2
# 强制删除（忽略依赖冲突）
conda remove package_name --force
```
```
pip uninstall package_name
```
查看包依赖
```
pip show package_name
```
```
pip install pipdeptree
# 查看完整依赖树
pipdeptree
```
```
conda info package_name
```
```
conda install -c conda-forge conda-tree
# 查看环境依赖树
conda tree -n env_name
```

虚拟环境内进入python
```python```

退出
```exit()```
## 在虚拟环境里下载 Jupyter Notebook核
notebook只用下载一次就好，base里面已经自动下载好了，每次要下载的只是核

在当前虚拟环境中
```
conda install ipykernel
python -m ipykernel install --user --name env_name --display-name ker_name(jb里面展示的名称)
```

查看可用内核
```
jupyter kernelspec list
```
删除内核
```
jupyter kernelspec remove ker_name
```
进入jupyter```jupyter notebook```
## 在notebook界面中执行命令行指令
```
在代码单元格中输入以 ! 开头的命令：
!pip install pandas
```
多行命令
```
%%bash
# 多行命令示例
pip install 包1
pip install 包2
echo "安装完成"
```
查看当前用的是哪个python 应该与你选择的python内核一致
```python
import sys
print(sys.executable)
```
# 命令行 git bash
- ``id``
uid=197609(徐日晞) gid=197609 groups=197609
### 目录
- ``pwd``当前目录
- ``pwd``当前目录的绝对路径
- ``echo $HOME``查看主目录，是这个：/c/Users/XRXRX
- ``cd /directory``改变目录
- ``cd ~``改变到主目录(这个可以在快捷方式右键更改)
- ``cd -``改变为上一个目录
- ``cd /``改变为根目录，这是git按装的位置
- .当前目录
- ..上一级目录
### 文件的查看
- ``ls``当前目录文件
- ``ls -l``详细版当前目录文件
- ``ls <address>``某个目录下的文件 
```
drwxr-xr-x 1 徐日晞 197609        0 Jan 25 14:28  sangfor/
drwxr-xr-x 1 徐日晞 197609        0 Sep 16  2024  source/
lrwxrwxrwx 1 徐日晞 197609       63 Jun 20 00:38  「开始」菜单 -> '/c/Users/徐日晞/AppData/Roaming/Microsoft/Windows/Start Menu'/
-rw-r--r-- 1 徐日晞 197609 10946424 Jul 16  2022  新高二暑假讲义12讲（编辑精美，暑假上课必备）.rar
-rw-r--r-- 1 徐日晞 197609 27285162 Apr 17  2022  生物创新设计（部分）.zip
```
- 第一个字符：文件类型 d文件夹、l链接文件、-普通文件
- 2-4个字符：文件所有者的权限 r读w写x执行，-表示没有该权力
- 5-7 所属用户组（group）权限
- 8-10 其他用户（others）权限
- 链接数 1表示只有一个文件名指向这个文件
- 文件所有者
- 文件所属组
- 文件大小
- 最后修改时间
访问、修改某个file必须要有各级父亲目录的执行权限
- ``mv old_address new_address``
- ``cp old_address new_address``
  例如$ cp 测试.txt /d/a_university
其中测试.txt在我当前工作目录下
- ``rm old_address(-r递归删除路径所有内容)``
- 删除文件``makedir "dir_name"``
  或者``makedir dir_name空格用转义字符\ ``
### 文件的输入输出
`` <file1 >file2``<表示这个程序的输入为file1的内容，>表示输出到file2
- ``echo hello > hello.txt``把hello写入hello.txt
- ``cat hello.txt``
- ``cat < hello.txt >hello2.txt``把前一个文件内容写入hello2.txt。
注意>>表示覆盖，只用一个大于号写两次hello第二次无效

``|运算符`` 将左边程序的输出作为输入输出到右边的程序中去
例如 ``ls -l | tail -n2 > test.txt``ls -l的输出作为tail函数的输入，n后面是参数，表示取最后多少line

### cs61a
``` cd D:/a_university/python/CS61A/lab00```进入对应lab的文件夹

``` python ok --local -q python-basics -u```做test文件夹里面的内容，通过测试，解锁

``` python ok --local```编译你的程序