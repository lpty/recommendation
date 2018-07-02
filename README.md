## 推荐系统实例
#### 前言
本项目代码是在看项亮的《推荐系统实践》时的练习代码，16年上传的第一版代码结构比较随意，为了对得起这几十个star，特地重构一遍。

#### 目录
* 基于协同过滤(UserCF)的模型
* 基于隐语义(LFM)的模型
* 基于图(PersonalRank)的模型

#### 快速开始
请自行下载数据(http://grouplens.org/datasets/movielens/1m)，解压到data/目录中

* 数据预处理

    python manage.py preprocess

* 模型运行

    python manage.py [cf/lfm/prank]


#### 其他 & 博客
博客：https://blog.csdn.net/sinat_33741547/article/category/6442592

#### 历史版本
##### 2018.07.01
* 重构user_cf, lfm代码

##### 2018.07.02
* 重构personal_rank代码
