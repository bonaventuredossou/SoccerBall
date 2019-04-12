参考文档: https://github.com/deep-diver/Soccer-Ball-Detection-YOLOv2/blob/master/YOLOv2-Train.ipynb
第一步，自建训练集:
标注工具链接: https://github.com/tzutalin/labelImg
1. 下载足球比赛视频
2. 截取视频素材图片
3. 使用标注工具对图片中的足球进行标注
4. 将image存在'./images/', annotations存放在'./annotations/'

第二步，安装darkflow:
git clone https://github.com/thtrieu/darkflow.git
cd darkflow
pip install .

第三步，下载pre-trained weights:
https://doc-08-as-docs.googleusercontent.com/docs/securesc/u4r830p1ivt4lggvj8062qva9ocv8eet/ph168enirb8mhp5pol44hbdqesiaifu6/1554127200000/16010642207042931662/01173952668393670523/0B1tW_VtY7oniTnBYYWdqSHNGSUU?e=download&nonce=9niuhviar6r74&user=01173952668393670523&hash=qecataaiu38asnfps7iurdmodkob21l0

第四步，跑训练代码:
python3 train.py

第五步, 跑预测代码:
python3 video_detect.py
