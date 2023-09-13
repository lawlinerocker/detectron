<h1>Object Detection by using Detectron</h1>
This is part of the project for object detection by using detectron2 to help learning how to validate and evaluated all the datas that we put in through the zoo model.


In earlys, The main idea is to extract the data from the reciept, which help the POS to store the data into the databases.Due to early year in Bachelor's degree,turns out it's only a object detection.
![test1](https://github.com/lawlinerocker/detectron/assets/38174412/444b575f-d9c3-4026-8a8c-52539891fe3b)
![test2](https://github.com/lawlinerocker/detectron/assets/38174412/f7928a0a-25f4-4939-a48d-71d3a71314b3)
![test](https://github.com/lawlinerocker/detectron/assets/38174412/ba6aa0ed-6bd8-4743-9385-4284a8fcd798)


We were using the polygon x,y to extract the object. In this case reciepts


We have some troubles about how we runs out of datasets to validation and evaluated so we try to multiply our datasets by using the technique such as : Motion Blur,Gaussian etc. and exports all of the region data (polygon x,y) 
into .json to let the model learning the new datasets.



There's the result in "folder1"
![Screenshot_2022-04-04_011811](https://github.com/lawlinerocker/detectron/assets/38174412/53df81c2-56e6-4e44-a885-3c8479fccbfc)![Screenshot_2022-04-04_011614](https://github.com/lawlinerocker/detectron/assets/38174412/1904a5da-cc25-430a-b8b3-1921bb148642)![Screenshot_2022-04-04_011714](https://github.com/lawlinerocker/detectron/assets/38174412/e19e46ba-e5a1-45fd-866a-a30b90706f50)![Screenshot_2022-04-04_011725](https://github.com/lawlinerocker/detectron/assets/38174412/7a2a8b0d-f1a1-43aa-866c-823a8483feb7)







Here are somes of the results from the last time config was from 1,200 datasets and 1,500 iterations with no CUDA:
![unknown](https://github.com/lawlinerocker/detectron/assets/38174412/9d194c11-b161-4b2e-a181-2172008d787a)
![a](https://github.com/lawlinerocker/detectron/assets/38174412/ccffc22d-4600-48d7-a250-05a4989c7c73)
![4](https://github.com/lawlinerocker/detectron/assets/38174412/8098f465-abb8-43cb-aaa3-806241a99a8a)
![51](https://github.com/lawlinerocker/detectron/assets/38174412/95bd7081-e44e-40b8-aa13-49777ad6535a)
![465](https://github.com/lawlinerocker/detectron/assets/38174412/8e660d66-1322-46c1-aac5-2eb31864faac)




With CUDA On:
![fasd](https://github.com/lawlinerocker/detectron/assets/38174412/3ddd5884-14ba-4364-804d-7c2509755821)
![asdasd](https://github.com/lawlinerocker/detectron/assets/38174412/61fa9f7d-7c13-4c2b-ba7f-1599ac6da24c)
![asfas](https://github.com/lawlinerocker/detectron/assets/38174412/a988b5c4-e074-45fb-b259-4a4093f3a99f)
![gasdasd](https://github.com/lawlinerocker/detectron/assets/38174412/c19df498-457a-4c41-af1d-47b00d9c5f77)

