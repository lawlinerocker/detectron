This is part of the project for object detection by using detectron2 to help learning how to validate and evaluated all the datas that we put in through the zoo model.


In earlys, The main idea is to extract the data from the reciept, which help the POS to store the data into the databases.Due to early year in Bachelor's degree,turns out it's only a object detection.


We were using the polygon x,y to extract the object. In this case reciepts


We have some troubles about how we runs out of datasets to validation and evaluated so we try to multiply our datasets by using the technique such as : Motion Blur,Gaussian etc. and exports all of the region data (polygon x,y) 
into .json to let the model learning the new datasets.



There's the result in "folder1"
![Screenshot_2022-04-04_011811](https://github.com/lawlinerocker/detectron/assets/38174412/b8eb45fb-32c6-4614-9f9c-b421f086cf4e)![Screenshot_2022-04-04_011714](https://github.com/lawlinerocker/detectron/assets/38174412/d13eedea-af2e-473d-9656-0db127b510d4)(https://github.com/lawlinerocker/detectron/assets/38174412/d8e2afed-a13b-4cf9-bfe1-6fcf31392d33)![Screenshot_2022-04-04_011725]



Here are somes of the results from the last time config was from 1,200 datasets and 1,500 iterations with no CUDA:


![unknown](https://github.com/lawlinerocker/detectron/assets/38174412/225d5d21-9c61-4682-9089-530ee96544e2)![Screenshot_2022-04-04_011811](https://github.com/lawlinerocker/detectron/assets/38174412/ecc75ff6-530d-4ff7-a7a7-7eebd4b3fc58)![4](https://github.com/lawlinerocker/detectron/assets/38174412/c68f64d5-ec95-4d97-a10f-744b82a92659)
![51](https://github.com/lawlinerocker/detectron/assets/38174412/a59e91ba-f501-47f9-843d-b66127249c32)
![465](https://github.com/lawlinerocker/detectron/assets/38174412/d6c0577f-1423-4457-9ba7-fc52be23edab)


With CUDA :

![fasd](https://github.com/lawlinerocker/detectron/assets/38174412/535d1ff8-4a9b-4d0b-8a8d-627a992e7a34)
![asdasd](https://github.com/lawlinerocker/detectron/assets/38174412/7facef4e-630d-48da-8933-ac53b6e2b6d5)
![asfas](https://github.com/lawlinerocker/detectron/assets/38174412/5ba70926-9474-41c2-95e6-364c2dad86ea)
![gasdasd](https://github.com/lawlinerocker/detectron/assets/38174412/8601ca8b-4c8d-4f03-8a3d-600509801fc2)

