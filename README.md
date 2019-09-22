# Image AI

#### Image AI for content publishing platforms.

#### AI experiments for content publishing tools. Identify politics figures (obama, biden, trump, hillary etc). Automatically crop images by keeping subject at focus. Gather Image caption and labelling. 

#### Goal
This is an experiment to utilize Image AI & Machine Learning in content publishing platforms such as CMS (Content management systems), curation tools, front end applications or even in backen API.

### Features
- Detect politicial figures (obama, biden, trump, hillary etc) using face detection.
- Face priority image crop
- Predict what image it is. Detect object and it's name.
- Predict what image it is using several different image datasets. Detect object and it's name.
- Detect scene or environment info.

=============================================
##### Deploy (Using Docker Compose)
`cd docker`

`docker-compose up -d`

##### Deploy (Native)
`cd docker`

`pip install -r requirements.txt `

`cd ../scripts`

`python web-server.py`

Visit http://localhost:5002.


### Endpoints

##### Face priority image crop
`curl 'http://localhost:5002/smart_crop?url=<YOUR_IMAGE_URL>'`

##### Predict what image it is. Detect object and it's name.
`curl 'http://localhost:5002/predict_image?url=<YOUR_IMAGE_URL>'`

##### Predict what image it is using several different image datasets. Detect object and it's name.
`curl 'http://localhost:5002/predict_image_all?url=<YOUR_IMAGE_URL>'`

##### Detect scene or environment info.
`curl 'http://localhost:5002/image_caption?url=<YOUR_IMAGE_URL>'`

##### Detect scene or environment info.
`curl 'http://localhost:5002/image_caption?url=<YOUR_IMAGE_URL>'`

##### Detect politicial figures (person) using face detection.
`curl 'http://localhost:5002/detect_person?url=<YOUR_IMAGE_URL>'`



##### THIS IS WORK IN PROGRESS.

#### TO DO
- .... huge list.

##### My other work related to AI [Content-AI](https://github.com/nycdidar/Content-AI)


> FULL CREDIT GOES TO EVERYONE INVOLVED IN ML/AI FIELD. WE WOULD BE IN STONE AGE WITHOUT THEIR DEDICATION AND HARD WORK.