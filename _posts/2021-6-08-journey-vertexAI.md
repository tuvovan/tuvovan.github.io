---
title: Journey with Vertex AI
tags: [Deep Learning, Artificial Intelligence, Vertex AI, Deployment]
style: fill
color: secondary
description: Deploying a DL model with Vertex AI (custom model).
---
<!-- 
Source: [GitHub](https://github.com/amorehead/jazz-nn)

![](https://amorehead.github.io/assets/img/jazz_nn.jpg) -->
## Introduction
As a part of the wound app development, the backend and frontend need to be considered. The initial plan was to build those things from scratch, meaning that we need to study about Model server, Web server, message broker, etc… That would take a couple of weeks or event month to make it run smoothly, so the CTO recommended this Google Vertex AI API to deploy the app directly on Google Cloud Platform.

Here I will walk through what I have done so far and the obstacles that I’m facing right now and I’m really appreciated if there is any comment or suggestion on those problems.

To use Vertex AI, make sure we got those following things:

- a Google Cloud subscription

- a trained model you want to deploy

that is!

First of all, I followed an official tutorial [here](https://cloud.google.com/vertex-ai/docs/tutorials/image-recognition-custom). This tutorial will show you how to prepare data, to train, and then deploy as well as clean the workspace after the successful deployment. 

My plan was to try the example from the official site, and then we can change their source code, apply it on our deployment (by just simply replacing their model by ours and some parameters)

Try the example from official doc
The document from GG is very informative, I believe any of you will be able to follow and have it done.

Basically, we need to set up a storage account to store all codes, data, models after training and use that for serving the model when the training is done.

If you follow the tutorial til the end, you will end up with this kind of url:

[https://storage.googleapis.com/flh/webapp/index.html](https://storage.googleapis.com/flh/webapp/index.html)

This will lead you to a very simple web page and you can try predicting six different types of flowers. 🌺 

From there, we are going to change the source code a little bit, to deploy our own model.
![](https://tuvovan.github.io/assets/img/webapp-screenshot_2x.png)

## Try with our Model
If you follow the example above, you may notice that all codes are from this link: 
```
gsutil cp gs://cloud-samples-data/ai-platform/hello-custom/hello-custom-sample-v1.tar.gz - | tar -xzv
cd hello-custom-sample
```

Explore a bit, we see that the project structure is as follow:

It has four items:

- ```trainer/```: A directory of TensorFlow Keras code for training the flower classification model.

- ```setup.py```: A configuration file for packaging the ```trainer/``` directory into a Python source distribution that Vertex AI can use.

- ```function/```: A directory of Python code for a Cloud Function that can receive and preprocess prediction requests from a web browser, send them to Vertex AI, process the prediction responses, and send them back to the browser.

- ```webapp/```: A directory with code and markup for a web app that gets flower classification predictions from Vertex AI. 

So as you can guess, we will change the code inside function folder, to fit our needs such as image_size, model name, etc.. For me, I change ```IMG_WIDTH = 224``` in line 12 only and there we go.

One more thing that needs to do to use our model instead of their model is to change the model. Their model has been trained on GCP, but our model is not. So in step 3 [Serving predictions from a custom image classification model](https://cloud.google.com/vertex-ai/docs/tutorials/image-recognition-custom/serving), we will not use the model that has been trained, we will import our model by clicking import button and follow the instruction. 

Finally, follow the remaining steps to create end-point and we will also end up with a link like the previous one. 

To try predicting an image, we can simply click on the image, but unlike the previous example, we’re gonna catch an issues as follow:

```
Prediction request failed: <class 'google.api_core.exceptions.FailedPrecondition'>: 400 The request size (1857939 bytes) exceeds 1.500MB limit. 
```

After searching the documents, I found that the error was because of the size of the requested image we want to send to the google endpoint to do the prediction. The limited size is ```1.5MB```, and the images with ```224*224``` size are normally bigger than that. [ref](https://cloud.google.com/vertex-ai/docs/predictions/online-predictions-custom-models)

Until now the problem is not solved yet, I would very appreciate if anyone can give a hand on this or any suggestions. In the mean time, I will go with developing the backend from scratch.