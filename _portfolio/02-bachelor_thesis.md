---
title: "Bachelor thesis - Predicting CPU utilization from Azure Cloud"
excerpt: "Research and developed a machine learning model to predict CPU utlization for AFRY AB."
toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
header:
  image: /assets/images/afry/afry_logo1.jpeg
  teaser: assets/images/afry/afry_logo.jpeg
sidebar:
  - title: "Role"
    text: "Researcher"
  - title: "Responsibilities"
    text: "Implemented with my co-author a machine learning model to predict CPU utilizaton."
  - title: "URL"
    text: "http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-301247"
gallery1:
  - url: /assets/images/afry/diagram1.png
    image_path: assets/images/afry/diagram1.png
    alt: "placeholder image 1"
  - url: /assets/images/afry/lstm.png
    image_path: assets/images/afry/lstm.png
    alt: "placeholder image 2"
gallery2:
  - url: /assets/images/afry/pvalue.png
    image_path: assets/images/afry/pvalue.png
    alt: "placeholder image 3"
gallery3:
  - url: /assets/images/afry/diagram2.png
    image_path: assets/images/afry/diagram2.png
    alt: "placeholder image 3"
    
---

## Overview

In this paper, I aimed to address the issue of fluctuating CPU utilization in cloud computing and its impact on cost and the environment. To mitigate this problem, I proposed the use of a long short-term memory (LSTM) machine learning model to predict future utilization. I believed that by predicting utilization up to 30 minutes in advance, companies could scale their capacity just in time and avoid unnecessary cost and harm to the environment. The study was divided into two parts: first, I compared the performance of the LSTM model with a state-of-the-art model in one-step predictions, and second, I evaluated the accuracy of the LSTM in making predictions up to 30 minutes into the future. To ensure objective analysis, I compared the LSTM with a standard RNN, which had a similar algorithmic structure. In conclusion, my results suggested that the LSTM model may have been a valuable tool for companies using public cloud to reduce cost and environmental impact.

{% include gallery id="gallery1" %}

## Model

{% include gallery id="gallery2" %}

## Results

{% include gallery id="gallery3" %}

## Reflection
