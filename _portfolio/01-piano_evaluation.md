---
title: "Piano music performance evaluation system"
excerpt: "A research collaboration is studying how to identify and analyze styles of piano performances in order to improve playing style skills."
toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
header:
  image: /assets/images/piano_ai/piano_ai.jpeg
  teaser: assets/images/piano_ai/piano_ai.jpeg
sidebar:
  - title: "Role"
    text: "Machine Learning Contributor"
  - title: "Responsibilities"
    text: "Contributing in making the ML model used to explain professional pianists different style and what they can
improvement at."
gallery1:
  - url: /assets/images/piano_ai/data.png
    image_path: assets/images/piano_ai/data.png
    alt: "placeholder image 1"
  - url: /assets/images/piano_ai/model.png
    image_path: assets/images/piano_ai/model.png
    alt: "placeholder image 2"
gallery2:
  - url: /assets/images/piano_ai/outcome.png
    image_path: assets/images/piano_ai/outcome.png
    alt: "placeholder image 3"
    
---

## Motivation

The proposed application aims to assist piano teachers in evaluating their student's performance by providing a visual comparison between the student's performance and an ideal performance. The application works by first having the teacher play the piano while their performance is displayed in MIDI bars on one monitor. The ideal performance can be set as the teacher's performance or a default version. Then, the student plays and their performance is also displayed in MIDI bars, with a purple bar representing the student's performance and a yellow bar representing the ideal performance. The application also provides style labels, shown in a spider chart, to help the teacher and student easily identify and understand differences in their performance. Users can also choose to display specific dimensions of the performance in a detailed graph with fewer labels. This allows the teacher to easily explain and pinpoint specific areas for improvement, and for the student to evaluate their own progress and compare with previous practice.

## Background

I joined an open-source project led by my professor from the Explainable AI course at Seoul National University. The team had already established a clear roadmap for the project and I was excited to be a part of it. During weekly meetings, we discussed progress and strategies for all aspects of the project. I worked closely with another ML engineer to develop the model architecture necessary for providing explanations. Additionally, data was being collected through an outsourcing company, with new data arriving weekly consisting of tagged piano performances and ratings for 25 categories that were deemed to represent a piano performer's playing style. The final model we developed was a RNN-based model as shown below.

{% include gallery id="gallery1" %}

## Model and performance

The proposed method involves preprocessing recorded piano performances to align them with the score using Performance Error Detection from Nakamura. Additionally, ground truth for the crowdsourced data was obtained by using the apex of the probability density function of the results. To analyze the performance, RNN-based GRU/LSTM models were used to generate output of styles scorings. This resulted in a strong correlation between the ground truth and the predicted values, with the highest correlation being 0.787.

## Outcome

The team showed that it is possible to play whilst visualising the explaination on a monitor. To better visualise this the team opted into a multidimensional display where the piano performer could see how much their play deviated from the prefered. I am very proad of what the end-result became and the team did an amazing. 

{% include gallery id="gallery2" %}

## Reflection

Being part of this incredible interdisciplinary team was an experience in itself. Collaborating between computer scientist to musicians creates a dynamic creativity that is hard to match. Furthermore, the need for tools to help createurs like piano performers are much appreciated and more is needed. This project demonstrates that it is possible. However, the project had its clear limitations, both in the technical aspect and unreasonable demands from above. Firstly, before I had joined much of the tagging had started. The question if it is possible the distinguish these 25 classes where not made. This resulted in poor tagging with almost random predictability. This underlying issue made it both difficult for the model to generalize while it also created wrong explanations. The second issue was that there was a constant change in demands from the superiors which made it increasingly frustrating. Therefore, I unfortunately decided to depart earlier from the project than I wanted. In the end however, I learned a lot in the realm of signal processing and what machine learning can provide to create next generation tools for creativity and learning. 🎹🕺



