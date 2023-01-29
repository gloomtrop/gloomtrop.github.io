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

{% include gallery1 %}

## Model and performance

The proposed method involves preprocessing recorded piano performances to align them with the score using Performance Error Detection from Nakamura. Additionally, ground truth for the crowdsourced data was obtained by using the apex of the probability density function of the results. To analyze the performance, RNN-based GRU/LSTM models were used to generate output of styles scorings. This resulted in a strong correlation between the ground truth and the predicted values, with the highest correlation being 0.787.

## Outcome

To better visualise this the team opted into a multidimensional display where the piano performer could see how much their play deviated from the prefered.

{% include gallery2 %}
![png](/assets/images/piano_ai/outcome.png)




