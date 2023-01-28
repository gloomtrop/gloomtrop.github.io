---
title: "News Recommendation system at Riedia"
excerpt: "Bringing Swedish news to internationals using next generation machine translation🆎🉐"
toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
header:
  image: /assets/images/riedia_images/riedia_front.png
  teaser: assets/images/riedia_images/riedia_front.png
  actions:
    - label: "View website"
      url: "https://www.riedia.com/"
sidebar:
  - title: "Role"
    text: "Machine Learning Engineer"
  - title: "Responsibilities"
    text: "Researched, developed and implemented a news recommender system for Riedia"
gallery:
  - url: /assets/images/unsplash-gallery-image-1.jpg
    image_path: assets/images/unsplash-gallery-image-1-th.jpg
    alt: "placeholder image 1"
  - url: /assets/images/unsplash-gallery-image-2.jpg
    image_path: assets/images/unsplash-gallery-image-2-th.jpg
    alt: "placeholder image 2"
  - url: /assets/images/unsplash-gallery-image-3.jpg
    image_path: assets/images/unsplash-gallery-image-3-th.jpg
    alt: "placeholder image 3"
---
## Previous recommender system at Riedia
The team at Riedia in their start-up phase used a Content-based filter to find similar articles to the articles the user have read. The other type is by recommending news articles which a similar user have read to the user which is recommended. This approach is straight forward and have been the staple for recommender systems until deep learning. Therefore, wanted to explore this and they choose me to research this and come up with a solution.
![png](/assets/images/riedia_images/recommender.png)

## New era of recommender systems leverage the dynamic nature of Deep learning
One type of news recommender system proposed by research is a click predictor. Probability over $0.5 := "User is interested in the news article"$ while below $0.5 := "User is not interested in the news article"$. Furthermore, the model is feeded with articles which the user have read and one candidate news article. These articles are then encoded into a representation vector. How to represent these vector is by combining the title and the body using Glove embeddings for each word. 
![png](/assets/images/riedia_images/model1.png)

### ”Neural News Recommendation with Multi-Head Self-Attention”
The model which was implemented was based on the paper ["Neural News Recommendation with Multi-Head Self-Attention"](https://aclanthology.org/D19-1671/).
![png](/assets/images/riedia_images/model2.png)

## The MIND competition
![png](/assets/images/riedia_images/leaderboard.png)
![png](/assets/images/riedia_images/dataset.png)

## Results
### Measuring of accuracy
![png](/assets/images/riedia_images/results1.png)

### Calculating the latency using GPU and CPU for interference
![png](/assets/images/riedia_images/results2.png)

## Next step
![png](/assets/images/riedia_images/development.png)

## Possible use-case for Bandits
![png](/assets/images/riedia_images/bandit1.png)
![jpg](/assets/images/riedia_images/bandit2.jpg)
