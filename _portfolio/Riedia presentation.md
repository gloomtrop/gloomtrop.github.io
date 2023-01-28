

NEWS

RECOMMEND

ER SYSTEM

FOR RIEDIA

BY AXEL ROOTH





• INTRO

TABLE OF

•

APPROACH

• SOLUTION

• RESULTS

CONTENTS

• FUTURE WORK





INTRO

¡ The fundamental types of

recommender systems:





NEW

APPROACH

¡ New era of recommender

systems leverage the

dynamic nature of Deep

learning

¡ Solve issues of cold-start,

static models and

complexity etc.





SOLUTION -

MODEL

¡ ”Neural News

Recommendation with

Multi-Head Self-Attention”

Source:

https://wuch15.github.io/paper/EMNLP2019-





SOLUTION -

D ATA

¡ The MIND competition

Source: https://msnews.github.io/index.html





RESULTS

**AUC**

**MRR**

**NDCG@5**

0.2912

**NDCG@10**

0.3625

SMALL

LARGE

0.6127

0.6609

0.2697

0.313

0.3451

0.4094

**ΔAccuracy +7.9%**

**+16.1%**

**+18.5%,**

**+12.9%**

¡ Measuring of accuracy and

calculation latency





RESULTS

**Batch size: Total Load Load Time Calc. Time Calc. Time**

**256**

**samples**

**Time**

**/ sample**

**/ sample**

0.0163 sec

0.00082 sec

**+20x**

CPU

0.00351 sec 1.37e-05

sec

4.17 sec

0.21 sec

**+20x**

¡ Measuring of accuracy and

calculation latency

GPU

0.00351 sec 1.37e-05

sec

**ΔFaster**

***Both use***

***CPU***

***Both use***

***CPU***





FUTURE WORK





SHORT TERM (1- 2 MONTHS PERIOD)

Fine-tune model Implement the

for Riedia’s users model in

and their production in

corresponding real-time or in

news articles batch mode





LONG TERM (5-6 MONTHS PERIOD)





EXTRA: MULTI-

ARMED

BANDITS

¡ Used by Netflix, Spotify

and Google etc.

¡ Scalable method for

testing and deployment

Source: https://cloud.google.com/blog/products/ai-machine-learning/how-to-

build-better-contextual-bandits-machine-learning-models





EXTRA: MULTI-

ARMED

BANDITS

¡ Used by Netflix, Spotify

and Google etc.

¡ Scalable method for

testing and deployment

Source: https://towardsdatascience.com/bandits-for-recommender-system-

optimization-1d702662346e





“Experiments based on multi-armed bandits are typically much more efficient than “classical” A-B

experiments based on statistical-hypothesis testing. They’re just as statistically valid, and in many

circumstances, they can produce answers far more quickly. They’re more efficient because they move

traffic towards winning variations gradually, instead of forcing you to wait for a “final answer” at the end

of an experiment. They’re faster because samples that would have gone to obviously inferior variations

*can be assigned to potential winners. The extra data collected on the high-performing variations can*

help separate the “good” arms from the “best” ones more quickly.

*Basically, bandits make experiments more efficient, so you can try more of them. You can also allocate*

*a larger fraction of your traffic to your experiments, because traffic will be automatically steered to better*

performing pages.”

By Steven L. Scott, Sr. Economic Analyst of Google Analytics





THANK YOU FOR YOUR

ATTENTION!

