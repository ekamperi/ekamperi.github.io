---
layout: post
title:  "Have you ever heard of anyone who became ill with Covid-19?"
date:   2020-11-27 20:23:00 +0000
categories: [mathematics]
tags: ['covid-19', 'graphs', 'mathematics', 'social networks']
---


While Newton stayed home, during the outbreak of the plague, he discovered (invented?) calculus, gravity, and optics laws. Being in lockdown myself, I decided to stand on the shoulders of the giant Newton and sketch the answer to a burning question, “Do you know anyone who became seriously ill with Covid-19?” I don’t know where you come from, but this question has often been asked by conspiracists here in Greece (always in a rhetorical manner).

For fun, I constructed dozens of toy models of social networks with different topologies ([Barabási – Albert](https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model), [Watts – Strogatz](https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model), [Price](https://en.wikipedia.org/wiki/Price%27s_model), and random), with various parameterizations. The blue line in the following plot is the percentage of people in the network infected with Covid-19. The orange line represents the percentage of people who heard that some friend of theirs has become ill with Covid-19. Even with favorable assumptions about the transmission of the virus (and, therefore, its message), the vast majority of plots had a tail at their origin.

<p align="center">
 <img style="width: 80%; height: 80%" src="{{ site.url }}/images/heard_of_covid1.png" alt="Covid-19 transmission in a social network">
</p>

This reflects the fact that initially, the disease has a low prevalence in the general population and that most people have relatively few acquaintances. Some have an intermediate number and just a tiny fraction many. For people to have heard of some of their friends being ill during the first stage of the pandemic, either the disease should have had a very high prevalence from day 1, or the social network should have been exceptionally densely connected (in the limit case, everyone being friends with anyone). Just like the following [complete graph](https://en.wikipedia.org/wiki/Complete_graph):

<p align="center">
 <img style="width: 40%; height: 40%" src="{{ site.url }}/images/complete_graph.png" alt="Covid-19 complete graph">
</p>

However, real social netowrks are more likely to resemble the following topologies:

<p align="center">
 <img style="width: 70%; height: 70%" src="{{ site.url }}/images/heard_of_covid2.png" alt="Covid-19 transmission in a social network">
</p>

And here you can see the transmission of the virus in a sample network:
<p align="center">
 <img style="width: 70%; height: 70%" src="{{ site.url }}/images/heard_of_covid3.png" alt="Social networks with various topologies">
</p>

**Epimyth**: If you haven’t heard of any of your acquaintances being seriously ill with Covid-19 at the start of the pandemic, it’s because there weren’t many cases. If, as the days pass by, you still don’t hear of anyone, it’s because you don’t have any friends :P
