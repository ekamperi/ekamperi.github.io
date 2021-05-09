---
layout: default
title: About
---

<img src="{{ site.url }}/images/me.jpg" width="100" style="border-radius: 50%">

My name is Stathis Kamperis, and I live in Greece. I am a [radiation oncologist](https://en.wikipedia.org/wiki/Radiation_oncologist) and physicist with a master degree in [computational physics](https://en.wikipedia.org/wiki/Computational_physics) and another one in [medical research methodology](https://en.wikipedia.org/wiki/Medical_research). I currently work on my Ph.D. thesis, which involves applying machine learning in [radiomics](https://en.wikipedia.org/wiki/Radiomics) in head neck cancer. You can [check my detailed bio in LinkedIn](https://www.linkedin.com/in/stathis-kamperis/).

# Hands-on experience
## Google Summer of Code
### CERN's Geant4 high energy physics simulation framework
<p float="left">
<img style="width: 13%; height: 13%" src="{{ site.url }}/images/logos/google_logo.png" alt="Google logo">
<img style="width: 10%; height: 10%" src="{{ site.url }}/images/logos/cern_logo.png" alt="CERN logo">
<img style="width: 15%; height: 15%" src="{{ site.url }}/images/logos/geant4_logo.png" alt="Geant4 logo">
<img style="width: 15%; height: 15%" src="{{ site.url }}/images/logos/solaris_logo.png" alt="Solaris logo">
</p>

I ported Geant4 to Solaris, used DTrace to debug performance issues, and implemented a smart particle bunching algorithm to reduce cache misses. Detailed information: https://ekamperi.github.io/Geant4

### NetBSD math library
<p float="left">
<img style="width: 13%; height: 13%" src="{{ site.url }}/images/logos/google_logo.png" alt="Google logo">
<img style="width: 11%; height: 11%" src="{{ site.url }}/images/logos/netbsd_logo.png" alt="NetBSD logo">
</p>

I audited and extended NetBSD's operating system math library. Specifically:
*	Wrote 80 test programs and 260 test cases for math.h, fenv.h, float.h, complex.h and tgmath.h interfaces (C, C++)
*	Profiled the entire math library in terms of accuracy and speed (C)
*	Added fenv.h support for amd64 and i386 CPU architectures (committed to official source tree)
*	Implemented experimental fenv.h support for sparc64 and m68k CPU architectures
Detailed information: https://ekamperi.github.io/Mathlib/

### DragonFlyBSD
<p float="left">
<img style="width: 13%; height: 13%" src="{{ site.url }}/images/logos/google_logo.png" alt="Google logo">
<img style="width: 20%; height: 20%" src="{{ site.url }}/images/logos/dflybsd_logo.png" alt="DragonFlyBSD logo">
<img style="width: 11%; height: 11%" src="{{ site.url }}/images/logos/netbsd_logo.png" alt="NetBSD logo">
<img style="width: 11%; height: 11%" src="{{ site.url }}/images/logos/linux_logo.png" alt="Linux logo">
<img style="width: 15%; height: 15%" src="{{ site.url }}/images/logos/pgsql_logo.png" alt="PostgreSQL logo">
</p>

I audited DragonflyBSD operating system against the latest POSIX and C99 standards. Specifically:
*	Detected and fixed many conformance bugs on both DragonFlyBSD and other systems as well, such as NetBSD and GNU C library
*	Ported POSIX message queues implementation from NetBSD to DragonFlyBSD
*	Wrote a web user interface in PHP for tracking conformance status, backed by PostgreSQL

# Academic
[You can read my first master thesis](http://ikee.lib.auth.gr/record/289589/files/GRI-2017-19273.pdf?version=1), where I used [Geant4](https://geant4.web.cern.ch/node/1) to run Monte Carlo simulations of external photon beams in human phantoms. Regrettably, the text is in Greek, but you might find the images and the code excerpts useful.

[You may also read my latest master thesis](https://ekamperi.github.io/mrm_thesis/abstract.html), where I analyzed the complexity of [Volumetric Modulated Arc Therapy](https://en.wikipedia.org/wiki/External_beam_radiotherapy#Volumetric_Modulated_Arc_Therapy) prostate plans. In the first part, I examined various complexity metrics with [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) and [Mutual Information Analysis](https://en.wikipedia.org/wiki/Mutual_information). In the second part, I developed both a linear and a [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) logistic regression model to predict complexity by clinical and dosimetric plan features.

[This is the list of my publications](https://scholar.google.gr/citations?hl=en&user=HMbAeKQAAAAJ) on medical journals.
