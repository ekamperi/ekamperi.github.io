---
layout: default
title: About
---

<img src="{{ site.url }}/images/me.jpg" width="100" style="border-radius: 50%">

My name is Stathis Kamperis, and I live in Greece. I am a [radiation oncologist](https://en.wikipedia.org/wiki/Radiation_oncologist) and physicist with a master degree in [computational physics](https://en.wikipedia.org/wiki/Computational_physics) and another one in [medical research methodology](https://en.wikipedia.org/wiki/Medical_research). I currently work on my Ph.D. thesis, which involves applying machine learning in [radiomics](https://en.wikipedia.org/wiki/Radiomics) in head neck cancer. You can [check my detailed bio in LinkedIn](https://www.linkedin.com/in/stathis-kamperis/).

# Hands-on experience

## ⇝ Machine-learning model for virtual QA/verification

<p float="left">
<img style="width: 20%; height: 20%" src="{{ site.url }}/images/logos/keras_logo.png" alt="Keras logo">
<img style="width: 7%; height: 7%" src="{{ site.url }}/images/logos/tensorflow_logo.png" alt="Tensorflow logo">
<img style="width: 11%; height: 11%" src="{{ site.url }}/images/logos/dicom_logo.jpg" alt="DICOM logo">
<img style="width: 9%; height: 9%" src="{{ site.url }}/images/logos/sqlite_logo.png" alt="SQLite logo">
</p>

I developed an end-to-end machine-learning model for predicting quality assurance and verification outcomes in radiation oncology departments. It uses as features a diverse set of clinical and dosimetric features, such as complexity metrics, dose distributions, clinical and dosimetric traits. The extraction and calculation of many of these features is done by "rteval", an evaluation framework I wrote in Python.

<p align="center">
<img style="width: 80%; height: 80%" src="https://ekamperi.github.io/mrm_thesis/images/dnn_arch.png" alt="Machine learning model">
</p>

## ⇝ Feature-engineering framework for radiation oncology treatment plans

<p float="left">
<img style="width: 10%; height: 10%" src="{{ site.url }}/images/logos/python_logo.png" alt="Python logo">
<img style="width: 9%; height: 9%" src="{{ site.url }}/images/logos/linux_logo.png" alt="Linux logo">
</p>

I developed a specialized feature engineering framework for radiation oncology departments.
* http://test.ogkologia.gr/rteval/ (Python)
* A web application for the statistical analysis and presentation of the extracted data (R Shiny)

## ⇝ DICOM viewer

<p float="left">
<img style="width: 6%; height: 6%" src="{{ site.url }}/images/logos/qt_logo.png" alt="Qt logo">
<img style="width: 9%; height: 9%" src="{{ site.url }}/images/logos/opengl_logo.png" alt="Opengl logo">
<img style="width: 11%; height: 11%" src="{{ site.url }}/images/logos/dicom_logo.jpg" alt="DICOM logo">
</p>

I developed a cross-platform, OpenGL-accelerated, multi-threaded DICOM viewer (C++, OpenGL, Qt).

For more details, check [this link](https://ekamperi.github.io/Volmetrics).

## Google Summer of Code
### ⇝ Profiling CERN's Geant4 high energy physics simulation framework
<p float="left">
<img style="width: 9%; height: 9%" src="{{ site.url }}/images/logos/google_logo.png" alt="Google logo">
<img style="width: 7%; height: 7%" src="{{ site.url }}/images/logos/cern_logo.png" alt="CERN logo">
<img style="width: 18%; height: 18%" src="{{ site.url }}/images/logos/geant4_logo.png" alt="Geant4 logo">
<img style="width: 14%; height: 14%" src="{{ site.url }}/images/logos/solaris_logo.png" alt="Solaris logo">
</p>

*	Ported [Geant4]((https://geant4.web.cern.ch/node/1)) to Solaris.
*	Used Solaris advanced built-in profiling tools to profile Geant4 with respect to cache misses, branch mispredictions, and total execution time (bash, awk, DTrace, C, C++).
*	Used DTrace's speculative tracing to successfully debug an unstable behavior of Geant4's particle tracking manager (DTrace, C, C++).

For more details, check [this link](https://ekamperi.github.io/Geant4).

<hr>

### ⇝ Auditing and extending NetBSD's math library
<p float="left">
<img style="width: 9%; height: 9%" src="{{ site.url }}/images/logos/google_logo.png" alt="Google logo">
<img style="width: 11%; height: 11%" src="{{ site.url }}/images/logos/netbsd_logo.png" alt="NetBSD logo">
<img style="width: 10%; height: 10%" src="{{ site.url }}/images/logos/amd64_logo.png" alt="AMD64 logo">
</p>

I audited and extended NetBSD's operating system math library. Specifically:
*	Wrote 80 test programs and 260 test cases for math.h, fenv.h, float.h, complex.h and tgmath.h interfaces (C, C++).
*	Profiled the entire math library in terms of accuracy and speed (C).
*	Added fenv.h support for amd64 and i386 CPU architectures (committed to official source tree).
*	Implemented experimental fenv.h support for sparc64 and m68k CPU architectures.

For more details, check [this link](https://ekamperi.github.io/Mathlib/).

<hr>

### ⇝ Auditing DragonFlyBSD's POSIX/C99 conformance
<p float="left">
<img style="width: 9%; height: 9%" src="{{ site.url }}/images/logos/google_logo.png" alt="Google logo">
<img style="width: 20%; height: 20%" src="{{ site.url }}/images/logos/dflybsd_logo.png" alt="DragonFlyBSD logo">
<img style="width: 11%; height: 11%" src="{{ site.url }}/images/logos/netbsd_logo.png" alt="NetBSD logo">
<img style="width: 11%; height: 11%" src="{{ site.url }}/images/logos/linux_logo.png" alt="Linux logo">
<img style="width: 15%; height: 15%" src="{{ site.url }}/images/logos/pgsql_logo.png" alt="PostgreSQL logo">
</p>

I audited DragonflyBSD operating system against the latest POSIX and C99 standards. Specifically:
*	Detected and fixed many conformance bugs on both DragonFlyBSD and other systems as well, such as NetBSD and GNU C library
* Ported POSIX message queues implementation from NetBSD to DragonFlyBSD
*	Wrote a web user interface in PHP for tracking conformance status, backed by PostgreSQL

# Academic
## ⇝ Monte Carlo simulations in human phantoms (Master thesis in Computational Physics)
<p float="left">
<img style="width: 18%; height: 18%" src="{{ site.url }}/images/logos/geant4_logo.png" alt="Geant4 logo">
<img style="width: 12%; height: 12%" src="{{ site.url }}/images/logos/monte_carlo_logo.png" alt="Monte Carlo logo">
</p>
[You can read my first master thesis](http://ikee.lib.auth.gr/record/289589/files/GRI-2017-19273.pdf?version=1), where I used [Geant4](https://geant4.web.cern.ch/node/1) to run Monte Carlo simulations of external photon beams in human phantoms. Regrettably, the text is in Greek, but you might find the images and the code excerpts useful.

<hr>

## ⇝ Complexity analysis of VMAT prostate plans (Master thesis in Medical Research Methodology)
<p float="left">
<img style="width: 10%; height: 10%" src="{{ site.url }}/images/logos/r_logo.png" alt="R logo">
<img style="width: 10%; height: 10%" src="{{ site.url }}/images/logos/python_logo.png" alt="Python logo">
<img style="width: 13%; height: 13%" src="{{ site.url }}/images/logos/scikit_logo.png" alt="Scikit logo">
<img style="width: 10%; height: 10%" src="{{ site.url }}/images/logos/jupyter_logo.png" alt="Jupyter logo">
<img style="width: 10%; height: 10%" src="{{ site.url }}/images/logos/mathematica_logo.png" alt="Mathematica logo">
</p>

[You may also read my latest master thesis](https://ekamperi.github.io/mrm_thesis/abstract.html), where I analyzed the complexity of [Volumetric Modulated Arc Therapy](https://en.wikipedia.org/wiki/External_beam_radiotherapy#Volumetric_Modulated_Arc_Therapy) prostate plans. In the first part, I examined various complexity metrics with [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) and [Mutual Information Analysis](https://en.wikipedia.org/wiki/Mutual_information). In the second part, I developed both a linear and a [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) logistic regression model to predict complexity by clinical and dosimetric plan features.

[This is the list of my publications](https://scholar.google.gr/citations?hl=en&user=HMbAeKQAAAAJ) on medical journals.
