---
layout: post
title:  "How to convert DICOM transfer syntax"
date:   2020-05-16
categories: [radiotherapy]
tags: ['DICOM', 'radiotherapy']
---

This post is going to be very esoterical, yet I'm writing it for my future self and perhaps for you. So, I was trying to import the MRI of a patient with locally advanced tongue cancer into the Varian Eclipse treatment planning system (version 15.1). Sadly, the import failed with tons of errors of the form:

{% raw %}
> Unsupported 'Transfer Syntax UID' (0002,0010) 'Explicit Big Endian'.<br>
> Could not convert DICOM stream: SOP Instance UID: 1.2.840.113619.2.134.... [MRI]
{% endraw %}

Quoting from the DICOM standard documentation, *Transfer Syntax* is a set of encoding rules able to represent one or more abstract syntaxes unambiguously. In particular, it allows communicating applications to negotiate common encoding techniques they both support (e.g., byte ordering, compression, etc.).

So, apparently Varian Eclipse does not like *Explicit Big Endian* transfer syntax. For the non-tech savvy, when we write the number "123", we write the most significant digit (e.g. 1) first. This is a big endian representation. If we'd like to write the "123" number in little endian format, we'd write "321". In order to figure out the supported transfer syntaxes oF Eclipse, I exported the treatment course a random patient, and, in the export's options dialog, the valid transfer syntaxes were listed as below:

<p align="center">
 <img style="width: 60%; height: 60%" src="{{ site.url }}/images/Varian_Eclipse_Export.png">
</p>

In layman's terms, the MRI data are written in big-endian format (e.g. "123"), but Eclipse can only read data written in little-endian format (e.g. "321"). Therefore, we need to convert from big-endian to little-endian.

I tried to do the conversion of my patient's DICOM files with `dcmconv`, but the latter would output other errors. I finally gave up on dcmconv and tried `gdcmconv`. The option `-w`, that is supposed to decompress a DICOM file, rewrote my files in `Explicit VR Little Endian` format:

{% highlight R %}
{% raw %}
~/dicom$ ls
1.dcm  11.dcm  13.dcm  15.dcm  17.dcm  19.dcm  20.dcm  22.dcm  24.dcm  26.dcm  28.dcm  3.dcm  5.dcm  7.dcm
9.dcm  10.dcm  12.dcm  14.dcm  16.dcm  18.dcm  2.dcm   21.dcm  23.dcm  25.dcm  27.dcm  29.dcm  4.dcm  6.dcm
8.dcm
~/dicom$
~/dicom$ for f in *.dcm; do gdcmconv -w -i "$f" -o "$f".converted; done
~/dicom$
~/dicom$ sdiff -s <(gdcmdump 1.dcm) <(gdcmdump 1.dcm.converted)
(0002,0000) UL 196                           | (0002,0000) UL 226
(0002,0001) OB 00\00                         | (0002,0001) OB 00\01
(0002,0010) UI [1.2.840.10008.1.2.2]         | (0002,0010) UI [1.2.840.10008.1.2.1]
(0002,0012) UI [1.2.840.114257.1123456]      | (0002,0012) UI [1.2.826.0.1.3680043.2.1143.107.104.103.115.2.
(0002,0013) SH [DICOM 3.0 ]                  | (0002,0013) SH [GDCM 2.8.4]
(0002,0016) AE [MR1 ]                        | (0002,0016) AE [gdcmconv]
# Used TransferSyntax: 1.2.840.10008.1.2.2   | # Used TransferSyntax: 1.2.840.10008.1.2.1
~/dicom$
~/dicom$ rm *.dcm
~/dicom$                                                         
{% endraw %}
{% endhighlight %}

I tried again to import the converted MRI DICOM files and the import succeeded with no errors whatsoever. The MRI images appeared perfectly fine and the simulation CT/MRI image registration process was performed unproblematically.
