---
layout: post
title:  "Github analytics with Mathematica"
date:   2020-10-14
categories: [programming]
tags: ['Mathematica', 'programming', 'REST API']
---

## Introduction

{% highlight mathematica %}
{% raw %}
ClearAll["Global`*"];
resp = URLRead@HTTPRequest["https://api.github.com/users/ekamperi"]
{% endraw %}
{% endhighlight %}

Mathematica will respond with something like:

<p align="center">
<img style="width: 75%; height: 75%" src="{{ site.url }}/images/http_response.png" alt="HTTPResponse Mathematica">
</p>

{% highlight mathematica %}
{% raw %}
resp["Properties"]
(* {"Body", "BodyByteArray", "BodyBytes", "CharacterEncoding", \
"ContentType", "Headers", "StatusCode", "StatusCodeDescription", \
"Version"} *)

resp[{"StatusCode", "StatusCodeDescription"}]
(* <|"StatusCode" -> 200, "StatusCodeDescription" -> "OK"|> *)

rj = ImportString[ur["Body"], "RawJSON"];
rj // Keys
(* {"login", "id", "node_id", "avatar_url", "gravatar_id", "url", \
"html_url", "followers_url", "following_url", "gists_url", \
"starred_url", "subscriptions_url", "organizations_url", "repos_url", \
"events_url", "received_events_url", "type", "site_admin", "name", \
"company", "blog", "location", "email", "hireable", "bio", \
"twitter_username", "public_repos", "public_gists", "followers", \
"following", "created_at", "updated_at"} *)

rj["bio"]
(* I am a radiation oncologist and physicist. I like to build bridges \
between different scientific disciplines (medicine, physics, \
informatics). *)
{% endraw %}
{% endhighlight %}


{% highlight mathematica %}
{% raw %}
ass = <|# -> {} & /@ langs|>
(* <|"ANTLR" -> {}, "Awk" -> {}, "C" -> {}, "C++" -> {}, "CSS" -> {}, 
 "HTML" -> {}, "Java" -> {}, "M4" -> {}, "Makefile" -> {}, 
 "MATLAB" -> {}, "Objective-C" -> {}, "Perl" -> {}, "Python" -> {}, 
 "QMake" -> {}, "R" -> {}, "Ruby" -> {}, "SCSS" -> {}, "Shell" -> {}, 
 "TeX" -> {}, "XSLT" -> {}|> *)
 
f[key_, val_] := AppendTo[ass[key], val]
g[lang_, values_] := {lang, Total@values}
res = KeyValueMap[g, ass]
(* {{"ANTLR", 139}, {"Awk", 88}, {"C", 1353282}, {"C++", 167337}, {"CSS",
   11430}, {"HTML", 3786584}, {"Java", 639285}, {"M4", 
  4315}, {"Makefile", 83869}, {"MATLAB", 1096}, {"Objective-C", 
  26289}, {"Perl", 25314}, {"Python", 6457462}, {"QMake", 2353}, {"R",
   28787}, {"Ruby", 30036}, {"SCSS", 11062}, {"Shell", 
  126919}, {"TeX", 109175}, {"XSLT", 13518}} *)
  
ListLogPlot[{#} & /@ (Transpose@{Range@Length@langs, res[[All, 2]]}), 
 Filling -> Axis, Joined -> False, PlotRange -> All, 
 PlotLegends -> langs, FillingStyle -> {Thickness[0.005]},
 Frame -> {True, True, False, False}, 
 FrameLabel -> {"Programming Language", "Source code lines"}, 
 AxesOrigin -> {0, 0}]  
{% endraw %}
{% endhighlight %}


<p align="center">
<img style="width: 100%; height: 100%" src="{{ site.url }}/images/languages.png" alt="Github analytics programming languages">
</p>

<p align="center">
<img style="width: 100%; height: 100%" src="{{ site.url }}/images/days_since_commit.png" alt="Github analytics commits">
</p>


## GraphQL

{% highlight mathematica %}
{% raw %}
gql[accessToken_] :=
 URLRead[
  HTTPRequest[
   "https://api.github.com/graphql",
   <|
    "Method" -> "POST",
    "Body" -> "{
     		\"query\": \"query { viewer { login } }\"
     	}",
    "Headers" -> {"Authorization" -> "Bearer " <> accessToken}|>
   ]
  ]

res = gql[gqlAccessToken];
res["Body"]
(* {"data":{"viewer":{"login":"ekamperi"}}} *)
{% endraw %}
{% endhighlight %}
