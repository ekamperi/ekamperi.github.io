---
layout: post
title:  "Github analytics with Mathematica"
date:   2020-10-14
categories: [programming]
tags: ['Github', 'GraphQL', 'JSON', 'Mathematica', 'Programming', 'REST API']
---

## Introduction

### Authorization
In order to experiment with Github's REST API, we need to authenticate to the service. User-to-server requests are rate-limited at 5.000 requests per hour and per authenticated user. For unauthenticated requests, only up to 60 requests per hour per originating IP are allowed. The best way to proceed is to create a [personal access token (PAT)](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/creating-a-personal-access-token), as an alternative to using passwords for authentication to GitHub when using the GitHub API or the command line.

{% highlight mathematica %}
{% raw %}
ClearAll["Global`*"];
resp = URLRead@HTTPRequest["https://api.github.com/users/ekamperi"]
{% endraw %}
{% endhighlight %}

Mathematica will respond with something like:

<p align="center">
<img style="width: 50%; height: 50%" src="{{ site.url }}/images/http_response.png" alt="HTTPResponse Mathematica">
</p>

We can request the properties of the response object returned by `URLRead[]`:
{% highlight mathematica %}
{% raw %}
resp["Properties"]
(* {"Body", "BodyByteArray", "BodyBytes", "CharacterEncoding", \
"ContentType", "Headers", "StatusCode", "StatusCodeDescription", \
"Version"} *)
{% endraw %}
{% endhighlight %}

And then print the value of some property:

{% highlight mathematica %}
{% raw %}
resp[{"StatusCode", "StatusCodeDescription"}]
(* <|"StatusCode" -> 200, "StatusCodeDescription" -> "OK"|> *)
{% endraw %}
{% endhighlight %}

We extract the data from HTTP Message Body (the data bytes transmitted immediately after the HTTP headers), import it as a JSON string and list the associated keys:

{% highlight mathematica %}
{% raw %}
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

## How to get the list of repositories

In order to get the list of repositories, we send a request to the https://api.github.com/user/repos endpoint.
However, we need to pass our personal access token to the list of headers that will be sent to the server.
The string that we will send must be of the form "Authorization token <access token>":

{% highlight mathematica %}
{% raw %}
getRepos[accessToken_] :=
 URLRead@HTTPRequest["https://api.github.com/user/repos",
   <|"Headers" -> {"Authorization" -> "token " <> accessToken}|>]
{% endraw %}
{% endhighlight %}

We send a request to the url, read back the response, interpret the body message as JSON and then display the results:
{% highlight mathematica %}
{% raw %}
resp = getRepos[accessToken];
rj = ImportString[resp["Body"], "RawJSON"];
repoNames = rj[[All, "name"]];
Table[{i, rj[[i]]["name"]}, {i, 1, Length@rj}] // Dataset
{% endraw %}
{% endhighlight %}

<p align="center">
<img style="width: 25%; height: 25%" src="{{ site.url }}/images/list_of_repos.png" alt="Github analytics commits">
</p>

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

GraphQL is a data query and a manipulation language for APIs. Initially, it was developed by Facebook for internal use, and then release to public. GraphQL provides an approach to developing web APIs similar to REST, yet it is different from REST. Its difference is that it allows clients to describe the structure of the data required. Other features include a type system, a query language and type introspection.

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
