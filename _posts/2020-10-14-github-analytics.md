---
layout: post
title:  "GitHub analytics with Mathematica"
date:   2020-10-14
categories: [programming]
tags: ['GitHub', 'GraphQL', 'git', 'JSON', 'Mathematica', 'Programming', 'REST API']
---

### Contents
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## Introduction
Why *Mathematica* and not *Python*? Well, for starters, there is a ton of examples in *Python*, so adding one more to the pile wouldn't make any difference. Plus, although I do program in *Python*, I don't enjoy it as much as I enjoy *Mathematica*. Also, *Jupyter* notebooks are nowhere near as polished as *Mathematica*'s.

### REST API
REST API stands for "Representational State Transfer Application Programming Interface". In simple terms, it's a set of agreed rules on how to retrieve data when you connect to a specific URL. To make a REST API call, you need to know the following ingredients of such a request:

1. The **endpoint**, which is basically the URL you request for. For example, GitHub's endpoint is *https://api.github.com*.
    + The **path** that determines the specific resource you are asking for. For example, in the URL *https://api.github.com/user/repos*, the path is */user/repos*, which captures our intention to have the user's repositories returned. When you read in a doc an expression like */repos/:owner/:repo/*, the *owner* and *repo* are variables. You need to replace them with the actual value of that variable. E.g., write */repos/ekamperi/rteval*, if you are interested in the repository named *rteval* of the user *ekamperi*.
    + **Query parameters**. Sometimes a request is accompanied by a list of parameters that modify the request. These always begin with a question mark "?" and each *parameter=value* pair is delimited by an ampersand "&". E.g., in */repos/ekamperi/rteval/commits&per_page=100&sha=master*, we inform the server that we want 100 commits to be returned, and we want the listing of commits to start from the *HEAD* of the *master* branch.
2. The **method** defines the kind of request that we are submitting to the server. It may be one of *GET*, *POST*, *PUT*, *PATCH*, *DELETE*. They allow the following operations: *Create*, *Read*, *Update*, and *Delete* (the so-called CRUD). In short, *GET* performs the READ operation (we ask the server to send us back some data). *POST* performs the CREATE operation (we ask the server to create a new resource in it). *PUT* and *PATCH* perform an UPDATE operation, and *DELETE*, well, you know what *DELETE* does.
3. The **headers** are used to exchange metadata between client and server. For example, they are used to perform authentication by injecting some authorization token into the HTTP header.
4. The **data** or **body** hold the client's information to the server, and it is used with *POST*, *PUT*, *PATCH*, and *DELETE* methods.

### Authentication
To experiment with GitHub's REST API, we need to authenticate to the service. User-to-server requests are rate-limited at 5.000 requests per hour and per authenticated user. However, for unauthenticated requests, only up to 60 requests per hour per originating IP are allowed. So, for any serious experimentation, authentication is a must. The best way to proceed is to create a [personal access token (PAT)](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/creating-a-personal-access-token), as an alternative to using passwords for authentication to GitHub when using the GitHub API or the command line. Here is how you could authenticate via *curl*:

<p align="center">
<img style="width: 100%; height: 100%" src="{{ site.url }}/images/github_cmd.png" alt="GitHub authenticate via curl">
</p>

### A simple example of a REST API call
{% highlight mathematica %}
{% raw %}
ClearAll["Global`*"];
resp = URLRead@HTTPRequest["https://api.github.com/users/ekamperi"]
{% endraw %}
{% endhighlight %}

*Mathematica* will respond with something like:

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
rj = ImportString[resp["Body"], "RawJSON"];
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

## More involved examples
### How to get the weekly commit count
We will issue a *GET /repos/:owner/:repo:/stats/participation* request, that returns the total commit
counts for the owner and total commit counts in all (all is everyone combined, including the owner in the last 52 weeks). 
The array order is oldest week (index 0) to most recent week.

{% highlight mathematica %}
{% raw %}
getWeeklyCommitCount[owner_, repo_, accessToken_] :=
 URLRead[HTTPRequest[
   "https://api.github.com/repos/" <> owner <> "/" <> repo <> "/stats/participation",
   <|"Headers" -> {"Authorization" -> "token " <> accessToken}|>]]

resp = getWeeklyCommitCount["ekamperi", "rteval", accessToken]
rj = ImportString[resp["Body"], "RawJSON"]

Grid[{
ListPlot[First@#, FrameLabel -> {"Week #", Last@#}, Frame -> {True, True, False, False},
   FrameTicks -> {Range[1, 52, 3], Automatic}, Joined -> True, InterpolationOrder -> 1,
   GridLines -> Automatic, Filling -> {1 -> {2}}, ImageSize -> Medium,
   PlotRange -> All, PlotLegends -> Placed[Style[#, 11] & /@ {"All", "Owner"}, Below] 
   ] & /@ {
  {{rj[[1]], rj[[2]]}, "# of commits"},
  {Accumulate /@ {rj[[1]], rj[[2]]}, "Cumulative # of commits"}}
}]
{% endraw %}
{% endhighlight %}

<p align="center">
<img style="width: 100%; height: 100%" src="{{ site.url }}/images/weekly_commits.png" alt="GitHub weekly commits">
</p>

### How to get the list of repositories

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
<img style="width: 25%; height: 25%" src="{{ site.url }}/images/list_of_repos.png" alt="GitHub analytics commits">
</p>

### How to get the size of all repositories broken down by language

We start by creating a function that talks to the */repos/:owner/:repo/languages* path. Same as before,
we pass our personal access token to the header of the request:

{% highlight mathematica %}
{% raw %}
getLanguages[owner_, repo_, accessToken_] :=
 URLRead[HTTPRequest[
   "https://api.github.com/repos/" <> owner <> "/" <> repo <>  "/languages",
   <|"Headers" -> {"Authorization" -> "token " <> accessToken}|>]]
{% endraw %}
{% endhighlight %}

Let's test what data the server returns:

{% highlight mathematica %}
{% raw %}
lang = getLanguages["ekamperi", "rteval", accessToken]
lang["Body"]
(* "Python":6440911, "R":28787, "CSS":1800, "MATLAB":1096} *)
{% endraw %}
{% endhighlight %}

So, the repository named *rteval* of the user *ekamperi* contains 6440911 bytes of Python, 28787 bytes of R, 1800 bytes of CSS and 1096 bytes of MATLAB code.
Let's collect the data for all languages:

{% highlight mathematica %}
{% raw %}
rv = getLanguages["ekamperi", #, accessToken] & /@ repoNames;
allLangs = ImportString[#["Body"], "RawJSON"] & /@ rv
(* {<|"Ruby" -> 9797, "CSS" -> 6146, 
  "HTML" -> 2536|>, <||>, <|"Shell" -> 1729|>, <|"C++" -> 165097, 
  "QMake" -> 2353, "Shell" -> 923|>, <||>, <|"Java" -> 291551, 
  "Shell" -> 512|>, <|"Java" -> 347734|>, <|"HTML" -> 268483, 
  "SCSS" -> 11094, "Shell" -> 1772|>, <|"C" -> 140177, 
  "Makefile" -> 2994, "Shell" -> 1139|>, <||>, <|"TeX" -> 99290, 
  "Shell" -> 60209, "C" -> 27165, "Perl" -> 20520, "Ruby" -> 3912, 
  "ANTLR" -> 139, "Makefile" -> 116, 
  "Awk" -> 88|>, <|"HTML" -> 3515565, "C" -> 489306, 
  "Objective-C" -> 26289, "Makefile" -> 25493, "XSLT" -> 13518, 
  "M4" -> 4315, "CSS" -> 3484, "Shell" -> 3050, 
  "C++" -> 2240|>, <|"C" -> 696634, "Shell" -> 57585, 
  "Makefile" -> 55266, "Python" -> 16551, "Ruby" -> 16327, 
  "Perl" -> 4794|>, <|"Python" -> 6440911, "R" -> 28787, 
  "CSS" -> 1800, "MATLAB" -> 1096|>, <|"TeX" -> 9885|>} *)
{% endraw %}
{% endhighlight %}

Now we'd like to calculate the aggregate data:

{% highlight mathematica %}
{% raw %}
langs = Union@Flatten[Keys /@ allLangs]
assoc = <|# -> {} & /@ langs|>
(* <|"ANTLR" -> {}, "Awk" -> {}, "C" -> {}, "C++" -> {}, "CSS" -> {}, 
 "HTML" -> {}, "Java" -> {}, "M4" -> {}, "Makefile" -> {}, 
 "MATLAB" -> {}, "Objective-C" -> {}, "Perl" -> {}, "Python" -> {}, 
 "QMake" -> {}, "R" -> {}, "Ruby" -> {}, "SCSS" -> {}, "Shell" -> {}, 
 "TeX" -> {}, "XSLT" -> {}|> *)
 
f[key_, val_] := AppendTo[assoc[key], val]
g[lang_, values_] := {lang, Total@values}
res = KeyValueMap[g, assoc]
(* {{"ANTLR", 139}, {"Awk", 88}, {"C", 1353282}, {"C++", 167337}, {"CSS",
   11430}, {"HTML", 3786584}, {"Java", 639285}, {"M4", 
  4315}, {"Makefile", 83869}, {"MATLAB", 1096}, {"Objective-C", 
  26289}, {"Perl", 25314}, {"Python", 6457462}, {"QMake", 2353}, {"R",
   28787}, {"Ruby", 30036}, {"SCSS", 11062}, {"Shell", 
  126919}, {"TeX", 109175}, {"XSLT", 13518}} *)
{% endraw %}
{% endhighlight %}

And then plot the results:

{% highlight mathematica %}
{% raw %}
ListLogPlot[{#} & /@ (Transpose@{Range@Length@langs, res[[All, 2]]}), 
 Filling -> Axis, Joined -> False, PlotRange -> All, 
 PlotLegends -> langs, FillingStyle -> {Thickness[0.005]},
 Frame -> {True, True, False, False}, 
 FrameLabel -> {"Programming Language", "Source code lines"}, 
 AxesOrigin -> {0, 0}]  
{% endraw %}
{% endhighlight %}


<p align="center">
<img style="width: 100%; height: 100%" src="{{ site.url }}/images/languages.png" alt="GitHub analytics programming languages">
</p>

### How to get the dates of the commits in a repository

First, we create a function that, given an SHA sum, it returns a list of (commit, date) tuples.

{% highlight mathematica %}
{% raw %}
getNextCommitsWithDate[owner_, repo_, sha_, accessToken_] :=
 Module[{bodyJ, commits = {}, resp},
  resp = URLRead[
    HTTPRequest[
     "https://api.github.com/repos/" <> owner <> "/" <> repo <> "/commits",
     <|"Query" -> {"per_page" -> 100, "sha" -> First@Last@sha},
      "Headers" -> {"Authorization" -> "token " <> accessToken}|>]];
  bodyJ = ImportString[resp["Body"], "RawJSON"];
  commits = 
   Append[commits, {#["sha"], #["commit"]["author"]["date"]} & /@ 
     bodyJ];
  Return[commits[[1]]]
  ]
{% endraw %}
{% endhighlight %}

We then apply the function above *repeatedly* (via `FixedPointList`) and accumulate the results:

{% highlight mathematica %}
{% raw %}
getAllCommitsWithDate[owner_, repo_, accessToken_] :=
 Union@
    Flatten[#, 1] &@
  FixedPointList[
   getNextCommitsWithDate[owner, repo, #, accessToken] &,
   {{"master", ""}}]
{% endraw %}
{% endhighlight %}

We sort the commits by their date:

{% highlight mathematica %}
{% raw %}
sortedCommits =
  DateString[#, {"Year", "-", "Month", "-", "Day", "T", "Time", "Z"}] & /@
   Sort[
    AbsoluteTime[
       {#, {"Year", "-", "Month", "-", "Day", "T", "Time", "Z"}}] & /@
     acs[[All, 2]]
    ];
{% endraw %}
{% endhighlight %}

Take their difference and plot the results:

{% highlight mathematica %}
{% raw %}
dds = DateDifference[First@#, Last@#] & /@ Partition[sortedDates, 2, 1];

Grid[{
  #[{MovingAverage[dds, 7], MovingAverage[dds, 14]}, Joined -> True, 
     InterpolationOrder -> 0, PlotRange -> All, Filling -> {1 -> Bottom, 2 -> None}, 
     Frame -> {True, True, False, False}, PlotStyle -> {Automatic, Red}, 
     FrameLabel -> {"Commit #", "Day passed since\nprevious commit"}, 
     PlotLegends -> Placed[Style[#, 9] & /@ {"Weekly moving average", 
         "Biweekly moving average"}, Below],
     ImageSize -> Medium] & /@ {ListPlot, ListLogPlot}
  }]
{% endraw %}
{% endhighlight %}

<p align="center">
<img style="width: 100%; height: 100%" src="{{ site.url }}/images/days_since_commit.png" alt="GitHub analytics commits">
</p>

## GraphQL

GraphQL is a data query and a manipulation language for APIs. Initially developed by Facebook for internal use was then released to the public. GraphQL provides an approach to developing web APIs similar to REST, yet it is different from REST. Its difference lies in that it allows clients to describe the structure of the data required. Other features include a type system, a query language, and type introspection. In GraphQL there is ony one endpoint, here https://api.github.com/graphql. The user submits a JSON formatted query describing what data exactly wants the server to return. For instance, in order to get the currently authenticated user, we need to issue a JSON query of the form `"query": "query { viewer { login } }"`. Note however that we must escape the **"**.

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

<p align="center">
<img style="width: 75%; height: 75%" src="{{ site.url }}/images/graphiql.png" alt="GraphiQL screenshot">
</p>

