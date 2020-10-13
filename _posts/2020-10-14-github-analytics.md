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
<img style="width: 100%; height: 100%" src="{{ site.url }}/images/http_resposne.png" alt="HTTPResponse Mathematica">
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

