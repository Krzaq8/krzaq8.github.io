---
layout: post
title:  "DrEureka on Go1 - results"
date:   2025-04-08 14:59:19 +0200
categories: quadruped-rl
---


# Experiment

First we run Eureka reward function generation. It's an iterative process that uses LLM to generate python code, which describes reward function.

Once we have sensible policy trained, we can set bounds for domain randomization, by running policy in slightly disturbed environment.

Finally we run additional training on environment with randomized physics parameters. This stage is similar to the Eureka stage, we use LLMs to take it into account and produce new reward functions.

## Config

### Eureka
```
num_envs: 4096
train_iterations: 1000
model: gpt-4-0125-preview
sample: 16
iterations: 4
```

### DrEureka
```
num_envs: 1024
train_iterations: 1000
play_iterations: 100
model: gpt-4-0125-preview
sample: 16
iterations: 1
```

## Versions
Docker image: ` nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 `

Rest of the environment is described [here](https://github.com/nomagiclab/quadruped-rl/blob/master/DrEureka_changes/startup.sh)

GPU: 
- RTX 4080S (16GB VRAM, 50TFLOPS) - Eureka stage,
- RTX A4000 (16GB VRAM, 20TFLOPS) - DrEureka stage


## Time consumed
Due to high VRAM usage of algorithm we decided to run Eureka algorithm sequentially so it can fit on GPU.
In case of DrEureka I've managed to enable 2 parallel runs of training.

Eureka stage: 35h

RAPP stage: <1h

DrEureka: 5h

## Results
[Eureka](/pages/eureka)

[DrEureka](/pages/dr_eureka)


<!-- You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-title.MARKUP`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `MARKUP` is the file extension representing the format used in the file. After that, include the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/ -->
