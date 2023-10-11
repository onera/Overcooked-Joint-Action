# Overcooked-AI 🧑‍🍳🤖


## Introduction 🥘

Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/).

The goal of the game is to deliver soups as fast as possible. Each soup requires placing up to 3 ingredients in a pot, waiting for the soup to cook, and then having an agent pick up the soup and delivering it. The agents should split up tasks on the fly and coordinate effectively in order to achieve high reward.

You can **try out the game [here](https://humancompatibleai.github.io/overcooked-demo/)** (playing with some previously trained DRL agents). To play with your own trained agents using this interface, you can use [this repo](https://github.com/HumanCompatibleAI/overcooked-demo). To run human-AI experiments, check out [this repo](https://github.com/HumanCompatibleAI/overcooked-hAI-exp). You can find some human-human and human-AI gameplay data already collected [here](https://github.com/HumanCompatibleAI/overcooked_ai/tree/master/src/human_aware_rl/static/human_data).

DRL implementations compatible with the environment are included in the repo as a submodule under src/human_aware_rl. 

The old [human_aware_rl](https://github.com/HumanCompatibleAI/human_aware_rl) is being deprecated and should only used to reproduce the results in the 2019 paper: *[On the Utility of Learning about Humans for Human-AI Coordination](https://arxiv.org/abs/1910.05789)* (also see our [blog post](https://bair.berkeley.edu/blog/2019/10/21/coordination/)).

For simple usage of the environment, it's worthwhile considering using [this environment wrapper](https://github.com/Stanford-ILIAD/PantheonRL).