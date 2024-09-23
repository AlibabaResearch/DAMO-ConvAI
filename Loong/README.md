<img src="assets/logo.png" alt="Loong" width="120" align="left"><div align="center"><h1>&nbsp; Loong: Benchmarking Long-Context LLMs with Extended Multi-Doc QA</h1></div>

<p align="center" style="font-size:200%">
    <img alt="GitHub" src="https://img.shields.io/github/license/MozerWang/Loong.svg?color=blue&style=flat-square">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/MozerWang/Loong">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/MozerWang/Loong">
</p>

<p align="center"><font size=6>📃</font> <a target="_self" href="https://arxiv.org/abs/2406.17419"> <img style="height:14pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a> <font size=6>•</font> <font size=6>🔔</font> <a target="_self" href="https://github.com/MozerWang/Loong"> <img style="height:14pt" src="https://img.shields.io/badge/-Code-pink?style=flat&logo=github"></a></p>

## 👀Overview
This repository contains code for our paper [Leave No Document Behind: Benchmarking Long-Context LLMs with Extended Multi-Doc QA](https://arxiv.org/abs/2406.17419). We propose a novel long-context benchmark, 🐉 **Loong**, aligning with realistic scenarios through extended multi-document question answering (QA). Loong typically consists of 11 documents per test instance on average, spanning three real-world scenarios in English and Chinese: (1) *Financial Reports*, (2) *Legal Cases*, and (3) *Academic Papers*. Meanwhile, Loong introduces new evaluation tasks from the perspectives of *Spotlight Locating*, *Comparison*, *Clustering*, and *Chain of Reasoning*, to facilitate a more realistic and comprehensive evaluation of long-context understanding. Furthermore, Loong features inputs of varying lengths (e.g., *10K-50K*, *50K-100K*, *100K-200K*, *beyond 200K*) and evaluation tasks of diverse difficulty, enabling fine-grained assessment of LLMs across different context lengths and task complexities.
> *Please find more details of this work in our paper.*

![Overview of Loong](assets/main_fig.jpg)
> Showcase of four evaluation tasks in Loong (\<di>...\</di> marks the content of the i-th document). (a) *Spotlight Locating*: Locate the evidence. (b) *Comparison*: Locate and compare the evidence. (c) *Clustering*: Locate and cluster the evidence into groups. (d) *Chain of Reasoning*: Locate and reasoning along a logical chain.

## 📰News
`[2024-09-20]` 📰Our paper has been accepted to the EMNLP Main Conference.

`[2024-07-30]` 🤖The performance of phi-3, llama-3.1-8B, gpt-4o-mini on Loong are updated.

`[2024-07-03]` 🔥The code and benchmark are releasing. If you encounter any issues, please feel free to contact us.

`[2024-06-25]` 👨‍💻The code is currently being refined, and we plan to release the evaluation code and benchmark within the next one or two weeks. If you encounter any issues, please feel free to contact me at wangminzheng2023@ia.ac.cn.

## 🏆Leaderboard
<table>
  <thead>
    <tr>
      <th>Models</th>
      <th>Claimed Length</th>
      <th colspan="2" style="text-align: center;">Spotlight Locating</th>
      <th colspan="2" style="text-align: center;">Comparison</th>
      <th colspan="2" style="text-align: center;">Clustering</th>
      <th colspan="2" style="text-align: center;">Chain of Reason</th>
      <th colspan="2" style="text-align: center;">Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://ai.google.dev/gemini-api/docs/models/gemini#:~:text=Gemini-,Gemini%201.5%20Pro%20(Preview%20only),-Text%20and%20images">Gemini-1.5-pro</a></td>
      <td style="text-align: center;">1000K</td>
      <td style="text-align: center;">75.02</td><td style="text-align: center;">0.56</td>
      <td style="text-align: center;">49.94</td><td style="text-align: center;">0.27</td>
      <td style="text-align: center;">44.10</td><td style="text-align: center;">0.09</td>
      <td style="text-align: center;">64.97</td><td style="text-align: center;">0.37</td>
      <td style="text-align: center;">55.37</td><td style="text-align: center;">0.27</td>
    </tr>
    <tr style="background-color:#f0f0f0;">
      <td><a href="https://platform.openai.com/docs/models/gpt-4o">GPT-4o</a></td>
      <td style="text-align: center;">128K</td>
      <td style="text-align: center;">73.95</td><td style="text-align: center;">0.62</td>
      <td style="text-align: center;">50.50</td><td style="text-align: center;">0.28</td>
      <td style="text-align: center;">44.29</td><td style="text-align: center;">0.09</td>
      <td style="text-align: center;">57.95</td><td style="text-align: center;">0.28</td>
      <td style="text-align: center;">53.47</td><td style="text-align: center;">0.26</td>
    </tr>
    <tr>
      <td><a href="https://docs.anthropic.com/en/docs/intro-to-claude#claude-3-5-family">Claude3.5-Sonnet</a></td>
      <td style="text-align: center;">200K</td>
      <td style="text-align: center;">58.45</td><td style="text-align: center;">0.49</td>
      <td style="text-align: center;">54.21</td><td style="text-align: center;">0.35</td>
      <td style="text-align: center;">45.77</td><td style="text-align: center;">0.07</td>
      <td style="text-align: center;">43.92</td><td style="text-align: center;">0.25</td>
      <td style="text-align: center;">48.85</td><td style="text-align: center;">0.23</td>
    </tr>
    <tr style="background-color:#f0f0f0;">
      <td><a href="https://docs.anthropic.com/en/docs/intro-to-claude#claude-3-family">Claude3-Haiku</a></td>
      <td style="text-align: center;">200K</td>
      <td style="text-align: center;">68.68</td><td style="text-align: center;">0.59</td>
      <td style="text-align: center;">42.10</td><td style="text-align: center;">0.21</td>
      <td style="text-align: center;">35.04</td><td style="text-align: center;">0.02</td>
      <td style="text-align: center;">47.59</td><td style="text-align: center;">0.17</td>
      <td style="text-align: center;">44.88</td><td style="text-align: center;">0.19</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2-72B-Instruct">Qwen2-72B-Instruct</a></td>
      <td style="text-align: center;">128K</td>
      <td style="text-align: center;">54.17</td><td style="text-align: center;">0.36</td>
      <td style="text-align: center;">42.38</td><td style="text-align: center;">0.20</td>
      <td style="text-align: center;">36.71</td><td style="text-align: center;">0.04</td>
      <td style="text-align: center;">47.76</td><td style="text-align: center;">0.18</td>
      <td style="text-align: center;">43.29</td><td style="text-align: center;">0.15</td>
    </tr>
    <tr style="background-color:#f0f0f0;">
      <td><a href="https://platform.openai.com/docs/models/gpt-4o-mini">GPT-4o-mini</a></td>
      <td style="text-align: center;">128K</td>
      <td style="text-align: center;">53.12</td><td style="text-align: center;">0.41</td>
      <td style="text-align: center;">44.27</td><td style="text-align: center;">0.20</td>
      <td style="text-align: center;">32.58</td><td style="text-align: center;">0.04</td>
      <td style="text-align: center;">52.34</td><td style="text-align: center;">0.23</td>
      <td style="text-align: center;">42.95</td><td style="text-align: center;">0.18</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/glm-4-9b-chat-1m">GLM4-9B-Chat</a></td>
      <td style="text-align: center;">1000K</td>
      <td style="text-align: center;">57.35</td><td style="text-align: center;">0.47</td>
      <td style="text-align: center;">40.38</td><td style="text-align: center;">0.20</td>
      <td style="text-align: center;">28.52</td><td style="text-align: center;">0.02</td>
      <td style="text-align: center;">39.94</td><td style="text-align: center;">0.16</td>
      <td style="text-align: center;">38.31</td><td style="text-align: center;">0.16</td>
    </tr>
    <tr style="background-color:#f0f0f0;">
      <td><a href="https://kimi.moonshot.cn/">Kimi-Chat</a></td>
      <td style="text-align: center;">200K</td>
      <td style="text-align: center;">60.98</td><td style="text-align: center;">0.50</td>
      <td style="text-align: center;">34.74</td><td style="text-align: center;">0.13</td>
      <td style="text-align: center;">28.76</td><td style="text-align: center;">0.04</td>
      <td style="text-align: center;">38.52</td><td style="text-align: center;">0.15</td>
      <td style="text-align: center;">37.49</td><td style="text-align: center;">0.16</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">Llama-3.1-8B-Instruct</a></td>
      <td style="text-align: center;">128K</td>
      <td style="text-align: center;">59.96</td><td style="text-align: center;">0.46</td>
      <td style="text-align: center;">35.73</td><td style="text-align: center;">0.18</td>
      <td style="text-align: center;">27.83</td><td style="text-align: center;">0.01</td>
      <td style="text-align: center;">35.59</td><td style="text-align: center;">0.14</td>
      <td style="text-align: center;">36.31</td><td style="text-align: center;">0.14</td>
    </tr>
    <tr style="background-color:#f0f0f0;">
      <td><a href="https://huggingface.co/microsoft/Phi-3-small-128k-instruct">Phi-3-small</a></td>
      <td style="text-align: center;">128K</td>
      <td style="text-align: center;">29.23</td><td style="text-align: center;">0.10</td>
      <td style="text-align: center;">20.12</td><td style="text-align: center;">0.06</td>
      <td style="text-align: center;">17.53</td><td style="text-align: center;">0.00</td>
      <td style="text-align: center;">14.36</td><td style="text-align: center;">0.01</td>
      <td style="text-align: center;">19.03</td><td style="text-align: center;">0.03</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">Phi-3-mini</a></td>
      <td style="text-align: center;">128K</td>
      <td style="text-align: center;">25.65</td><td style="text-align: center;">0.15</td>
      <td style="text-align: center;">13.34</td><td style="text-align: center;">0.04</td>
      <td style="text-align: center;">12.00</td><td style="text-align: center;">0.00</td>
      <td style="text-align: center;">12.61</td><td style="text-align: center;">0.01</td>
      <td style="text-align: center;">14.54</td><td style="text-align: center;">0.04</td>
    </tr>
  </tbody>
</table>

> Overall results on four evaluation tasks. For each task, the indicator on the left represents the **_Avg Scores`(0～100)`_**, while the right one represents the **_Perfect Rate`(0~1)`_**.

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th style="text-align: center;">Claimed Length</th>
            <th colspan="2" style="text-align: center;">Spotlight Locating</th>
            <th colspan="2" style="text-align: center;">Comparison</th>
            <th colspan="2" style="text-align: center;">Clustering</th>
            <th colspan="2" style="text-align: center;">Chain of Reasoning</th>
            <th colspan="2" style="text-align: center;">Overall</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan="12" style="text-align: center;"><b>Set1 (10K-50K)</b></td>
        </tr>
        <tr>
            <td><a href="https://platform.openai.com/docs/models/gpt-4o">GPT-4o</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">85.67</td><td style="text-align: center;">0.81</td>
            <td style="text-align: center;">64.27</td><td style="text-align: center;">0.33</td>
            <td style="text-align: center;">57.01</td><td style="text-align: center;">0.24</td>
            <td style="text-align: center;">81.58</td><td style="text-align: center;">0.55</td>
            <td style="text-align: center;">70.40</td><td style="text-align: center;">0.44</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://docs.anthropic.com/en/docs/intro-to-claude#claude-3-5-family">Claude3.5-Sonnet</a></td>
            <td style="text-align: center;">200K</td>
            <td style="text-align: center;">60.85</td><td style="text-align: center;">0.55</td>
            <td style="text-align: center;">69.07</td><td style="text-align: center;">0.47</td>
            <td style="text-align: center;">58.63</td><td style="text-align: center;">0.13</td>
            <td style="text-align: center;">68.57</td><td style="text-align: center;">0.50</td>
            <td style="text-align: center;">63.69</td><td style="text-align: center;">0.37</td>
        </tr>
        <tr>
            <td><a href="https://ai.google.dev/gemini-api/docs/models/gemini#:~:text=Gemini-,Gemini%201.5%20Pro%20(Preview%20only),-Text%20and%20images">Gemini-1.5-pro</a></td>
            <td style="text-align: center;">1000K</td>
            <td style="text-align: center;">75.00</td><td style="text-align: center;">0.60</td>
            <td style="text-align: center;">54.88</td><td style="text-align: center;">0.28</td>
            <td style="text-align: center;">56.15</td><td style="text-align: center;">0.23</td>
            <td style="text-align: center;">70.64</td><td style="text-align: center;">0.37</td>
            <td style="text-align: center;">63.36</td><td style="text-align: center;">0.34</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://platform.openai.com/docs/models/gpt-4o-mini">GPT-4o-mini</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">62.49</td><td style="text-align: center;">0.56</td>
            <td style="text-align: center;">65.48</td><td style="text-align: center;">0.40</td>
            <td style="text-align: center;">45.81</td><td style="text-align: center;">0.12</td>
            <td style="text-align: center;">79.85</td><td style="text-align: center;">0.55</td>
            <td style="text-align: center;">62.42</td><td style="text-align: center;">0.36</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/Qwen/Qwen2-72B-Instruct">Qwen2-72B-Instruct</a></td>
            <td style="text-align: center;">200K</td>
            <td style="text-align: center;">68.49</td><td style="text-align: center;">0.55</td>
            <td style="text-align: center;">60.60</td><td style="text-align: center;">0.37</td>
            <td style="text-align: center;">47.08</td><td style="text-align: center;">0.08</td>
            <td style="text-align: center;">70.39</td><td style="text-align: center;">0.36</td>
            <td style="text-align: center;">60.11</td><td style="text-align: center;">0.29</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://docs.anthropic.com/en/docs/intro-to-claude#claude-3-family">Claude3-Haiku</a></td>
            <td style="text-align: center;">200K</td>
            <td style="text-align: center;">60.94</td><td style="text-align: center;">0.55</td>
            <td style="text-align: center;">59.97</td><td style="text-align: center;">0.40</td>
            <td style="text-align: center;">45.53</td><td style="text-align: center;">0.04</td>
            <td style="text-align: center;">66.85</td><td style="text-align: center;">0.34</td>
            <td style="text-align: center;">57.14</td><td style="text-align: center;">0.28</td>
        </tr>
        <tr>
            <td><a href="https://kimi.moonshot.cn/">Kimi-Chat</a></td>
            <td style="text-align: center;">200K</td>
            <td style="text-align: center;">81.11</td><td style="text-align: center;">0.74</td>
            <td style="text-align: center;">46.70</td><td style="text-align: center;">0.20</td>
            <td style="text-align: center;">47.84</td><td style="text-align: center;">0.07</td>
            <td style="text-align: center;">53.77</td><td style="text-align: center;">0.17</td>
            <td style="text-align: center;">55.02</td><td style="text-align: center;">0.24</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://huggingface.co/THUDM/glm-4-9b-chat-1m">GLM4-9B-Chat</a></td>
            <td style="text-align: center;">1000K</td>
            <td style="text-align: center;">63.11</td><td style="text-align: center;">0.53</td>
            <td style="text-align: center;">54.10</td><td style="text-align: center;">0.27</td>
            <td style="text-align: center;">39.50</td><td style="text-align: center;">0.08</td>
            <td style="text-align: center;">56.32</td><td style="text-align: center;">0.28</td>
            <td style="text-align: center;">51.43</td><td style="text-align: center;">0.25</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">Llama-3.1-8B-Instruct</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">67.91</td><td style="text-align: center;">0.57</td>
            <td style="text-align: center;">41.62</td><td style="text-align: center;">0.20</td>
            <td style="text-align: center;">36.55</td><td style="text-align: center;">0.04</td>
            <td style="text-align: center;">54.74</td><td style="text-align: center;">0.34</td>
            <td style="text-align: center;">48.10</td><td style="text-align: center;">0.24</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">Phi-3-mini</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">46.13</td><td style="text-align: center;">0.30</td>
            <td style="text-align: center;">22.18</td><td style="text-align: center;">0.05</td>
            <td style="text-align: center;">19.30</td><td style="text-align: center;">0.02</td>
            <td style="text-align: center;">20.44</td><td style="text-align: center;">0.03</td>
            <td style="text-align: center;">24.58</td><td style="text-align: center;">0.07</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/microsoft/Phi-3-small-128k-instruct">Phi-3-small</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">35.00</td><td style="text-align: center;">0.15</td>
            <td style="text-align: center;">26.83</td><td style="text-align: center;">0.12</td>
            <td style="text-align: center;">17.01</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">15.87</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">21.44</td><td style="text-align: center;">0.05</td>
        </tr>
        <tr>
            <td colspan="12" style="text-align: center;"><b>Set2 (50K-100K)</b></td>
        </tr>
        <tr>
            <td><a href="https://platform.openai.com/docs/models/gpt-4o">GPT-4o</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">86.76</td><td style="text-align: center;">0.72</td>
            <td style="text-align: center;">59.81</td><td style="text-align: center;">0.40</td>
            <td style="text-align: center;">47.83</td><td style="text-align: center;">0.11</td>
            <td style="text-align: center;">62.09</td><td style="text-align: center;">0.34</td>
            <td style="text-align: center;">58.38</td><td style="text-align: center;">0.29</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://ai.google.dev/gemini-api/docs/models/gemini#:~:text=Gemini-,Gemini%201.5%20Pro%20(Preview%20only),-Text%20and%20images">Gemini-1.5-pro</a></td>
            <td style="text-align: center;">1000K</td>
            <td style="text-align: center;">76.50</td><td style="text-align: center;">0.57</td>
            <td style="text-align: center;">54.51</td><td style="text-align: center;">0.34</td>
            <td style="text-align: center;">44.58</td><td style="text-align: center;">0.09</td>
            <td style="text-align: center;">64.87</td><td style="text-align: center;">0.34</td>
            <td style="text-align: center;">55.56</td><td style="text-align: center;">0.26</td>
        </tr>
        <tr>
            <td><a href="https://docs.anthropic.com/en/docs/intro-to-claude#claude-3-5-family">Claude3.5-Sonnet</a></td>
            <td style="text-align: center;">200K</td>
            <td style="text-align: center;">63.83</td><td style="text-align: center;">0.53</td>
            <td style="text-align: center;">58.90</td><td style="text-align: center;">0.39</td>
            <td style="text-align: center;">50.96</td><td style="text-align: center;">0.10</td>
            <td style="text-align: center;">46.09</td><td style="text-align: center;">0.26</td>
            <td style="text-align: center;">52.73</td><td style="text-align: center;">0.24</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://platform.openai.com/docs/models/gpt-4o-mini">GPT-4o-mini</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">63.54</td><td style="text-align: center;">0.46</td>
            <td style="text-align: center;">51.48</td><td style="text-align: center;">0.26</td>
            <td style="text-align: center;">36.56</td><td style="text-align: center;">0.04</td>
            <td style="text-align: center;">56.51</td><td style="text-align: center;">0.25</td>
            <td style="text-align: center;">47.74</td><td style="text-align: center;">0.19</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/Qwen/Qwen2-72B-Instruct">Qwen2-72B-Instruct</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">64.53</td><td style="text-align: center;">0.43</td>
            <td style="text-align: center;">42.60</td><td style="text-align: center;">0.21</td>
            <td style="text-align: center;">38.52</td><td style="text-align: center;">0.05</td>
            <td style="text-align: center;">51.18</td><td style="text-align: center;">0.20</td>
            <td style="text-align: center;">45.71</td><td style="text-align: center;">0.17</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://docs.anthropic.com/en/docs/intro-to-claude#claude-3-family">Claude3-Haiku</a></td>
            <td style="text-align: center;">200K</td>
            <td style="text-align: center;">73.71</td><td style="text-align: center;">0.66</td>
            <td style="text-align: center;">41.90</td><td style="text-align: center;">0.22</td>
            <td style="text-align: center;">36.18</td><td style="text-align: center;">0.02</td>
            <td style="text-align: center;">50.20</td><td style="text-align: center;">0.15</td>
            <td style="text-align: center;">45.45</td><td style="text-align: center;">0.17</td>
        </tr>
        <tr>
            <td><a href="https://kimi.moonshot.cn/">Kimi-Chat</a></td>
            <td style="text-align: center;">200K</td>
            <td style="text-align: center;">72.82</td><td style="text-align: center;">0.52</td>
            <td style="text-align: center;">46.77</td><td style="text-align: center;">0.21</td>
            <td style="text-align: center;">33.46</td><td style="text-align: center;">0.06</td>
            <td style="text-align: center;">40.51</td><td style="text-align: center;">0.15</td>
            <td style="text-align: center;">42.40</td><td style="text-align: center;">0.16</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">Llama-3.1-8B-Instruct</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">72.79</td><td style="text-align: center;">0.59</td>
            <td style="text-align: center;">44.51</td><td style="text-align: center;">0.27</td>
            <td style="text-align: center;">32.98</td><td style="text-align: center;">0.01</td>
            <td style="text-align: center;">40.53</td><td style="text-align: center;">0.15</td>
            <td style="text-align: center;">41.98</td><td style="text-align: center;">0.16</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/THUDM/glm-4-9b-chat-1m">GLM4-9B-Chat</a></td>
            <td style="text-align: center;">1000K</td>
            <td style="text-align: center;">65.04</td><td style="text-align: center;">0.54</td>
            <td style="text-align: center;">41.80</td><td style="text-align: center;">0.23</td>
            <td style="text-align: center;">30.72</td><td style="text-align: center;">0.02</td>
            <td style="text-align: center;">42.34</td><td style="text-align: center;">0.17</td>
            <td style="text-align: center;">40.19</td><td style="text-align: center;">0.17</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://huggingface.co/microsoft/Phi-3-small-128k-instruct">Phi-3-small</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">34.17</td><td style="text-align: center;">0.16</td>
            <td style="text-align: center;">22.08</td><td style="text-align: center;">0.08</td>
            <td style="text-align: center;">20.51</td><td style="text-align: center;">0.01</td>
            <td style="text-align: center;">16.20</td><td style="text-align: center;">0.01</td>
            <td style="text-align: center;">21.40</td><td style="text-align: center;">0.04</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">Phi-3-mini</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">44.71</td><td style="text-align: center;">0.29</td>
            <td style="text-align: center;">22.81</td><td style="text-align: center;">0.09</td>
            <td style="text-align: center;">16.37</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">15.39</td><td style="text-align: center;">0.01</td>
            <td style="text-align: center;">20.84</td><td style="text-align: center;">0.05</td>
        </tr>
        <tr>
            <td colspan="12" style="text-align: center;"><b>Set3 (100K-200K)</b></td>
        </tr>
        <tr>
            <td><a href="https://ai.google.dev/gemini-api/docs/models/gemini#:~:text=Gemini-,Gemini%201.5%20Pro%20(Preview%20only),-Text%20and%20images">Gemini-1.5-pro</a></td>
            <td style="text-align: center;">1000K</td>
            <td style="text-align: center;">81.25</td><td style="text-align: center;">0.56</td>
            <td style="text-align: center;">44.66</td><td style="text-align: center;">0.20</td>
            <td style="text-align: center;">39.90</td><td style="text-align: center;">0.05</td>
            <td style="text-align: center;">58.38</td><td style="text-align: center;">0.36</td>
            <td style="text-align: center;">52.05</td><td style="text-align: center;">0.24</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://platform.openai.com/docs/models/gpt-4o">GPT-4o</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">74.84</td><td style="text-align: center;">0.65</td>
            <td style="text-align: center;">42.40</td><td style="text-align: center;">0.21</td>
            <td style="text-align: center;">38.70</td><td style="text-align: center;">0.04</td>
            <td style="text-align: center;">45.06</td><td style="text-align: center;">0.09</td>
            <td style="text-align: center;">46.95</td><td style="text-align: center;">0.19</td>
        </tr>
        <tr>
            <td><a href="https://docs.anthropic.com/en/docs/intro-to-claude#claude-3-5-family">Claude3.5-Sonnet</a></td>
            <td style="text-align: center;">200K</td>
            <td style="text-align: center;">65.36</td><td style="text-align: center;">0.56</td>
            <td style="text-align: center;">50.32</td><td style="text-align: center;">0.34</td>
            <td style="text-align: center;">37.79</td><td style="text-align: center;">0.03</td>
            <td style="text-align: center;">25.95</td><td style="text-align: center;">0.11</td>
            <td style="text-align: center;">42.06</td><td style="text-align: center;">0.19</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://docs.anthropic.com/en/docs/intro-to-claude#claude-3-family">Claude3-Haiku</a></td>
            <td style="text-align: center;">200K</td>
            <td style="text-align: center;">77.81</td><td style="text-align: center;">0.67</td>
            <td style="text-align: center;">37.07</td><td style="text-align: center;">0.17</td>
            <td style="text-align: center;">30.94</td><td style="text-align: center;">0.01</td>
            <td style="text-align: center;">36.87</td><td style="text-align: center;">0.12</td>
            <td style="text-align: center;">41.41</td><td style="text-align: center;">0.18</td>
        </tr>
        <tr>
            <td><a href="https://platform.openai.com/docs/models/gpt-4o-mini">GPT-4o-mini</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">58.27</td><td style="text-align: center;">0.49</td>
            <td style="text-align: center;">33.46</td><td style="text-align: center;">0.09</td>
            <td style="text-align: center;">27.33</td><td style="text-align: center;">0.01</td>
            <td style="text-align: center;">35.67</td><td style="text-align: center;">0.04</td>
            <td style="text-align: center;">35.63</td><td style="text-align: center;">0.11</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://huggingface.co/Qwen/Qwen2-72B-Instruct">Qwen2-72B-Instruct</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">46.99</td><td style="text-align: center;">0.27</td>
            <td style="text-align: center;">37.06</td><td style="text-align: center;">0.13</td>
            <td style="text-align: center;">31.50</td><td style="text-align: center;">0.02</td>
            <td style="text-align: center;">35.01</td><td style="text-align: center;">0.07</td>
            <td style="text-align: center;">35.94</td><td style="text-align: center;">0.09</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/THUDM/glm-4-9b-chat-1m">GLM4-9B-Chat</a></td>
            <td style="text-align: center;">1000K</td>
            <td style="text-align: center;">69.19</td><td style="text-align: center;">0.56</td>
            <td style="text-align: center;">37.99</td><td style="text-align: center;">0.18</td>
            <td style="text-align: center;">26.63</td><td style="text-align: center;">0.01</td>
            <td style="text-align: center;">32.30</td><td style="text-align: center;">0.09</td>
            <td style="text-align: center;">37.36</td><td style="text-align: center;">0.16</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://kimi.moonshot.cn/">Kimi-Chat</a></td>
            <td style="text-align: center;">200K</td>
            <td style="text-align: center;">62.13</td><td style="text-align: center;">0.54</td>
            <td style="text-align: center;">24.20</td><td style="text-align: center;">0.05</td>
            <td style="text-align: center;">21.98</td><td style="text-align: center;">0.01</td>
            <td style="text-align: center;">31.02</td><td style="text-align: center;">0.14</td>
            <td style="text-align: center;">31.37</td><td style="text-align: center;">0.14</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">Llama-3.1-8B-Instruct</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">60.05</td><td style="text-align: center;">0.46</td>
            <td style="text-align: center;">25.86</td><td style="text-align: center;">0.11</td>
            <td style="text-align: center;">21.96</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">19.14</td><td style="text-align: center;">0.02</td>
            <td style="text-align: center;">28.41</td><td style="text-align: center;">0.10</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://huggingface.co/microsoft/Phi-3-small-128k-instruct">Phi-3-small</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">25.12</td><td style="text-align: center;">0.06</td>
            <td style="text-align: center;">15.26</td><td style="text-align: center;">0.01</td>
            <td style="text-align: center;">16.80</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">12.75</td><td style="text-align: center;">0.01</td>
            <td style="text-align: center;">16.94</td><td style="text-align: center;">0.01</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">Phi-3-mini</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">7.40</td><td style="text-align: center;">0.03</td>
            <td style="text-align: center;">1.97</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">6.07</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">7.38</td><td style="text-align: center;">0.01</td>
            <td style="text-align: center;">5.79</td><td style="text-align: center;">0.01</td>
        </tr>
        <tr>
            <td colspan="12" style="text-align: center;"><b>Set4 (200K-250K)</b></td>
        </tr>
        <tr>
            <td><a href="https://ai.google.dev/gemini-api/docs/models/gemini#:~:text=Gemini-,Gemini%201.5%20Pro%20(Preview%20only),-Text%20and%20images">Gemini-1.5-pro</a></td>
            <td style="text-align: center;">1000K</td>
            <td style="text-align: center;">62.23</td><td style="text-align: center;">0.49</td>
            <td style="text-align: center;">43.08</td><td style="text-align: center;">0.20</td>
            <td style="text-align: center;">36.48</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">68.51</td><td style="text-align: center;">0.49</td>
            <td style="text-align: center;">50.70</td><td style="text-align: center;">0.25</td>
        </tr>
        <tr  style="background-color:#f0f0f0;">
            <td><a href="https://docs.anthropic.com/en/docs/intro-to-claude#claude-3-family">Claude3-Haiku</a></td>
            <td style="text-align: center;">200K</td>
            <td style="text-align: center;">53.26</td><td style="text-align: center;">0.40</td>
            <td style="text-align: center;">27.00</td><td style="text-align: center;">0.03</td>
            <td style="text-align: center;">25.36</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">28.11</td><td style="text-align: center;">0.05</td>
            <td style="text-align: center;">32.15</td><td style="text-align: center;">0.10</td>
        </tr>
        <tr>
            <td><a href="https://platform.openai.com/docs/models/gpt-4o">GPT-4o</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">36.79</td><td style="text-align: center;">0.19</td>
            <td style="text-align: center;">23.97</td><td style="text-align: center;">0.08</td>
            <td style="text-align: center;">30.40</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">32.89</td><td style="text-align: center;">0.07</td>
            <td style="text-align: center;">31.11</td><td style="text-align: center;">0.07</td>
        </tr>
        <tr  style="background-color:#f0f0f0;">
            <td><a href="https://docs.anthropic.com/en/docs/intro-to-claude#claude-3-5-family">Claude3.5-Sonnet</a></td>
            <td style="text-align: center;">200K</td>
            <td style="text-align: center;">36.91</td><td style="text-align: center;">0.24</td>
            <td style="text-align: center;">28.82</td><td style="text-align: center;">0.05</td>
            <td style="text-align: center;">28.68</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">28.77</td><td style="text-align: center;">0.08</td>
            <td style="text-align: center;">30.51</td><td style="text-align: center;">0.08</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/Qwen/Qwen2-72B-Instruct">Qwen2-72B-Instruct</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">33.18</td><td style="text-align: center;">0.16</td>
            <td style="text-align: center;">26.59</td><td style="text-align: center;">0.08</td>
            <td style="text-align: center;">29.84</td><td style="text-align: center;">0.01</td>
            <td style="text-align: center;">25.81</td><td style="text-align: center;">0.04</td>
            <td style="text-align: center;">28.92</td><td style="text-align: center;">0.06</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">Llama-3.1-8B-Instruct</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">31.72</td><td style="text-align: center;">0.13</td>
            <td style="text-align: center;">27.27</td><td style="text-align: center;">0.10</td>
            <td style="text-align: center;">15.17</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">22.89</td><td style="text-align: center;">0.02</td>
            <td style="text-align: center;">22.51</td><td style="text-align: center;">0.05</td>
        </tr>
        <tr>
            <td><a href="https://platform.openai.com/docs/models/gpt-4o-mini">GPT-4o-mini</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">20.66</td><td style="text-align: center;">0.09</td>
            <td style="text-align: center;">19.18</td><td style="text-align: center;">0.03</td>
            <td style="text-align: center;">16.03</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">27.81</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">20.41</td><td style="text-align: center;">0.02</td>
        </tr>
        <tr  style="background-color:#f0f0f0;">
            <td><a href="https://huggingface.co/THUDM/glm-4-9b-chat-1m">GLM4-9B-Chat</a></td>
            <td style="text-align: center;">1000K</td>
            <td style="text-align: center;">15.67</td><td style="text-align: center;">0.12</td>
            <td style="text-align: center;">21.33</td><td style="text-align: center;">0.05</td>
            <td style="text-align: center;">12.35</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">21.04</td><td style="text-align: center;">0.05</td>
            <td style="text-align: center;">16.84</td><td style="text-align: center;">0.05</td>
        </tr>
        <tr>
            <td><a href="https://kimi.moonshot.cn/">Kimi-Chat</a></td>
            <td style="text-align: center;">200K</td>
            <td style="text-align: center;">20.17</td><td style="text-align: center;">0.12</td>
            <td style="text-align: center;">9.17</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">5.65</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">22.61</td><td style="text-align: center;">0.11</td>
            <td style="text-align: center;">13.50</td><td style="text-align: center;">0.05</td>
        </tr>
        <tr style="background-color:#f0f0f0;">
            <td><a href="https://huggingface.co/microsoft/Phi-3-small-128k-instruct">Phi-3-small</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">22.36</td><td style="text-align: center;">0.02</td>
            <td style="text-align: center;">16.43</td><td style="text-align: center;">0.05</td>
            <td style="text-align: center;">11.50</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">10.35</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">14.27</td><td style="text-align: center;">0.01</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">Phi-3-mini</a></td>
            <td style="text-align: center;">128K</td>
            <td style="text-align: center;">5.21</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">2.20</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">3.45</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">2.58</td><td style="text-align: center;">0.00</td>
            <td style="text-align: center;">3.38</td><td style="text-align: center;">0.00</td>
        </tr>
    </tbody>
</table>

> The performance of LLMs on four evaluation tasks with different length sets. For each task, the indicator on the left represents the **_Avg Scores`(0～100)`_**, while the right one represents the **_Perfect Rate`(0~1)`_**.

- Following previous work, we prompt GPT-4 as a judge to evaluate the model's output based on the golden answer and the question's requirements from three aspects: *Accuracy*, *Hallucinations*, and *Completeness*, scoring from 0 to 100. For a detailed prompt, please refer to our paper. 
- We design two indicators: (1) **_Avg Scores_**: the average value of scores given by GPT-4 for all questions; (2) **_Perfect Rate_**: the proportion of cases scoring 100 out of the total cases. The latter is a more stringent evaluation metric compared to the former.
- We set `temperature = 0` to eliminate randomness and keep other hyper-parameters default. For API-Based LLMs, we directly utilize the official API for testing. Since the Kimi-Chat-200k currently does not provide an interface, we manually input content on the web. As for open-source models, we conduct experiments on a server with 8 $\times$ A100 80GB.

## 🔧Evaluate long-context LLMs
**Step1** Download Loong benchmark
```shell
git clone https://github.com/MozerWang/Loong.git
cd Loong
wget -P data/ http://alibaba-research.oss-cn-beijing.aliyuncs.com/loong/doc.zip
unzip data/doc.zip -d data/
```

**Step2** Create a conda environment and Install other dependencies.
```shell
conda create --name loong python=3.9 -y
conda activate loong
pip install -r requirements.txt
```

**Step3** Preparing the Model

1. (**Must**) Set up your OPENAI key in config/models/gpt4.yaml
```shell
api_key: "Your OPENAI key"
```
2. If you are using API-based LLM
```shell
# Firstly, Set up your key in config/models/*.yaml
api_key: "Your API key"
```
3. If you are using Open-sourced LLM
```shell
# We recommend using vLLM. And we use HTTP server that implements OpenAI’s Completions and Chat API.
# We have provided using example for Qwen2 and GLM4. See details in Loong/src/vllm_eample.sh
cd src
sh vllm_example.sh
```

**Step4** Evaluate
```shell
cd src
sh run.sh
```

**Things To Know**
- We provide a complete evaluation process:  
`step1_load_data.py` # Data loading  
`step2_model_generate.py` # Model generation  
`step3_model_evaluate.py` # GPT-4 evaluation    
`step4_cal_metric.py` # Result statistics  

- For `step2_model_generate.py`, you can design the model generation part yourself, modifying it to use your own model's inference method. Just make sure the input and output interfaces in `src/utils/generate.py` remain consistent:
```shell
# Input
generate(prompts, config, output_path, process_num, tag)

# Output
result = prompt.copy() # for prompt in prompts
result[tag] = response_content # Your LLM's response
with open(output_path, 'a', encoding='utf-8') as fw:
    fw.write(json.dumps(result, ensure_ascii=False) + '\n')
```

- In `data/loong.jsonl`, the `level` key means task:  
`level1` # Spotlight Locating    
`level2` # Comparison    
`level3` # Clustering    
`level4` # Chain of Reasoning   

## Citation
```
@article{wang2024loong,
  title={Leave No Document Behind: Benchmarking Long-Context LLMs with Extended Multi-Doc QA},
  author={Minzheng Wang, Longze Chen, Cheng Fu, Shengyi Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan Xu, Lei Zhang, Run Luo, Yunshui Li, Min Yang, Fei Huang, Yongbin Li},
  year={2024}
  journal={arXiv preprint arXiv:2406.17419},
}
```
