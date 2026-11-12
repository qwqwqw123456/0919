---
name: text-stats
description: 统计一段文本的字符数、中文字数、词数、行数与高频词
tags: [文本, 统计]
version: 0.1
runtime: python
entry: main.py
risk: L0
timeout: 20
input_schema:
  type: object
  properties:
    text:
      type: string
      description: 要统计的文本
    top_k:
      type: integer
      description: 返回高频词的数量
      default: 10
  required: [text]
---

# text-stats

把文本作为 `text` 传入，返回字符数、中文字数、词数、行数和出现频次最高的词。

适合快速核对字数、发现重复用词、生成词频摘要。纯本地计算，不联网。
