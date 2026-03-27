**课题中文名称**  
基于Agent技术的多模态时间序列预测研究  

**课题英文名称**  
Research on Multi-modal Time Series Prediction Based on Agent Technology  

**课题简介（3000字以内）**  
随着人工智能技术的快速发展，大模型在多模态理解和时序数据分析方面展现出强大潜力。本课题旨在探索基于Agent技术的大模型在多模态时间序列预测中的应用。通过结合文本、图像、传感器数据等多种模态信息，构建具有自主学习和推理能力的智能Agent，实现对复杂时序数据的精准预测。  

研究内容主要包括：  

1. 多模态数据融合：针对时间序列数据与文本、图像等多模态信息的异质性，设计有效的特征提取与融合方法，构建统一的多模态表示。  
2. Agent架构设计：基于大模型构建具有推理、决策和自适应学习能力的Agent，使其能够根据历史数据与实时信息动态调整预测策略。  
3. 预测模型优化：结合深度学习和强化学习方法，提升Agent在长时序、多模态场景下的预测精度与鲁棒性。  
4. 实验验证与应用：在金融、交通、气象等典型时序预测场景中进行实验，验证模型的有效性与实用性。  

本课题的创新点在于将大模型的多模态理解能力与Agent的自主决策机制相结合，为解决复杂时序预测问题提供新的思路与方法。预期成果包括一套可扩展的多模态时序预测框架，以及在公开数据集上的性能验证报告。


**主要研究学科和方向（500字以内）**  
**所属学科**：人工智能、数据科学、计算机应用技术  
**研究方向**：  

1. 多模态时间序列分析：研究文本、图像、时序数据等多源信息的融合与表示学习方法。  
2. 大模型与Agent技术：探索基于大模型的智能Agent构建方法，增强其在时序任务中的推理与决策能力。  
3. 预测建模与优化：结合深度学习、强化学习等方法，开发高效、鲁棒的多模态时序预测模型。  
4. 交叉应用研究：将所提方法应用于金融、交通、气象等实际场景，验证其跨领域适用性。

---

## 工程实现说明

仓库已经补上了一个可执行的澳洲五州日前电力预测工程骨架，核心代码在 `src/aupower/`：

- `prepare-data`：统一负荷到 `30 分钟` 粒度，清洗天气并生成次日天气伪预测。
- `extract-events`：从新闻语料中过滤电力相关文本，生成结构化事件记录和州级日特征。
- `train`：训练 `Lag` 基线和三个专家模型：`BaseExpert`、`WeatherExpert`、`EventExpert`。
- `backtest`：评估固定融合、规则路由和 `Ollama` 路由。
- `predict`：对指定州和日期输出 `48` 个半小时预测点与风险标签。

默认配置在 `configs/default.yaml`，Conda 依赖定义在 `environment.yml`。

典型运行顺序：

```bash
conda env create -f environment.yml
conda activate aupower-agent
PYTHONPATH=src python -m aupower prepare-data --config configs/default.yaml
PYTHONPATH=src python -m aupower extract-events --config configs/default.yaml --limit 500
PYTHONPATH=src python -m aupower train --config configs/default.yaml
PYTHONPATH=src python -m aupower backtest --config configs/default.yaml --use-ollama
```
