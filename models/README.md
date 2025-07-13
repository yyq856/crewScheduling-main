# 模型文件管理 (Model Files Management)

本目录包含机组排班优化系统的AI模型文件。

## 📁 当前模型文件

### 生产模型 (Production Models)
- `best_model.pth` - 当前最佳模型，用于生产环境
- `best_model_0606.pth` - 历史版本模型 (2024年6月6日)

## 🔄 模型版本管理

### 命名规范
```
模型类型_版本号_日期.pth
例如:
- best_model.pth           # 当前最佳模型
- best_model_v2.1.pth      # 版本号模型
- best_model_20250101.pth  # 日期版本模型
- experimental_v1.0.pth    # 实验性模型
```

### 模型分类

#### 🏆 生产模型 (Production)
- **用途**: 正式环境使用的稳定模型
- **命名**: `best_model*.pth`
- **要求**: 经过充分测试和验证

#### 🧪 实验模型 (Experimental)
- **用途**: 研究和开发阶段的模型
- **命名**: `experimental_*.pth`
- **要求**: 包含详细的实验记录

#### 📚 基准模型 (Baseline)
- **用途**: 性能对比的基准模型
- **命名**: `baseline_*.pth`
- **要求**: 稳定的参考标准

## 📊 模型信息记录

每个模型文件应包含以下信息（在对应的.json文件中）：

```json
{
  "model_name": "best_model.pth",
  "version": "2.1.0",
  "creation_date": "2025-01-01",
  "training_data": "dataset_v2.0",
  "performance_metrics": {
    "accuracy": 0.95,
    "coverage_rate": 0.98,
    "optimization_time": "120s"
  },
  "model_architecture": {
    "layers": 4,
    "hidden_size": 256,
    "attention_heads": 8
  },
  "training_config": {
    "epochs": 100,
    "learning_rate": 0.001,
    "batch_size": 32
  },
  "description": "优化后的注意力机制模型，提升了航班覆盖率",
  "author": "开发团队",
  "notes": "在大规模数据集上训练，适用于复杂排班场景"
}
```

## 🚀 使用指南

### 加载模型
```python
import torch
from attention.model import AttentionModel

# 加载最佳模型
model = AttentionModel()
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()
```

### 模型评估
```python
# 评估模型性能
from attention.utils import evaluate_model

metrics = evaluate_model(model, test_data)
print(f"模型准确率: {metrics['accuracy']:.3f}")
print(f"覆盖率: {metrics['coverage_rate']:.3f}")
```

## 📋 维护清单

### 定期任务
- [ ] 每月备份重要模型文件
- [ ] 清理过期的实验模型
- [ ] 更新模型性能记录
- [ ] 验证模型兼容性

### 版本发布
- [ ] 性能测试通过
- [ ] 文档更新完整
- [ ] 代码审查完成
- [ ] 备份旧版本模型

## ⚠️ 注意事项

1. **文件大小**: 模型文件可能较大，注意Git LFS的使用
2. **版本控制**: 重要模型变更需要详细的提交说明
3. **安全性**: 不要提交包含敏感信息的模型
4. **兼容性**: 确保模型与当前代码版本兼容

## 🔗 相关文档

- [训练指南](../attention/README.md)
- [API文档](../API.md)
- [配置说明](../unified_config.py)

---

*最后更新: 2025年1月*