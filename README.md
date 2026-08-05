# 无线算法大赛：Physical AI 信道生成

本项目根据采样点位置、环境点云地图和已知 MIMO-OFDM 信道建立 Physical AI 无线数字孪生模型，并生成测试点信道。方案不使用传统射线追踪。

## 最终结果（v50）

- 输出文件：`Round1_Test_Channel.npy`
- 形状：`(500, 256, 4, 192)`
- 类型：`complex64`
- 文件大小：`786432128` 字节
- SHA-256：`648ED578F70AD026C0B72DFD01C798DC2C0E0E13E9793CCDEFC80A47994874C8`
- 五折 PAS：`0.7648172498`
- 五折 PDP：`0.8519760847`
- 五折 NMSE：`0.6143360417`（越低越好）
- 五折综合分数：`0.7706072767`
- 校验：无 NaN/Inf、无全零样本、形状与类型正确，8 项核心单元测试全部通过

输出 `.npy` 超过 GitHub 普通文件限制，仓库使用 Git LFS 管理。克隆后需要先安装 Git LFS，再执行 `git lfs pull`。

## 方法概览

1. 按“极化 → H → V”顺序解析 256 维基站阵列。
2. 从点云地图提取位置附近障碍物、走廊、材料与基站方向上下文。
3. 将信道压缩为与评分函数直接对齐的 PAS/PDP 物理谱特征。
4. 使用分组局部克里金、条件专家集成和空间图传播预测测试点物理谱。
5. 通过径向载波相位、阵列导向补偿、分组复数克里金和局部相位同步生成复信道初值。
6. 进行角域/时延域交替投影与分组能量校正。
7. 使用严格折外复数标量、局部物理邻域和 UE 级残差进行 NMSE 校准。
8. 使用两种空间几何的四动作样本门控，在“不修正、仅 PAS、仅 PDP、PAS+PDP”之间逐样本选择。

训练集中检测到 16 个全零信道并作为异常值剔除：

`22, 205, 221, 466, 695, 976, 1071, 1113, 1369, 1390, 1414, 1501, 1728, 1815, 1872, 1889`

## 主要文件

- `predict.py`：v50 最终推理流程
- `train_model.py`：地图条件神经核训练
- `prepare_map.py` / `prepare_advanced_map.py`：点云地图上下文预处理
- `prepare_features.py`：PAS/PDP 评分对齐特征提取
- `verify_submission.py`：输出文件严格校验
- `metrics.py`：官方指标参考实现
- `physical_ai/`：数据、点云编码、特征、神经模型、邻域、空间插值及谱投影模块
- `artifacts/`：最终模型、特征、地图上下文和校准器
- `tests/`：核心单元测试
- `CODE_EXPLANATION.md`：详细代码与算法说明

## 运行方式

```powershell
python -m pip install -r requirements.txt
python predict.py --device cuda --batch-size 8
python verify_submission.py --file Round1_Test_Channel.npy
python -m unittest discover -s tests -v
```

`predict.py` 默认加载 v50 最终 artifacts。原始赛题数据仍需按官方目录结构放置；如需从原始数据重建地图与谱特征，可运行 `prepare_advanced_map.py` 和 `prepare_features.py`。
