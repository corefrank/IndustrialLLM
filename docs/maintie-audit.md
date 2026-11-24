# IndustrialLLM 第一轮 MaintIE 数据审计

基于 Work 环境下载并读取的官方 JSON。索引从 0 开始。项目切分、训练与测试尚未执行。

- [MaintIE 固定 commit](https://github.com/nlp-tlp/maintie/tree/825e7710e4129598da997276a38b3cb40b200bc2)
- [论文](https://aclanthology.org/2024.lrec-main.954/)

## 文件与统计

### Gold

- 文件 `data/gold_release.json`，1,076 条，817,880 字节。
- SHA-256 `ef811cf4ffd0c90e6a13b1369e24e9d1d08f9a294d873ab4fb1a580096d7d6e0`。
- 3,397 个实体实例，155 个实际发行标签，2,341 条关系。
- 空文本或空 token 列表 0 条，零实体 0 条，零关系 0 条。
- 按原文本精确去重 1,075 条，按 NFKC、casefold、空白折叠去重 1,074 条。
- 文本是含占位符和缩写的英文维修短文本。

| 长度口径 | 最小 | 中位数 | P95 | 最大 | 均值 |
|---|---:|---:|---:|---:|---:|
| 字符 | 3 | 34 | 52 | 74 | 34.37 |
| 发行数据 token | 2 | 6 | 9 | 12 | 5.69 |

| 原始关系标签 | 实例数 |
|---|---:|
| hasParticipant/hasPatient | 1,206 |
| hasPart | 533 |
| isA | 364 |
| hasParticipant/hasAgent | 166 |
| contains | 38 |
| hasProperty | 34 |

<details>
<summary>Gold 全部原始实体标签与次数</summary>

| 标签 | 实例数 | MVP 字段 |
|---|---:|---|
| Activity | 1 | `activities` |
| Activity/MaintenanceActivity/Adjust | 10 | `activities` |
| Activity/MaintenanceActivity/Calibrate | 10 | `activities` |
| Activity/MaintenanceActivity/Diagnose | 32 | `activities` |
| Activity/MaintenanceActivity/Inspect | 76 | `activities` |
| Activity/MaintenanceActivity/Repair | 128 | `activities` |
| Activity/MaintenanceActivity/Replace | 347 | `activities` |
| Activity/MaintenanceActivity/Service | 48 | `activities` |
| Activity/SupportingActivity/Admin | 10 | `activities` |
| Activity/SupportingActivity/Assemble | 31 | `activities` |
| Activity/SupportingActivity/Isolate | 13 | `activities` |
| Activity/SupportingActivity/Measure | 9 | `activities` |
| Activity/SupportingActivity/Modify | 18 | `activities` |
| Activity/SupportingActivity/Move | 10 | `activities` |
| Activity/SupportingActivity/Operate | 17 | `activities` |
| Activity/SupportingActivity/Perform | 20 | `activities` |
| Activity/SupportingActivity/Teamwork | 4 | `activities` |
| PhysicalObject | 14 | `physical_objects` |
| PhysicalObject/ControllingObject | 3 | `physical_objects` |
| PhysicalObject/ControllingObject/ElectricControllingObject | 4 | `physical_objects` |
| PhysicalObject/ControllingObject/ElectricEarthingObject | 5 | `physical_objects` |
| PhysicalObject/ControllingObject/ElectricSeparatingObject | 2 | `physical_objects` |
| PhysicalObject/ControllingObject/MechanicalMovementControllingObject | 40 | `physical_objects` |
| PhysicalObject/ControllingObject/OpenFlowControllingObject | 5 | `physical_objects` |
| PhysicalObject/ControllingObject/SealedFluidSwitchingObject | 10 | `physical_objects` |
| PhysicalObject/ControllingObject/SealedFluidVaryingObject | 14 | `physical_objects` |
| PhysicalObject/ControllingObject/SolidSubstanceFlowVaryingObject | 1 | `physical_objects` |
| PhysicalObject/ControllingObject/SpaceAccessObject | 33 | `physical_objects` |
| PhysicalObject/CoveringObject/ClosureObject | 17 | `physical_objects` |
| PhysicalObject/CoveringObject/FinishingObject | 9 | `physical_objects` |
| PhysicalObject/CoveringObject/InfillingObject | 27 | `physical_objects` |
| PhysicalObject/DrivingObject/CombustionEngine | 55 | `physical_objects` |
| PhysicalObject/DrivingObject/ElectromagneticLinearDrivingObject | 4 | `physical_objects` |
| PhysicalObject/DrivingObject/ElectromagneticRotationalDrivingObject | 62 | `physical_objects` |
| PhysicalObject/DrivingObject/FluidPoweredDrivingObject | 119 | `physical_objects` |
| PhysicalObject/DrivingObject/MagneticForceDrivingObject | 1 | `physical_objects` |
| PhysicalObject/DrivingObject/MechanicalEnergyDrivingObject | 5 | `physical_objects` |
| PhysicalObject/EmittingObject | 2 | `physical_objects` |
| PhysicalObject/EmittingObject/AcousticWaveEmittingObject | 6 | `physical_objects` |
| PhysicalObject/EmittingObject/CombustionHeatingObject | 2 | `physical_objects` |
| PhysicalObject/EmittingObject/ElectricCoolingObject | 49 | `physical_objects` |
| PhysicalObject/EmittingObject/ElectricHeatingObject | 2 | `physical_objects` |
| PhysicalObject/EmittingObject/LightObject | 35 | `physical_objects` |
| PhysicalObject/EmittingObject/ThermalCoolingObject | 28 | `physical_objects` |
| PhysicalObject/EmittingObject/ThermalEnergyTransferObject | 4 | `physical_objects` |
| PhysicalObject/EmittingObject/ThermalHeatingObject | 1 | `physical_objects` |
| PhysicalObject/EmittingObject/WirelessPowerObject | 3 | `physical_objects` |
| PhysicalObject/GeneratingObject/ChemicalToElectricalEnergyGeneratingObject | 6 | `physical_objects` |
| PhysicalObject/GeneratingObject/ContinuousTransferObject | 9 | `physical_objects` |
| PhysicalObject/GeneratingObject/DiscontinuousTransferObject | 18 | `physical_objects` |
| PhysicalObject/GeneratingObject/GaseousFlowGeneratingObject | 53 | `physical_objects` |
| PhysicalObject/GeneratingObject/LiquidFlowGeneratingObject | 72 | `physical_objects` |
| PhysicalObject/GeneratingObject/MechanicalToElectricalEnergyGeneratingObject | 13 | `physical_objects` |
| PhysicalObject/GeneratingObject/SignalGeneratingObject | 1 | `physical_objects` |
| PhysicalObject/GuidingObject/ClosedEnclosureGuidingObject | 120 | `physical_objects` |
| PhysicalObject/GuidingObject/ElectricEnergyGuidingObject | 20 | `physical_objects` |
| PhysicalObject/GuidingObject/ElectricSignalGuidingObject | 4 | `physical_objects` |
| PhysicalObject/GuidingObject/HighVoltageElectricEnergyGuidingObject | 1 | `physical_objects` |
| PhysicalObject/GuidingObject/LightGuidingObject | 3 | `physical_objects` |
| PhysicalObject/GuidingObject/MechanicalEnergyGuidingObject | 115 | `physical_objects` |
| PhysicalObject/GuidingObject/OpenEnclosureGuidingObject | 4 | `physical_objects` |
| PhysicalObject/GuidingObject/RailObject | 1 | `physical_objects` |
| PhysicalObject/GuidingObject/SolidMatterGuidingObject | 2 | `physical_objects` |
| PhysicalObject/GuidingObject/ThermalEnergyGuidingObject | 5 | `physical_objects` |
| PhysicalObject/HoldingObject | 1 | `physical_objects` |
| PhysicalObject/HoldingObject/CarryingObject | 17 | `physical_objects` |
| PhysicalObject/HoldingObject/EnclosingObject | 31 | `physical_objects` |
| PhysicalObject/HoldingObject/FasteningObject | 34 | `physical_objects` |
| PhysicalObject/HoldingObject/FramingObject | 5 | `physical_objects` |
| PhysicalObject/HoldingObject/JointingObject | 38 | `physical_objects` |
| PhysicalObject/HoldingObject/PositioningObject | 11 | `physical_objects` |
| PhysicalObject/HoldingObject/ReinforcingObject | 6 | `physical_objects` |
| PhysicalObject/HoldingObject/StructuralSupportingObject | 70 | `physical_objects` |
| PhysicalObject/HumanInteractionObject/FingerInteractionObject | 1 | `physical_objects` |
| PhysicalObject/HumanInteractionObject/FootInteractionObject | 1 | `physical_objects` |
| PhysicalObject/HumanInteractionObject/HandInteractionObject | 9 | `physical_objects` |
| PhysicalObject/HumanInteractionObject/MovementInteractionObject | 16 | `physical_objects` |
| PhysicalObject/InformationProcessingObject/ElectricSignalProcessingObject | 5 | `physical_objects` |
| PhysicalObject/InformationProcessingObject/ElectricSignalRelayingObject | 1 | `physical_objects` |
| PhysicalObject/InformationProcessingObject/FluidSignallingObject | 5 | `physical_objects` |
| PhysicalObject/InterfacingObject/CollectingInterfacingObject | 3 | `physical_objects` |
| PhysicalObject/InterfacingObject/DetachableCoupling | 15 | `physical_objects` |
| PhysicalObject/InterfacingObject/HighVoltageConnectingObject | 2 | `physical_objects` |
| PhysicalObject/InterfacingObject/LevelConnectingObject | 4 | `physical_objects` |
| PhysicalObject/InterfacingObject/LowVoltageConnectingObject | 4 | `physical_objects` |
| PhysicalObject/InterfacingObject/NonDetachableCoupling | 1 | `physical_objects` |
| PhysicalObject/InterfacingObject/PotentialConnectingObject | 2 | `physical_objects` |
| PhysicalObject/InterfacingObject/SealedFlowConnectingObject | 2 | `physical_objects` |
| PhysicalObject/InterfacingObject/SpaceLinkingObject | 3 | `physical_objects` |
| PhysicalObject/MatterProcessingObject | 1 | `physical_objects` |
| PhysicalObject/MatterProcessingObject/AssemblingObject | 2 | `physical_objects` |
| PhysicalObject/MatterProcessingObject/ChemicalSeparatingObject | 2 | `physical_objects` |
| PhysicalObject/MatterProcessingObject/ForceSeparatingObject | 19 | `physical_objects` |
| PhysicalObject/MatterProcessingObject/GrindingAndCrushingObject | 31 | `physical_objects` |
| PhysicalObject/MatterProcessingObject/MechanicalSeparatingObject | 43 | `physical_objects` |
| PhysicalObject/MatterProcessingObject/MixingObject | 2 | `physical_objects` |
| PhysicalObject/MatterProcessingObject/SurfaceTreatmentObject | 2 | `physical_objects` |
| PhysicalObject/MatterProcessingObject/ThermalSeparatingObject | 1 | `physical_objects` |
| PhysicalObject/Organism/Person | 16 | `physical_objects` |
| PhysicalObject/PresentingObject/AcousticDevice | 26 | `physical_objects` |
| PhysicalObject/PresentingObject/GraphicalDisplay | 6 | `physical_objects` |
| PhysicalObject/PresentingObject/MultipleFormPresentingObject | 9 | `physical_objects` |
| PhysicalObject/PresentingObject/ScalarDisplay | 15 | `physical_objects` |
| PhysicalObject/PresentingObject/VisibleStateIndicator | 9 | `physical_objects` |
| PhysicalObject/ProtectingObject/EnvironmentProtectingObject | 3 | `physical_objects` |
| PhysicalObject/ProtectingObject/FireProtectingObject | 16 | `physical_objects` |
| PhysicalObject/ProtectingObject/OvercurrentProtectingObject | 5 | `physical_objects` |
| PhysicalObject/ProtectingObject/PressureProtectingObject | 11 | `physical_objects` |
| PhysicalObject/ProtectingObject/PreventiveProtectingObject | 13 | `physical_objects` |
| PhysicalObject/ProtectingObject/TemperatureProtectingObject | 6 | `physical_objects` |
| PhysicalObject/ProtectingObject/WearProtectingObject | 77 | `physical_objects` |
| PhysicalObject/RestrictingObject/AccessRestrictingObject | 2 | `physical_objects` |
| PhysicalObject/RestrictingObject/ElectricityRestrictingObject | 10 | `physical_objects` |
| PhysicalObject/RestrictingObject/ElectricityStabilisingObject | 2 | `physical_objects` |
| PhysicalObject/RestrictingObject/FlowRestrictor | 3 | `physical_objects` |
| PhysicalObject/RestrictingObject/LocalClimateStabilisingObject | 3 | `physical_objects` |
| PhysicalObject/RestrictingObject/MovementRestrictingObject | 46 | `physical_objects` |
| PhysicalObject/RestrictingObject/ReturnFlowRestrictingObject | 2 | `physical_objects` |
| PhysicalObject/SensingObject | 8 | `physical_objects` |
| PhysicalObject/SensingObject/AudioVisualSensingObject | 5 | `physical_objects` |
| PhysicalObject/SensingObject/ConcentrationSensingObject | 15 | `physical_objects` |
| PhysicalObject/SensingObject/FieldSensingObject | 1 | `physical_objects` |
| PhysicalObject/SensingObject/FlowSensingObject | 3 | `physical_objects` |
| PhysicalObject/SensingObject/ForceSensingObject | 4 | `physical_objects` |
| PhysicalObject/SensingObject/LevelSensingObject | 9 | `physical_objects` |
| PhysicalObject/SensingObject/PhysicalDimensionSensingObject | 3 | `physical_objects` |
| PhysicalObject/SensingObject/PowerSensingObject | 7 | `physical_objects` |
| PhysicalObject/SensingObject/PressureSensingObject | 7 | `physical_objects` |
| PhysicalObject/SensingObject/ResistivitySensingObject | 1 | `physical_objects` |
| PhysicalObject/SensingObject/TemperatureSensingObject | 3 | `physical_objects` |
| PhysicalObject/SensingObject/TimeRatingObject | 3 | `physical_objects` |
| PhysicalObject/SensingObject/TimeSensingObject | 2 | `physical_objects` |
| PhysicalObject/StoringObject/ElectrochemicalStoringObject | 12 | `physical_objects` |
| PhysicalObject/StoringObject/EnclosedStationaryStoringObject | 30 | `physical_objects` |
| PhysicalObject/StoringObject/InductiveStoringObject | 1 | `physical_objects` |
| PhysicalObject/StoringObject/MechanicalEnergyStoringObject | 5 | `physical_objects` |
| PhysicalObject/StoringObject/MoveableStoringObject | 25 | `physical_objects` |
| PhysicalObject/StoringObject/OpenStationaryStoringObject | 8 | `physical_objects` |
| PhysicalObject/Substance/Gas | 8 | `physical_objects` |
| PhysicalObject/Substance/Liquid | 5 | `physical_objects` |
| PhysicalObject/Substance/Mixture | 1 | `physical_objects` |
| PhysicalObject/Substance/Solid | 4 | `physical_objects` |
| PhysicalObject/TransformingObject/ElectricEnergyConvertingObject | 8 | `physical_objects` |
| PhysicalObject/TransformingObject/ElectricEnergyTransformingObject | 11 | `physical_objects` |
| PhysicalObject/TransformingObject/MatterReshapingObject | 7 | `physical_objects` |
| PhysicalObject/TransformingObject/MechanicalEnergyTransformingObject | 97 | `physical_objects` |
| PhysicalObject/TransformingObject/OrganicPlant | 4 | `physical_objects` |
| PhysicalObject/TransformingObject/SignalConvertingObject | 18 | `physical_objects` |
| Process/UndesirableProcess | 146 | `processes` |
| Property | 2 | `properties` |
| Property/DesirableProperty | 1 | `properties` |
| Property/UndesirableProperty | 32 | `properties` |
| State/UndesirableState | 178 | `states` |
| State/UndesirableState/DegradedState | 45 | `states` |
| State/UndesirableState/FailedState | 215 | `states` |

</details>

### Silver

- 文件 `data/silver_release.json`，7,000 条，4,607,249 字节。
- SHA-256 `af8f785e5c03fdaefeb9b94dad7d24cbd030c2fa4b53ee452069d035f4bf4877`。
- 22,122 个实体实例，5 个实际发行标签，15,200 条关系。
- 空文本或空 token 列表 0 条，零实体 22 条，零关系 272 条。
- 按原文本精确去重 6,996 条，按 NFKC、casefold、空白折叠去重 6,995 条。
- 文本是含占位符和缩写的英文维修短文本。

| 长度口径 | 最小 | 中位数 | P95 | 最大 | 均值 |
|---|---:|---:|---:|---:|---:|
| 字符 | 4 | 33 | 50 | 78 | 33.33 |
| 发行数据 token | 1 | 5 | 8 | 13 | 5.37 |

| 原始关系标签 | 实例数 |
|---|---:|
| hasPatient | 7,761 |
| hasPart | 3,837 |
| isA | 2,512 |
| hasAgent | 789 |
| contains | 178 |
| hasProperty | 123 |

<details>
<summary>Silver 全部原始实体标签与次数</summary>

| 标签 | 实例数 | MVP 字段 |
|---|---:|---|
| Activity | 5,045 | `activities` |
| PhysicalObject | 13,472 | `physical_objects` |
| Process | 728 | `processes` |
| Property | 130 | `properties` |
| State | 2,747 | `states` |

</details>

## 完整性与重叠

- Gold 与 Silver 归一化文本交集 7 个。原文本的精确交集同样为 7 个。
- 至少 5 个按空白切分的 token、等长、恰好一个同位置 token 不同的跨库候选共有 157 对，涉及 86 条 Gold 和 132 条 Silver 归一化文本。候选包含独立工单的真实差异，需结合规则或抽查处理。
- 发行数据实体 token 区间与关系索引均未发现越界。严格将跨度 token 用空格拼接后检查原文包含关系，Gold 有 7 个跨度、5 条记录不匹配，0 基索引为 825、858、882、905、1038。Silver 在同一口径下为 0。
- 发行脚本按单个语料做 80/10/10 切片，源码中的 shuffle 作用于构建完 output_data 后的原 data。项目采用显式的跨语料去重与分组划分。
- Gold 原始标签细分至最多三级，按第一个 / 之前的根标签归入上述五字段。Silver 直接映射五个顶层名称。P0 保留实体文本，不训练关系抽取。覆盖多个文本位置的同类同名实体在输出与评测中按集合去重。
- Silver 先由用细粒度语料训练的 SpERT 预标注，再由单人复核。基于此版 Silver 训练并在 Gold 上评测仍受上游标注器来源限制。
- [仓库许可证](https://github.com/nlp-tlp/maintie/blob/825e7710e4129598da997276a38b3cb40b200bc2/LICENSE.md) 为 MIT，分发副本或实质性内容时保留上游许可证与版权声明。论文 PDF 标注 CC BY-NC 4.0。

## 十条真实文本示例

| 语料 | 0 基索引 | 原文 |
|---|---:|---|
| Gold | 16 | `<id> air leak` |
| Gold | 24 | `<id> alternator failed` |
| Gold | 27 | `<id> alternator overcharged` |
| Gold | 30 | `<id> armrest unserviceable` |
| Gold | 45 | `<id> batteries discharging` |
| Silver | 6 | `replace drive pulley` |
| Silver | 9 | `fuel nozzle` |
| Silver | 22 | `repair cabin floodlight` |
| Silver | 47 | `west side bearing` |
| Silver | 67 | `hoist hose leaking` |

## 五条原始记录及 SFT 转换示例

下列记录用于解释转换规则，实际 few-shot 示例由冻结划分后的训练集另行选取。

### Silver [6]

原始记录

```json
{"text": "replace drive pulley", "tokens": ["replace", "drive", "pulley"], "entities": [{"start": 0, "end": 1, "type": "Activity"}, {"start": 1, "end": 3, "type": "PhysicalObject"}, {"start": 2, "end": 3, "type": "PhysicalObject"}], "relations": [{"head": 0, "tail": 1, "type": "hasPatient"}, {"head": 1, "tail": 2, "type": "isA"}]}
```

SFT 样本

```json
{"prompt": [{"role": "system", "content": "Extract verbatim entities as one JSON object with the keys physical_objects, activities, states, processes, properties. Use arrays of strings. Use empty arrays when absent. Do not infer actions or causes."}, {"role": "user", "content": "replace drive pulley"}], "completion": [{"role": "assistant", "content": "{\"physical_objects\": [\"drive pulley\", \"pulley\"], \"activities\": [\"replace\"], \"states\": [], \"processes\": [], \"properties\": []}"}]}
```

### Silver [9]

原始记录

```json
{"text": "fuel nozzle", "tokens": ["fuel", "nozzle"], "entities": [{"start": 0, "end": 2, "type": "PhysicalObject"}, {"start": 1, "end": 2, "type": "PhysicalObject"}], "relations": [{"head": 0, "tail": 1, "type": "isA"}]}
```

SFT 样本

```json
{"prompt": [{"role": "system", "content": "Extract verbatim entities as one JSON object with the keys physical_objects, activities, states, processes, properties. Use arrays of strings. Use empty arrays when absent. Do not infer actions or causes."}, {"role": "user", "content": "fuel nozzle"}], "completion": [{"role": "assistant", "content": "{\"physical_objects\": [\"fuel nozzle\", \"nozzle\"], \"activities\": [], \"states\": [], \"processes\": [], \"properties\": []}"}]}
```

### Silver [22]

原始记录

```json
{"text": "repair cabin floodlight", "tokens": ["repair", "cabin", "floodlight"], "entities": [{"start": 0, "end": 1, "type": "Activity"}, {"start": 1, "end": 2, "type": "PhysicalObject"}, {"start": 2, "end": 3, "type": "PhysicalObject"}], "relations": [{"head": 0, "tail": 2, "type": "hasPatient"}, {"head": 1, "tail": 2, "type": "hasPart"}]}
```

SFT 样本

```json
{"prompt": [{"role": "system", "content": "Extract verbatim entities as one JSON object with the keys physical_objects, activities, states, processes, properties. Use arrays of strings. Use empty arrays when absent. Do not infer actions or causes."}, {"role": "user", "content": "repair cabin floodlight"}], "completion": [{"role": "assistant", "content": "{\"physical_objects\": [\"cabin\", \"floodlight\"], \"activities\": [\"repair\"], \"states\": [], \"processes\": [], \"properties\": []}"}]}
```

### Silver [47]

原始记录

```json
{"text": "west side bearing", "tokens": ["west", "side", "bearing"], "entities": [{"start": 2, "end": 3, "type": "PhysicalObject"}], "relations": []}
```

SFT 样本

```json
{"prompt": [{"role": "system", "content": "Extract verbatim entities as one JSON object with the keys physical_objects, activities, states, processes, properties. Use arrays of strings. Use empty arrays when absent. Do not infer actions or causes."}, {"role": "user", "content": "west side bearing"}], "completion": [{"role": "assistant", "content": "{\"physical_objects\": [\"bearing\"], \"activities\": [], \"states\": [], \"processes\": [], \"properties\": []}"}]}
```

### Silver [67]

原始记录

```json
{"text": "hoist hose leaking", "tokens": ["hoist", "hose", "leaking"], "entities": [{"start": 0, "end": 1, "type": "PhysicalObject"}, {"start": 1, "end": 2, "type": "PhysicalObject"}, {"start": 2, "end": 3, "type": "Process"}], "relations": [{"head": 0, "tail": 1, "type": "hasPart"}, {"head": 2, "tail": 1, "type": "hasAgent"}]}
```

SFT 样本

```json
{"prompt": [{"role": "system", "content": "Extract verbatim entities as one JSON object with the keys physical_objects, activities, states, processes, properties. Use arrays of strings. Use empty arrays when absent. Do not infer actions or causes."}, {"role": "user", "content": "hoist hose leaking"}], "completion": [{"role": "assistant", "content": "{\"physical_objects\": [\"hoist\", \"hose\"], \"activities\": [], \"states\": [], \"processes\": [\"leaking\"], \"properties\": []}"}]}
```

