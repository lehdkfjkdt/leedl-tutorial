# 机器人“大脑”层：任务、地图与全局路径规划

这里介绍与 [小脑](../小脑/README.md) 对应的机器人上层“大脑”：它决定**去哪里、为什么去、经过哪些区域**，再把全局路线或局部子目标交给 NavRL 等局部策略执行。

主文档：[01_机器狗导航大脑_论文与系统详解.md](./01_机器狗导航大脑_论文与系统详解.md)

本文基于 `paper-review` 中的 ABot-N0、InternVLA-N1/DualVLN、NavRL++ 和 Xiaomi-Robotics-0 论文导读，重点讨论 Unitree Go2 等四足机器狗的认知大脑、waypoint 规划、异步执行和 NavRL 小脑接口。
