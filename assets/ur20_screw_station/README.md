# UR20 电池拆解实验场景

这是后续 LLM policy 电池拆解实验使用的 MuJoCo 场景。场景将 UR20 机械臂、附件中的螺丝刀末端工具、NS.10LA 套筒、电池包和 M6 螺钉组装在一起。

## 场景入口

- MuJoCo 场景：`scene.xml`
- 场景生成器：`../../scripts/build_ur20_screw_scene.py`
- UR20 原始模型：`ur20/ur20.xml`
- 工具和工件模型：`workcell/`
- 合并后的网格资源：`scene_assets/`

## 当前实验状态

- 螺丝刀工具刚性安装在 UR20 腕部法兰；工具不再有未驱动的轴向滑动自由度。
- 螺钉碰撞头为 10 mm 对边外六角；套筒使用六个碰撞面构成 10.3 mm 对边开口，留有 0.3 mm 名义间隙。
- 电池放置在工作台上，场景可直接加载并运行 MuJoCo 仿真。
- UR20 benchmark 从螺钉上方约 13 cm 且水平偏移 8–10 cm 处开始，需要策略自行定位、移动、对准、套入并旋出。
- 观测相机为 `agentview` 和随腕部移动的 `wristview`。成功要求有效套接期间发生碰撞接触，并将螺钉旋出至少 20 mm；仅正常结束 rollout 不算任务成功。

## 重新生成场景

在仓库根目录执行：

```bash
/home/czx/miniconda3/envs/llm-libero/bin/python scripts/build_ur20_screw_scene.py
```

生成器会更新本目录下的 `scene.xml`，并保持 `scene_assets/` 中的相对资源路径。

policy 可使用 `attachment_site` 作为末端位姿参考，以 `tool_tip` 作为套筒工作点，并通过 `socket_velocity` 控制套筒旋转。

## Policy 接入进度

Inspect Robots embodiment 与本地 mock 闭环已接入；真实 API rollout 需显式运行对应命令：

- JSON-lines worker：`../../scripts/ur20_screw_worker.py`
- 本地 mock policy：`../../scripts/mock_ur20_screw_policy.py`
- worker 提供 `agentview`、`wristview` 两路 JPEG 观测，以及关节、速度和关键 site 状态。

运行无 API 的本地闭环测试：

```bash
MUJOCO_GL=egl /home/czx/miniconda3/envs/libero/bin/python \
  scripts/mock_ur20_screw_policy.py
```

该测试验证 worker 和动作接口；`scripts/test_ur20_screw_benchmark.py` 会验证 10 个可复现初始状态，以及通过套筒碰撞带动螺钉旋出。

## 螺丝拆卸任务基准

任务从 M6 外六角上方、带横向偏差的位置开始；策略需要移动末端、对准并套入套筒，随后旋转螺丝，直到观测到至少 20 mm 的旋出量。

共提供 10 个确定性初始状态（`state_id` 0–9）。单独运行一个状态可使用 `scripts/run_ur20_agent_rollout.py --state-id N`；依次运行全部状态使用 `scripts/run_ur20_benchmark.py`，结果自动保存到带时间戳的目录中，每个状态各有独立子目录。

螺钉旋转与轴向位移仍通过理想螺旋约束耦合。因此，该基准覆盖视觉对准、套筒接合和旋出过程，不模拟真实螺纹制造或电池外壳形变。运行 `scripts/build_ur20_collision_mesh.py` 可重新生成外六角碰撞网格。
