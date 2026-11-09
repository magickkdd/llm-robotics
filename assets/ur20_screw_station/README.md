# UR20 电池拆解实验场景

这是后续 LLM policy 电池拆解实验使用的 MuJoCo 场景。场景将 UR20 机械臂、附件中的螺丝刀末端工具、NS.10LA 套筒、电池包和 M6 螺钉组装在一起。

## 场景入口

- MuJoCo 场景：`scene.xml`
- 场景生成器：`../../scripts/scene/build_screw_scene.py`
- UR20 原始模型：`ur20/ur20.xml`
- 工具和工件模型：`workcell/`
- 合并后的网格资源：`scene_assets/`

## 当前实验状态

- 螺丝刀工具刚性安装在 UR20 腕部法兰；工具不再有未驱动的轴向滑动自由度。
- 螺钉碰撞头为 10 mm 对边外六角；套筒六个碰撞面的实际内对边约 11.0 mm，对边单侧间隙约 0.5 mm（比 10.3 mm 的名义视觉开口略松）。
- 电池放置在工作台上，场景可直接加载并运行 MuJoCo 仿真。
- 当前目标螺钉装在电池裙边法兰的前侧 09 号孔位（电池局部 `(0.2882, -0.4668)`）；螺钉名称、头顶高度和 `agentview` ROI 集中声明在 `../../src/ur20_screw/target.py`，迁移到其他孔位时同步修改该文件与 `scene.xml` 即可。
- UR20 benchmark 从螺钉上方约 13 cm 且水平偏移 8–10 cm 处开始，需要策略自行定位、移动、对准、套入并旋出。
- 观测相机均为固定相机：`agentview` 固定对准目标孔位；`wristview` 刚性固定在腕部工具上，因此套筒在腕部图像中的位置恒定，画面移动只来自工具与工件的相对运动。成功要求有效套接期间发生碰撞接触，并将螺钉旋出至少 20 mm；仅正常结束 rollout 不算任务成功。

## 重新生成场景

在仓库根目录执行：

```bash
python scripts/scene/build_screw_scene.py
```

生成器会重建本目录下的 `scene.xml`（目标孔位、4 条 equality、相机、FT 传感器与力矩 `motor` 执行器），并保持 `scene_assets/` 中的相对资源路径；重复运行结果一致。

迁移到其他螺钉时：修改 `scripts/scene/build_screw_scene.py` 的 `TARGET_SCREW_POS`，在 `../../src/ur20_screw/target.py` 同步名称、头顶高度与 `agentview` ROI，从 `workcell/parts/installed_bolts.xml` 删除该孔位的装饰螺钉，然后重新运行生成器。

policy 可使用 `attachment_site` 作为末端位姿参考，以 `tool_tip` 作为套筒工作点，并通过 `socket_velocity` 控制套筒旋转。

## Policy 接入进度

UR20 控制接口与本地 mock 闭环已接入；真实 API rollout 需显式运行对应命令：

- JSON-lines worker：`../../src/ur20_screw/simulation/worker.py`
- 本地 mock policy：`../../scripts/rollout/run_mock.py`
- worker 提供 `agentview`、`wristview` 两路 JPEG 观测，以及关节、速度和关键 site 状态。
- `screw_front`、`screw_back`、`screw_side` 仅供内部评估和套接判定，不发送给策略模型。

运行无 API 的本地闭环测试：

```bash
MUJOCO_GL=egl python scripts/rollout/run_mock.py
```

该测试验证 worker 和动作接口；`scripts/rollout/test_benchmark.py` 会验证 10 个可复现初始状态，以及通过套筒碰撞带动螺钉旋出。

## 螺丝拆卸任务基准

任务从 M6 外六角上方、带横向偏差的位置开始；策略需要移动末端、对准并套入套筒，随后旋转螺丝，直到观测到至少 20 mm 的旋出量。

共提供 10 个确定性初始状态（`state_id` 0–9）。单独运行一个状态可使用 `scripts/rollout/run_episode.py --state-id N`，程序会自动创建唯一的时间戳目录；依次运行全部状态使用 `scripts/rollout/run_benchmark.py`，结果自动保存到带时间戳的目录中，每个状态各有独立子目录。

螺钉旋转与轴向位移仍通过理想螺旋约束耦合。因此，该基准覆盖视觉对准、套筒接合和旋出过程，不模拟真实螺纹制造或电池外壳形变。运行 `scripts/scene/build_collision_mesh.py` 可重新生成外六角碰撞网格。
