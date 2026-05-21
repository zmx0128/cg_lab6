# 可微分渲染与网格优化

> 将一个初始的"球体"通过梯度下降，逐渐"捏"成一头"奶牛"的形状。

## 📋 实验目标

1. **理解可微光栅化原理**：掌握在处理离散几何体（Mesh）边界时的数学近似方法
2. **多视角图像反推三维网格**：通过二维图像（剪影/RGB）优化三维空间中的网格顶点坐标
3. **网格正则化的重要性**：深刻理解正则化对于防止拓扑崩坏和陷入局部最优的决定性作用

## 🔬 实验原理

### 核心问题：从球体到奶牛

本实验的核心是通过梯度下降算法，将一个初始球体逐步变形为目标奶牛形状。这个过程需要解决两个关键问题：

### 1. 防梯度消失：软光栅化 (Soft Rasterization)

在传统渲染（硬光栅化）中，像素要么在三角形内，要么在三角形外。这种阶跃变化导致边界处的梯度为 0，网络无法知道顶点该往哪个方向移动。

**解决方案**：使用软光栅化，通过计算像素到三角形边缘的距离，并利用 Sigmoid 函数在边界处产生平滑的概率过渡：

$$A(d) = \text{sigmoid}\left(\frac{d}{\sigma}\right)$$

其中 $\sigma$ 控制边缘的模糊程度。这样，即使顶点在像素外部，也能提供微小但非零的梯度，引导顶点向正确方向移动。

### 2. 防局部最优：网格正则化 (Mesh Regularization)

如果仅依靠图像差异（Loss）去移动顶点，顶点会为了迎合某个视角的投影而疯狂交叉、重叠，最终变成一团充满尖刺的"刺猬"，彻底陷入局部最优。

**解决方案**：引入三种正则化损失：

| 正则化类型 | 作用 | 权重 |
| :--- | :--- | :--- |
| **拉普拉斯平滑** | 约束相邻顶点，防止表面出现尖锐突起 | 1.0 |
| **边长一致性** | 惩罚过长或过短的边，防止三角形严重拉伸 | 0.1 |
| **法线一致性** | 约束相邻三角形面的法线方向接近，保持表面平滑 | 0.01 |

**最终总 Loss**：

$$L_{total} = L_{silhouette} + w_{lap}L_{lap} + w_{edge}L_{edge} + w_{normal}L_{normal}$$

## 📁 文件结构

```
├── main.ipynb             
 # 基于剪影的形状优化
├── texture_optimization.ipynb  
# 联合纹理优化
├── cow.obj                 
# 目标奶牛模型（仅几何信息）
├── cow_with_color.obj      
# 带顶点颜色的奶牛模型
├── output_meshes/            
 # 形状优化输出（300轮迭代）
│   ├── mesh_epoch_000.obj
│   ├── mesh_epoch_020.obj
│   └── ...
├── output_textured_meshes/  
 # 纹理优化输出（2000轮迭代）
│   ├── mesh_epoch_0000.obj
│   ├── mesh_epoch_0200.obj
│   └── ...
└── sphere_to_cow_color/  
 # 纹理优化输出（3000轮迭代）
    ├── mesh_epoch_0000.obj
    └── ...
```

## 🛠️ 环境配置

### 依赖安装

由于涉及底层 CUDA/C++ 算子，需安装以下依赖：

```bash
# 更新 pip
pip install --upgrade pip

# 基础依赖
pip install fvcore iopath matplotlib ninja

# PyTorch3D（从国内镜像安装）
pip install "git+https://gitee.com/hongwenzhang/pytorch3d.git" --no-build-isolation
```

### 环境要求

| 依赖 | 版本 |
| :--- | :--- |
| Python | >= 3.8 |
| PyTorch | >= 1.10 |
| PyTorch3D | >= 0.7.0 |
| CUDA | >= 11.0 |

### ModelScope 云平台配置

阿里云魔搭社区（ModelScope）是一个稳定、免费、且自带预装深度学习环境的纯净云平台。它为实名认证用户提供免费的 GPU 算力（基于阿里云 DSW 实验室），能够尽可能避免不同操作系统下环境配置带来的困难。

#### 1. 注册并领取免费算力

1. 打开浏览器，访问 [魔搭社区 (ModelScope)](https://modelscope.cn/)
2. 点击右上角的"登录/注册"，使用手机号或阿里云账号快捷登录
3. 登录后，点击顶部导航栏的 **"我的Notebook"**
4. 在弹出的页面中，找到 **"免费算力"** 区域，点击"启动"

#### 2. 基础配置

1. 等待几分钟，机器启动完成后，点击 **"查看Notebook"** 进入网页版的编程界面（JupyterLab）
2. 下载 `cow.obj` 文件
3. 将文件直接拖拽到云平台中
4. 在模型同目录下新建 `.ipynb` 文件

> **中间状态查看**：优化过程中生成的 `.obj` 文件可以下载到本地，使用 [MeshLab](https://www.meshlab.net/) 软件查看三维模型的变形过程。

## 🚀 使用方法

### 实验一：基础形状优化

运行 `main.ipynb`，执行基于剪影的形状优化：

1. 加载目标奶牛模型 `cow.obj`
2. 初始化一个细分等级为 4 的球体网格
3. 在 20 个均匀分布的视角下渲染目标剪影
4. 通过梯度下降优化顶点偏移量
5. 每 20 轮迭代保存一次中间结果到 `output_meshes/`

### 实验二：联合纹理优化（选做）

运行 `texture_optimization.ipynb`，执行形状与纹理的联合优化：

1. 加载目标奶牛模型并生成带纹理的目标图像
2. 同时优化顶点偏移量（形状）和顶点颜色（纹理）
3. 使用 SoftPhongShader 进行 RGB 渲染
4. 每 300 轮迭代保存一次中间结果到 `sphere_to_cow/`

## 📊 实验结果

### 形状优化过程

| 迭代次数 | 状态 |
| :--- | :--- |
| Epoch 0 | 初始球体 |
| Epoch 100 | 开始出现奶牛轮廓 |
| Epoch 200 | 形状逐渐清晰 |
| Epoch 300 | 优化完成，接近目标形状 |

### 目标剪影图

运行 `main.ipynb` 后，优化过程中的剪影对比效果：

![目标剪影图](images/silhouette_comparison.png)

### 目标模型多视角展示

从多个角度观察奶牛模型：

![目标模型多视角](images/target_model_views.png)

### 初始球体 vs 目标奶牛

![初始球体与目标奶牛对比](images/initial_vs_target.png)

### 联合优化过程

RGB + 剪影联合优化过程中的对比：

![联合优化过程](images/optimization_process.png)

### 球变牛变形过程

从球体逐渐变形为奶牛的完整过程：

![球变牛变形过程](images/deformation_process.png)

### 损失曲线

优化过程中监控以下损失项：
- **剪影损失**：预测剪影与目标剪影的 MSE
- **拉普拉斯损失**：网格平滑度约束
- **边长损失**：边长度一致性约束
- **法线损失**：法线方向一致性约束

## 🔧 参数说明

### 形状优化参数（main.ipynb）

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| `num_views` | 20 | 摄像机视角数量 |
| `epochs` | 300 | 迭代次数 |
| `lr` | 1.0 | 学习率 |
| `momentum` | 0.9 | 动量系数 |

### 联合优化参数（texture_optimization.ipynb）

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| `num_views` | 20 | 摄像机视角数量 |
| `epochs` | 3000 | 迭代次数 |
| `num_views_per_iter` | 2 | 每次迭代随机选择的视角数 |
| `plot_period` | 300 | 可视化周期 |

### 损失权重配置

```python
loss_weights = {
    "rgb": 1.5,           # RGB 损失权重
    "silhouette": 1.0,    # 剪影损失权重
    "edge": 0.1,          # 边长损失权重
    "laplacian": 1.0,     # 拉普拉斯损失权重
    "normal": 0.01        # 法线损失权重
}
```

## 📝 核心代码片段

### 可微渲染管线

```python
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras,
    RasterizationSettings, MeshRasterizer, SoftSilhouetteShader
)

# 摄像机配置
cameras = FoVPerspectiveCameras(
    device=device,
    R=look_at_view_transform(2.7, torch.zeros(num_views), torch.linspace(-180, 180, num_views))[0],
    T=look_at_view_transform(2.7, torch.zeros(num_views), torch.linspace(-180, 180, num_views))[1]
)

# 光栅化器与着色器
rasterizer = MeshRasterizer(
    cameras=cameras,
    raster_settings=RasterizationSettings(
        image_size=256,
        blur_radius=np.log(1./1e-4 - 1.)*1e-4,
        faces_per_pixel=50
    )
)
shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4))
```

### 优化循环

```python
for i in range(epochs):
    optimizer.zero_grad()
    
    # 形变网格
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    
    # 渲染剪影
    pred_silhouette = shader(rasterizer(new_src_mesh.extend(num_views)), new_src_mesh.extend(num_views))[..., 3]
    
    # 计算损失（含正则化）
    loss_silhouette = ((pred_silhouette - target_silhouette) ** 2).mean()
    loss = loss_silhouette + \
           1.0 * mesh_laplacian_smoothing(new_src_mesh) + \
           0.1 * mesh_edge_loss(new_src_mesh) + \
           0.01 * mesh_normal_consistency(new_src_mesh)
    
    loss.backward()
    optimizer.step()
```

## 🌟 关键技术点

### 1. 软剪影光栅化

使用 `SoftSilhouetteShader` 替代传统的硬光栅化，在三角形边界处产生平滑过渡，确保梯度能够正常流动。

### 2. 顶点偏移量优化

不直接优化顶点坐标，而是优化顶点偏移量 `deform_verts`，这样可以保持网格的拓扑结构不变。

### 3. 多视角训练

通过在多个视角下进行优化，确保生成的三维模型在各个角度都能匹配目标形状。

### 4. 正则化平衡

三种正则化项共同作用，防止网格退化：
- **拉普拉斯平滑**：保持表面光滑
- **边长一致性**：防止三角形过度拉伸
- **法线一致性**：保持相邻面的朝向一致

## 📚 参考资料

1. **PyTorch3D 官方文档**: [https://pytorch3d.org/](https://pytorch3d.org/)

2. **PyTorch3D 纹理优化教程**: [https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/fit_textured_mesh.ipynb](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/fit_textured_mesh.ipynb)
