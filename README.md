# yolop_ncnn
手把手教你从模型转换到部署YOLOP多任务模型（Hand in hand teaching you how to transition from models to deploying YOLOP multitasking models）



## 起

最近有点想学习YOLOP这个模型，因为脑子里对自驾领域的工作还是有所期待，想着先学习一点相关的模型，以后如果真的想跳槽的话，面试被问到还能回答出一二。于是跟着原始[YOLOP](https://github.com/hustvl/YOLOP)的Github仓库先学习网络结构和训练细节，最后我还是想要学习如何部署这个模型。因为我最常使用的部署平台是ncnn，所以想将YOLOP这个模型部署到该平台上。

在Github上搜索一番之后，发现针对YOLOP的ncnn部署不能说是没有，但是感觉很迷不直观。另外YOLOP在模型转换过程中有很多的坑，比如说用到的Focus模块转换到ncnn时会报错，另外比如目标检测的几个检测头在转换过程中也会出现这样那样的问题。经过我在Github上地毯式地搜索之后，我终于发现一个让我比较满意的YOLOP模型的ncnn实现，这里给出该仓库的链接：[lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit/tree/main)。该仓库涵盖了各种模型的各种平台部署，其中就包含有YOLOP的。由于该库的集成度很强，并且我只想使用里面YOLOP的ncnn平台部署代码，所以我想尝试将该部分代码剥离出来，更方便使用，因此便有了本仓库的代码实现。

**本仓库的代码，绝大部分是来自[lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit/tree/main)已有的实现，本人只做了些许耦合模块的剥离。由于该仓库只有ncnn平台的推理代码，而并未提供从Pytorch训练出的模型，转到到onnx，再转换到ncnn框架的整个过程。为了让大家少走弯路，本仓库提供完整的模型转换到ncnn平台部署过程。**



## 承

1. 从pth转onnx

   原始的YOLOP的官方仓库给出了[export_onnx.py](https://github.com/hustvl/YOLOP/blob/main/export_onnx.py)转换脚本，但是，如果你想成功的将权重转换到ncnn平台，并且使用本仓库的部署代码是无法成功推理成功的，上述内容中我也提到过，YOLOP有些许模块或者算子无法成功转换到ncnn框架中，因此需要做进一步处理。为了能够契合已有的ncnn推理代码，在将训练出的pth权重转换到onnx模型时，需要做以下处理：

   （1）修改官方lib/models下[common.py](https://github.com/hustvl/YOLOP/blob/main/lib/models/common.py)中的Detect模块，将其替换成如下代码：

   ```python
   class Detect(nn.Module):
       stride = None  # strides computed during build
   
       def __init__(self, nc=13, anchors=(), ch=()):  # detection layer
           super(Detect, self).__init__()
           self.nc = nc  # number of classes
           self.no = nc + 5  # number of outputs per anchor 85
           self.nl = len(anchors)  # number of detection layers 3
           self.na = len(anchors[0]) // 2  # number of anchors 3
           self.grid = [torch.zeros(1)] * self.nl  # init grid 
           a = torch.tensor(anchors).float().view(self.nl, -1, 2)
           self.register_buffer('anchors', a)  # shape(nl,na,2)
           self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
           self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv  
   
       def forward(self, x):
           if not torch.onnx.is_in_onnx_export():
               z = []  # inference output
               for i in range(self.nl):
                   x[i] = self.m[i](x[i])  # conv
                   # print(str(i)+str(x[i].shape))
                   bs, _, ny, nx = x[i].shape  # x(bs,255,w,w) to x(bs,3,w,w,85)
                   x[i]=x[i].view(bs, self.na, self.no, ny*nx).permute(0, 1, 3, 2).view(bs, self.na, ny, nx, self.no).contiguous()
                   # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                   # print(str(i)+str(x[i].shape))
   
                   if not self.training:  # inference
                       if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                           self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                       y = x[i].sigmoid()
                       #print("**")
                       #print(y.shape) #[1, 3, w, h, 85]
                       #print(self.grid[i].shape) #[1, 3, w, h, 2]
                       y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                       y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                       """print("**")
                       print(y.shape)  #[1, 3, w, h, 85]
                       print(y.view(bs, -1, self.no).shape) #[1, 3*w*h, 85]"""
                       z.append(y.view(bs, -1, self.no))
               return x if self.training else (torch.cat(z, 1), x)
           else:
               for i in range(self.nl):
                   x[i] = self.m[i](x[i])  # conv
                   bs, _, ny, nx = x[i].shape  # x(bs,255,w,w) to x(bs,3,w,w,85)
                   x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                   x[i] = x[i].view(bs, -1, self.no)
               return x
   ```

   （2）修改官方给出的[export_onnx.py](https://github.com/hustvl/YOLOP/blob/main/export_onnx.py)脚本，由于这里贴出修改内容篇幅过长影响阅读，可以将本仓库提供的"export_onnx.py"对原始仓库的转换脚本进行替换。随后便可将训练好的权重转换到onnx平台。转换成功后你可以在控制台看到如下输出：

   ```
   Input:  NodeArg(name='images', type='tensor(float)', shape=[1, 3, 640, 640])
   Output:  NodeArg(name='det_stride_8', type='tensor(float)', shape=[1, 19200, 6])
   Output:  NodeArg(name='det_stride_16', type='tensor(float)', shape=[1, 4800, 6])
   Output:  NodeArg(name='det_stride_32', type='tensor(float)', shape=[1, 1200, 6])
   Output:  NodeArg(name='drive_area_seg', type='tensor(float)', shape=[1, 2, 640, 640])
   Output:  NodeArg(name='lane_line_seg', type='tensor(float)', shape=[1, 2, 640, 640])
   read onnx using onnxruntime sucess
   ```

从onnx到ncnn

经过以上pth到onnx的转换，距离成功将pth模型转换到ncnn平台仅剩一个拦路虎，那就是模型中的`Focus`模块，当然解决该问题并不困难，因为YOLOV5中也有该模块，只要参考YOLOv5部署到ncnn平台对该模块的处理过程，便可成功将YOLOP转换到ncnn平台。具体的转换过程如下：

（1）使用`onnx2ncnn`将onnx模型转换到ncnn平台，转换脚本如下：

```
onnx2ncnn.exe yolop-640-640.onnx yolop-640-640.param yolop-640-640.bin # (Windows平台）
./onnx2ncnn yolop-640-640.onnx yolop-640-640.param yolop-640-640.bin # (Linux平台)
```

![](https://pic.imgdb.cn/item/674ff8c7d0e0a243d4dd1112.jpg)

从图中可以看出，出现一些算子不支持的提示，因此需要进一步处理。

(2) 使用Notepad++(记事本也可以)打开`yolop-640-640.param`，进行模块参数修改。

`yolop-640-640.param`打开后如下图所示：

[![pAodhOP.jpg](https://s21.ax1x.com/2024/12/04/pAodhOP.jpg)](https://imgse.com/i/pAodhOP)

图中画红框的地方为'Focus'模块的参数，我们需要对其进行替换，替换后的内容如下图所示：

[![pAodIw8.jpg](https://s21.ax1x.com/2024/12/04/pAodIw8.jpg)](https://imgse.com/i/pAodIw8)

参数修改准则可以参考如下博客：[**使用ncnn部署yolox详细记录**](https://www.bilibili.com/opus/766952404741521446)。

使用Netron对ncnn模型进行可视化，模型参数修改前后，模型的结构有如下变化：

- 修改前

  [![pAodoTS.jpg](https://s21.ax1x.com/2024/12/04/pAodoTS.jpg)](https://imgse.com/i/pAodoTS)

- 修改后

[![pAod7Fg.jpg](https://s21.ax1x.com/2024/12/04/pAod7Fg.jpg)](https://imgse.com/i/pAod7Fg)

（3）对修改后的参数进行`ncnnoptimize`优化，增加推理速度，转换脚本如下：

```
ncnnoptimize.exe yolop-640-640.param yolop-640-640.bin yolop-640-640-opt.param yolop-640-640-opt.bin 1 # (Windows平台）
./ncnnoptimize yolop-640-640.param yolop-640-640.bin yolop-640-640-opt.param yolop-640-640-opt.bin 1 #  # (Linux平台)
```

转换后便可获得优化后的`.param`和`.bin`文件，生成的两个文件可用于本仓库ncnn框架的YOLOP代码推理。

转换过程如下图所示：

[![pAodHYQ.jpg](https://s21.ax1x.com/2024/12/04/pAodHYQ.jpg)](https://imgse.com/i/pAodHYQ)



## 转

经过以上步骤，便可成功将YOLOP训练好的pt权重转换到ncnn平台进行部署，生成的权重可以完美匹配本仓库提供的推理代码。

无论你是在Windows平台还是Linux(Ubuntu)平台，为了能够成功运行本仓库的推理代码，你还需要准备以下的依赖库（版本仅供参考）：

- ncnn-20240820-full-source
- opencv-3.4.10

这些依赖库的安装可以自行百度或者谷歌。

部署成功之后，进行图片推理结果如下：

-测试图片

[![pAodOln.jpg](https://s21.ax1x.com/2024/12/04/pAodOln.jpg)](https://imgse.com/i/pAodOln)

-推理结果（目标检测）

![](https://pic.imgdb.cn/item/674ff7d8d0e0a243d4dd10d7.jpg)

--推理结果（可行驶区域掩码）

![](https://pic.imgdb.cn/item/674ff84fd0e0a243d4dd10f5.jpg)

-推理结果（车道线掩码）

![](https://pic.imgdb.cn/item/674ff850d0e0a243d4dd10f7.jpg)

-推理结果（整体结果融合）

![](https://pic.imgdb.cn/item/674ff850d0e0a243d4dd10f8.jpg)



注：为了让本仓库的代码显得简洁，这里给出的是在Ubuntu平台的推理代码，我自己在Windows平台的Visual Studio2019部署过，也同样可以正常运行。



## 合

为了能够成功将YOLOP部署到ncnn平台上，自己也花了一周多的时间，踩了各种坑最终才能部署成功，如果你觉得本仓库还算有用的话，麻烦给我一个Star或者Fork，这是我的更新和学习的动力。

以上。



## Reference

-[YOLOP](https://github.com/hustvl/YOLOP)

-[lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit/tree/main)
