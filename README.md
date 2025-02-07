# AIModelHub

需要依赖opencv和onnxruntime。

VC++目录 - 包含库目录引入opencv\build\include头文件

VC++目录 - 库目录引入opencv\build\x64\vc16\lib静态库文件

链接器 - 输入 - 附加依赖项引入对应的静态库文件名称

onnxruntime可以通过nuget引入