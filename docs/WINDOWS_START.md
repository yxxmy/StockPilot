# Windows 一键启动（本地版）

2026-11-20修复：请使用Fixed分享包。启动PowerShell脚本必须保留UTF-8 BOM编码；Windows PowerShell5可能将无BOM中文脚本错误解码，导致路径变量为空。

适合把源码分享给朋友，在自己的 Windows 10/11 x64 电脑运行。网页、行情和报告均由使用者电脑处理；不是公网部署，也不是已打包的exe安装程序。

## 使用

1. 解压项目到有写入权限的普通文件夹，例如“文档/StockPilot”，不要在压缩包内直接运行。
2. 双击根目录 `Start_StockPilot.bat`，保留黑色窗口。
3. 启动器检查当前项目的 `.venv`、上次通过检查的环境、用户目录下的共享环境，以及系统可发现的 Python。先验证依赖版本、依赖冲突和关键库导入，通过即可复用。缺少依赖环境时才在 `%LOCALAPPDATA%\StockPilot\environments` 准备，缺少可用的 Python 3.12 x64 时再从 python.org 下载并验证签名。
4. 检查依赖并启动本地服务，健康检查成功后自动打开默认浏览器。终端会显示实际网址，每次可能不同，只监听127.0.0.1；不会误连8502测试页面。
5. 选择“本地规则”，点击“开始分析”。启动不会自动提交任务。关闭服务请在窗口按 Ctrl+C；不要关闭窗口后继续访问旧网址。

首次安装需要联网，PyTorch 等包较大，时间和磁盘占用取决于机器及网络。新版启动成功后会记录通过验证的 Python 路径。以后即使把源码解压到新目录，只要该环境还在且符合新版依赖，就能继续使用。报告、PDF、索引和模型文件跟随各自的项目目录。

如果可用的旧环境在另一个文件夹，且从未被新版启动器记录，可在新目录的 `scripts/python-local.txt` 写入旧环境中 `Scripts\python.exe` 的完整路径，每个文件只写一行。启动器验证通过后会记住这个位置。

## RAG 与 DeepSeek

需要PDF检索时，先双击 `Download_RAG_Model.bat` 下载本地BGE模型，然后按 README 导入真实PDF。此入口会准备依赖和下载模型，不启动分析、不创建公告内容。

本地规则、时序功能不需要密钥。DeepSeek仍只读取使用者电脑的 `DEEPSEEK_API_KEY` 环境变量；启动器不收集、显示或保存密钥。不要把自己的 `.env`、密钥、`.venv`、`.runtime`、`data` 或私人报告一起发给别人。已有系统环境变量可能被应用继承，使用者选择DeepSeek并提交才产生费用。

## 故障与手动模式

安装和服务输出保留在窗口，失败时可查看具体提示。若签名验证、下载或安装失败，请处理对应问题后重新运行启动器。已有环境无法通过检查时，启动器会继续找其他可用环境。

PowerShell策略由本次启动进程使用，不修改系统策略；企业禁止脚本或软件安装时需遵守本机管理要求。路径建议避免特殊字符 `%`、`!`。安装依赖没有锁定全部间接版本，未来新机器安装仍需回归验证。

在项目目录可执行：

```powershell
# 只检查已有环境，不下载、不启动网页
powershell -NoProfile -File scripts/start_windows.ps1 -CheckOnly
# 只准备依赖，不启动网页
powershell -NoProfile -File scripts/start_windows.ps1 -PrepareOnly
# 原手动入口仍可用
$stockPilotPython = Get-Content -LiteralPath (Join-Path $env:LOCALAPPDATA 'StockPilot\python-ready.txt') -Encoding UTF8 -First 1
& $stockPilotPython web_app.py --server.port 8501
```

验证记录见 PROJECT_PLAN.md：502项回归通过，已有环境检查及真实服务健康/首页200通过，默认浏览器调用通过替身测试。固定Python3.12.10有官方传统Windows安装程序，但不是最新安全修订版本，发布前需评估运行时更新。本机已有环境验证不能代替全新 Windows 机器的安装验收；正式对外发放前，还需在无Python的干净Windows x64机器完成下载、签名、安装、重启和异常网络测试。
