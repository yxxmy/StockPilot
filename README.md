# StockPilot 本地分享版

适用：Windows 10/11 x64。收到本压缩包后请先完整解压，再启动。

## 第一次使用

1. 解压到有写入权限的普通目录，例如“文档/StockPilot”。不要在压缩包内双击运行。
2. 双击 Start_StockPilot.bat。首次会联网下载独立Python、创建环境并安装依赖，包较大，需要等待，请保留终端窗口。
3. 环境检查和服务启动成功后，自动打开浏览器；若没有自动打开，请使用窗口中显示的网址。
4. 输入六位A股代码，先选“本地规则”，点击“开始分析”。启动本身不会自动提交任务。
5. 窗口中按Ctrl+C停止服务。窗口关闭后，原网址将无法继续使用；下次重新双击启动。

## 功能

- 真实历史行情、K线、成交量、MA/MACD/RSI等指标。
- LangGraph多Agent分析；可选财务新闻研究及DeepSeek文字解读。
- SQLite历史报告；你的报告保存在你自己的电脑。
- 本地PDF公告检索，返回原文、来源、物理页码和引用。
- 时序波动率实验：滚动方差与GARCH比较，预测波动幅度，不保证涨跌或盈利。
- MCP stdio只读接口；开发者可运行 .venv\Scripts\python.exe mcp_stdio.py。

## 可选功能

普通行情、本地规则和时序实验不需要大模型密钥。
DeepSeek读取你自己电脑的DEEPSEEK_API_KEY环境变量；选择DeepSeek并提交才产生API费用。本包不包含作者的密钥或额度。

PDF检索：先双击 Download_RAG_Model.bat 下载BGE模型，再按下方命令导入你自己合法取得、核验身份的文字PDF。本包不附带任何公司的正式财报、作者的PDF、索引、历史报告或行情快照。

在本目录打开PowerShell：

```powershell
# 导入PDF：下列路径、公司名称和其他元数据须换成核验后的真实值
.\.venv\Scripts\python.exe -m rag import-pdf "C:/Reports/已核验公告.pdf" --symbol 600519 --company-name "贵州茅台酒股份有限公司" --report-year 2023 --document-type announcement --disclosure-date 2024-06-12
.\.venv\Scripts\python.exe -m rag search "权益分派日期是什么时候？" --symbol 600519 --top-k 3
```

扫描PDF目前不支持OCR。元数据由导入者核验，不按文件名猜股票身份。
时序页面首次请选择“联网创建快照”，准备至少621条有效价格（通常需要三年以上区间），然后显式运行实验。

## 故障与限制

首次安装需要访问python.org及依赖下载源。出现错误请保留窗口查看提示，解决网络或权限问题后重试；不会用虚假行情补齐失败结果。不要转发包含敏感信息的完整终端截图。
自动环境准备仅适配Windows x64；其他系统需自行准备Python3.12环境。企业策略可能阻止脚本或安装。
本包是“源码+启动器”，不是离线安装包。运行环境和RAG模型需要首次联网下载。最新项目回归为503项通过；已有环境启动已验证，但无Python的全新机器自动安装尚待验收。
进一步说明见 docs/WINDOWS_START.md。


## 版本与测试

AKShare 固定为 1.20.35。行情页面向前读取最多 365 个自然日的真实日线用于指标预热，再显示所选区间。

测试使用 pytest。完整回归中的本地 BGE 集成测试需先运行 python -m rag download-model 准备模型。
