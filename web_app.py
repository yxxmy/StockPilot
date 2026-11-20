"""完整网页推荐入口：主线程预载统计库，再启动Streamlit。"""
from pathlib import Path
import sys


def main():
    # Windows下首次在线程中载入原生依赖可能阻塞；这里不拟合、不读取密钥。
    import arch  # noqa:F401
    from streamlit.web import cli
    sys.argv=["streamlit","run",str(Path(__file__).resolve().with_name("streamlit_app.py")),*sys.argv[1:]]
    cli.main()


if __name__=="__main__":
    main()
