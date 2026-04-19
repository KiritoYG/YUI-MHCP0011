import shutil
from fastapi import APIRouter

router = APIRouter(prefix="/api/docker")

@router.get("/probe")
def probe_docker():
    """
    检查系统环境变量中是否有 docker 命令
    shutil.which 在 Windows 下会检查 .exe, Linux/Mac 下检查执行权限
    """
    docker_path = shutil.which("docker")
    return {
        "installed": docker_path is not None,
        "path": docker_path  # 可选：返回具体安装路径
    }