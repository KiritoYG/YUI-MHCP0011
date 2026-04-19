@echo off

REM 显示项目信息
echo GitHub Repository: https://github.com/KiritoYG/yui-mhcp001
echo.

REM 设置 node 和 npm 的路径
set "NODE_PATH=%~dp0nodejs"
set "NPM_CLI_PATH=%~dp0nodejs\node_modules\npm\bin\npm-cli.js"

REM 将本地 nodejs 目录加入 PATH，使 electron/cross-env 能找到 node 命令
set "PATH=%NODE_PATH%;%PATH%"

REM 使用本地 node 执行本地 npm-cli.js
call "%NODE_PATH%\node.exe" "%NPM_CLI_PATH%" run dev

pause