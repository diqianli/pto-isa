@echo off
REM Claude-Mem 文档一键推送脚本
REM 版本: 1.0
REM 日期: 2026-02-14

echo ========================================
echo Claude-Mem 文档推送到 GitHub
echo ========================================
echo.

echo 当前状态:
cd /d E:\cccode\pto-isa
git status
echo.

echo 即将推送以下提交:
git log --oneline origin/main..HEAD
echo.

echo ========================================
echo 选择推送方法:
echo ========================================
echo.
echo 1. 使用 GitHub CLI (推荐)
echo 2. 使用 Personal Access Token
echo 3. 手动推送（显示命令）
echo 4. 取消
echo.

set /p choice="请选择 (1-4): "

if "%choice%"=="1" goto :gh_cli
if "%choice%"=="2" goto :pat
if "%choice%"=="3" goto :manual
if "%choice%"=="4" goto :end
echo 无效选择，退出
goto :end

:gh_cli
echo.
echo [1/3] 检查 GitHub CLI...
where gh >nul 2>&1
if %errorlevel% neq 0 (
    echo GitHub CLI 未安装，正在安装...
    winget install --id GitHub.cli -e --accept-source-agreements --accept-package-agreements
    if %errorlevel% neq 0 (
        echo 安装失败，请使用其他方法
        goto :end
    )
)
echo ✓ GitHub CLI 已安装

echo.
echo [2/3] 登录 GitHub...
gh auth login
if %errorlevel% neq 0 (
    echo 登录失败
    goto :end
)
echo ✓ 已登录

echo.
echo [3/3] 推送到 GitHub...
git push origin main
if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo ✓ 推送成功！
    echo ========================================
    echo.
    echo 访问 https://github.com/diqianli/pto-isa 查看更新
) else (
    echo 推送失败，请检查网络连接和凭据
)
goto :end

:pat
echo.
echo ========================================
echo 使用 Personal Access Token 推送
echo ========================================
echo.
echo 步骤：
echo 1. 访问 https://github.com/settings/tokens
echo 2. 点击 "Generate new token (classic)"
echo 3. 勾选 "repo" 权限
echo 4. 生成并复制 token
echo.
pause

echo.
echo 正在推送到 GitHub...
echo 提示输入凭据时：
echo   Username: diqianli ^(你的 GitHub 用户名^)
echo   Password: 粘贴 token ^(不是你的密码^)
echo.
git push origin main
if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo ✓ 推送成功！
    echo ========================================
    echo.
    echo 访问 https://github.com/diqianli/pto-isa 查看更新
) else (
    echo 推送失败，请检查凭据
)
goto :end

:manual
echo.
echo ========================================
echo 手动推送命令
echo ========================================
echo.
echo 方法 1: GitHub CLI
echo   gh auth login
echo   git push origin main
echo.
echo 方法 2: Personal Access Token
echo   git push origin main
echo   ^(输入用户名和 token^)
echo.
echo 方法 3: 配置 SSH
echo   ssh-keygen -t ed25519 -C "your_email@example.com"
echo   cat ~/.ssh/id_ed25519.pub
echo   ^(添加公钥到 https://github.com/settings/ssh/new^)
echo   git remote set-url origin git@github.com:diqianli/pto-isa.git
echo   git push origin main
echo.
pause
goto :end

:end
echo.
pause
