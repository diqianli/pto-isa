# GitHub 访问和 npm Git 集成修复指南

## 问题诊断结果

### 问题1: GitHub 连接问题
**症状**: 无法访问 GitHub (443端口超时)
**原因**: 防火墙阻止了到 github.com:443 的直接连接
**DNS状态**: ✅ 可以解析 (20.205.243.166)
**TCP连接**: ❌ 443端口连接超时

### 问题2: npm 无法使用 git 克隆
**症状**: npm install 失败，错误 `spawn git ENOENT`
**原因**: npm 在 Windows 环境下找不到 git.exe
**Git状态**: ✅ 已安装 (2.53.0)
**npm配置**: ❌ 未正确配置 git 路径

---

## 解决方案总览

| 方案 | 难度 | 效果 | 推荐度 |
|------|------|------|--------|
| 方案A: 配置系统代理 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 🔥 强烈推荐 |
| 方案B: 使用 SSH 密钥 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ 推荐 |
| 方案C: 配置镜像源 | ⭐ | ⭐⭐⭐⭐ | ✅ 推荐 |
| 方案D: 修改 npm git 配置 | ⭐ | ⭐⭐⭐⭐ | ✅ 必须 |

---

## 🎯 方案A: 配置系统代理（最推荐）

### 步骤1: 确认您的代理服务器

如果您有企业/学校代理，请获取：
- 代理服务器地址
- 端口号（通常是 1080, 8080, 7890 等）

### 步骤2: 配置 Git 代理

```bash
# 设置 HTTP 代理（替换为您的代理地址）
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# 或者使用 SOCKS5 代理
git config --global core.gitproxy 'socks5://127.0.0.1:1080'
```

### 步骤3: 配置 npm 代理

```bash
# 设置 npm 代理
npm config set proxy http://127.0.0.1:7890
npm config set https-proxy http://127.0.0.1:7890

# 验证配置
npm config get proxy
npm config get https-proxy
```

### 步骤4: 配置环境变量（可选但推荐）

**Windows PowerShell**:
```powershell
# 设置用户环境变量
[System.Environment]::SetEnvironmentVariable('HTTP_PROXY', 'http://127.0.0.1:7890', 'User')
[System.Environment]::SetEnvironmentVariable('HTTPS_PROXY', 'http://127.0.0.1:7890', 'User')

# 或者在当前会话设置
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
```

**Git Bash / MSYS2** (添加到 `~/.bashrc`):
```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

### 步骤5: 验证修复

```bash
# 测试 git 连接
git ls-remote https://github.com/vercel-labs/skills.git

# 测试 npm git 功能
npm install -g https://github.com/vercel-labs/skills.git --yes
```

---

## 🔑 方案B: 使用 SSH 密钥（推荐）

### 步骤1: 生成 SSH 密钥

```bash
# 生成 ed25519 密钥（推荐）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 或者使用 RSA 密钥
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

### 步骤2: 添加 SSH 密钥到 GitHub

1. 复制公钥内容：
```bash
cat ~/.ssh/id_ed25519.pub
# 或
cat ~/.ssh/id_rsa.pub
```

2. 访问 GitHub 添加密钥：
   - 打开 https://github.com/settings/keys
   - 点击 "New SSH key"
   - 粘贴公钥内容
   - 点击 "Add SSH key"

### 步骤3: 配置 Git 使用 SSH

```bash
# 测试 SSH 连接
ssh -T git@github.com

# 如果成功，会看到：Hi username! You've successfully authenticated...
```

### 步骤4: 配置 npm 使用 SSH

```bash
# 为 GitHub 配置 SSH
git config --global url."git@github.com:".insteadOf "https://github.com/"

# 验证配置
git config --global --get-regexp url
```

### 步骤5: 测试 npm 安装

```bash
# 使用 SSH URL 安装
npm install -g git@github.com:vercel-labs/skills.git --yes
```

---

## 🌐 方案C: 配置镜像源（推荐用于中国用户）

### 步骤1: 配置 npm 淘宝镜像

```bash
# 设置 npm 淘宝镜像
npm config set registry https://registry.npmmirror.com

# 验证配置
npm config get registry
```

### 步骤2: 配置 GitHub 镜像

**方法1: 使用 ghproxy**
```bash
# 克隆 GitHub 仓库
git clone https://mirror.ghproxy.com/https://github.com/vercel-labs/skills.git

# 进入目录
cd skills

# 安装
npm install
npm run build
npm install -g .
```

**方法2: 使用 gitclone**
```bash
git clone https://gitclone.github.com/github/vercel-labs/skills
```

**方法3: 使用 cnpmjs 镜像**
```bash
git clone https://github.com.cnpmjs.org/vercel-labs/skills.git
```

### 步骤3: 配置 Git 使用镜像

```bash
# 为 GitHub 配置镜像
git config --global url."https://mirror.ghproxy.com/https://github.com/".insteadOf "https://github.com/"

# 测试连接
git ls-remote https://github.com/vercel-labs/skills.git
```

### 步骤4: 恢复官方源（当需要时）

```bash
# 取消镜像配置
git config --global --unset url."https://mirror.ghproxy.com/https://github.com/".insteadOf

# 恢复 npm 官方源
npm config set registry https://registry.npmjs.org
```

---

## 🔧 方案D: 修复 npm Git 集成（必须）

### 步骤1: 确保 Git 在 PATH 中

```bash
# 检查 git 位置
where git
# 或
which git

# 输出应该是：/mingw64/bin/git
```

### 步骤2: 配置 npm 使用正确的 Git

**方法1: 使用 npm 配置**
```bash
# 明确告诉 npm 使用哪个 git
npm config set git "C:/Program Files/Git/bin/git.exe"
# 或
npm config set git "C:/mingw64/bin/git.exe"
```

**方法2: 添加到系统 PATH**

1. 打开"系统属性" → "高级" → "环境变量"
2. 在"系统变量"中找到 `Path`
3. 添加 Git 路径：
   - `C:\Program Files\Git\bin`
   - `C:\Program Files\Git\cmd`
   - `C:\mingw64\bin`

### 步骤3: 使用完整路径调用 git

```bash
# 使用完整路径
npm config set git "C:/mingw64/bin/git.exe"

# 验证配置
npm config get git
```

### 步骤4: 测试 npm git 功能

```bash
# 创建测试目录
cd /tmp
mkdir test-npm-git
cd test-npm-git

# 初始化 package.json
echo '{"name":"test"}' > package.json

# 测试 git 依赖
npm install git+https://github.com/user/repo.git --yes
```

---

## 🚀 完整修复脚本

### Windows 修复脚本（保存为 `fix-github.sh`）

```bash
#!/bin/bash
echo "=== GitHub 访问和 npm Git 集成修复脚本 ==="

# 检查代理配置
echo "请输入您的代理地址（如 http://127.0.0.1:7890），如果没有直接按回车："
read -r PROXY_ADDR

if [ ! -z "$PROXY_ADDR" ]; then
    echo "配置代理..."
    git config --global http.proxy "$PROXY_ADDR"
    git config --global https.proxy "$PROXY_ADDR"
    npm config set proxy "$PROXY_ADDR"
    npm config set https-proxy "$PROXY_ADDR"
    echo "✓ 代理配置完成"
fi

# 配置 npm 使用完整 git 路径
echo "配置 npm git 路径..."
GIT_PATH=$(which git)
if [ ! -z "$GIT_PATH" ]; then
    npm config set git "$GIT_PATH"
    echo "✓ npm git 路径配置完成: $GIT_PATH"
fi

# 配置 npm 淘宝镜像
echo "配置 npm 淘宝镜像..."
npm config set registry https://registry.npmmirror.com
echo "✓ npm 淘宝镜像配置完成"

# 测试连接
echo ""
echo "测试连接..."
echo "1. 测试 git..."
git ls-remote https://github.com/vercel-labs/skills.git && echo "✓ Git 连接成功" || echo "✗ Git 连接失败"

echo ""
echo "2. 测试 npm..."
npm config get registry && echo "✓ npm 配置成功"

echo ""
echo "=== 修复完成 ==="
echo "请重启终端并测试："
echo "  npm install -g https://github.com/vercel-labs/skills.git --yes"
```

### 使用修复脚本

```bash
# 给脚本执行权限
chmod +x fix-github.sh

# 运行脚本
./fix-github.sh
```

---

## ✅ 验证修复

### 测试1: Git 连接测试

```bash
# 测试 HTTPS 连接
git ls-remote https://github.com/vercel-labs/skills.git

# 应该输出：
# f7b8d9f3...    HEAD
# f7b8d9f3...    refs/heads/main
# ...
```

### 测试2: npm Git 集成测试

```bash
# 测试安装 git 包
cd /tmp
mkdir test-npm
cd test-npm
echo '{"name":"test-npm-git"}' > package.json

# 尝试安装
npm install git+https://github.com/user/repo.git --dry-run --yes
```

### 测试3: 完整安装测试

```bash
# 最终测试：安装 skills
npm install -g https://github.com/vercel-labs/skills.git --yes
```

---

## 🔧 高级配置

### 使用多个代理（备用配置）

```bash
# 配置多个代理
git config --global http.proxy http://proxy1.example.com:8080
git config --global https.proxy http://proxy2.example.com:8080

# 为特定域名设置代理
git config --global http.https://github.com.proxy http://github-proxy:8080
```

### 配置自动切换代理

创建 `~/.gitconfig-auto-proxy`:
```bash
#!/bin/bash
# 自动选择最快的代理
PROXIES=("http://proxy1:8080" "http://proxy2:8080" "http://proxy3:8080")

for proxy in "${PROXIES[@]}"; do
    if curl -x "$proxy" --connect-timeout 3 https://github.com >/dev/null 2>&1; then
        git config --global http.proxy "$proxy"
        git config --global https.proxy "$proxy"
        echo "使用代理: $proxy"
        exit 0
    fi
done

echo "所有代理都不可用，使用直连"
git config --global --unset http.proxy
git config --global --unset https.proxy
```

---

## 📋 故障排除清单

### 如果代理配置后仍无法连接

- [ ] 代理服务器是否运行中？
- [ ] 代理端口是否正确？
- [ ] 防火墙是否允许代理端口？
- [ ] 代理是否支持 HTTPS？

### 如果 SSH 密钥无法工作

- [ ] 密钥是否正确添加到 GitHub？
- [ ] SSH 私钥权限是否正确（600）？
- [ ] SSH 配置文件是否存在 (~/.ssh/config)？
- [ ] 是否使用正确的密钥类型（ed25519 或 RSA）？

### 如果 npm 仍无法使用 git

- [ ] `npm config get git` 是否返回有效路径？
- [ ] Git 是否在 PATH 中？
- [ ] 是否重启了终端？
- [ ] 是否尝试了完整路径配置？

---

## 📞 获取帮助

如果以上方案都无法解决问题：

1. **检查网络环境**
   - 联系网络管理员
   - 确认防火墙规则
   - 测试代理服务器

2. **使用离线方法**
   - 手动下载 ZIP 文件
   - 使用本地文件安装

3. **临时解决方案**
   - 使用手机热点下载
   - 使用其他网络环境
   - 使用云服务器中转

---

**文档版本**: 2.0
**最后更新**: 2025-02-10
**状态**: ✅ 完整修复方案已准备就绪
