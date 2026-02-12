# 🔧 GitHub 访问和 npm Git 修复 - 快速参考卡

## ⚡ 快速修复（3步）

### Windows 用户
```batch
# 1. 运行修复脚本
fix-network.bat

# 2. 重启命令提示符

# 3. 测试安装
npm install -g https://github.com/vercel-labs/skills.git --yes
```

### Git Bash / MSYS2 用户
```bash
# 1. 运行修复脚本
chmod +x fix-network.sh
./fix-network.sh

# 2. 重启终端

# 3. 测试安装
npm install -g https://github.com/vercel-labs/skills.git --yes
```

---

## 🎯 问题诊断结果

| 问题 | 状态 | 原因 |
|------|------|------|
| GitHub 访问 | ❌ 443端口超时 | 防火墙阻止 |
| Git 安装 | ✅ 2.53.0 | 正常 |
| npm Git集成 | ❌ 找不到git.exe | 路径问题 |
| DNS解析 | ✅ 20.205.243.166 | 正常 |

---

## 📋 修复方案选择

### 方案1: 使用代理 ⭐⭐⭐⭐⭐
**最适合**: 有企业/学校代理的用户

```bash
# Git 代理
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# npm 代理
npm config set proxy http://127.0.0.1:7890
npm config set https-proxy http://127.0.0.1:7890
```

### 方案2: 使用镜像 ⭐⭐⭐⭐⭐
**最适合**: 中国大陆用户

```bash
# npm 淘宝镜像
npm config set registry https://registry.npmmirror.com

# GitHub 镜像
git clone https://mirror.ghproxy.com/https://github.com/vercel-labs/skills.git
```

### 方案3: 使用 SSH 密钥 ⭐⭐⭐⭐
**最适合**: 开发者账户

```bash
# 1. 生成密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 添加到 GitHub: https://github.com/settings/keys

# 3. 配置 Git
git config --global url."git@github.com:".insteadOf "https://github.com/"

# 4. 测试
ssh -T git@github.com
npm install -g git@github.com:vercel-labs/skills.git --yes
```

### 方案4: 修复 npm Git 路径 ⭐⭐⭐⭐⭐
**必须执行**

```bash
# Windows
npm config set git "C:/Program Files/Git/bin/git.exe"
# 或
npm config set git "C:/mingw64/bin/git.exe"

# Git Bash
npm config set git $(which git)
```

---

## ✅ 验证修复

```bash
# 测试 Git 连接
git ls-remote https://github.com/vercel-labs/skills.git

# 应该看到：
# f7b8d9f3...    HEAD
# f7b8d9f3...    refs/heads/main
```

```bash
# 测试 npm Git
cd /tmp
mkdir test-npm
cd test-npm
echo '{"name":"test"}' > package.json
npm install git+https://github.com/user/repo.git --dry-run --yes
```

---

## 🚨 常见错误及解决

### 错误1: `spawn git ENOENT`
**原因**: npm 找不到 git.exe

**解决**:
```bash
# 方法1: 配置 npm git 路径
npm config set git "C:/mingw64/bin/git.exe"

# 方法2: 添加 Git 到 PATH
# 系统属性 → 高级 → 环境变量 → Path → 新建
# C:\Program Files\Git\bin
# C:\Program Files\Git\cmd
```

### 错误2: `Connection timed out`
**原因**: 防火墙阻止 GitHub 443端口

**解决**:
```bash
# 使用代理
git config --global http.proxy http://proxy-server:port
npm config set proxy http://proxy-server:port

# 或使用镜像
git clone https://mirror.ghproxy.com/https://github.com/repo.git
```

### 错误3: `Permission denied (publickey)`
**原因**: SSH 密钥未配置或权限错误

**解决**:
```bash
# 使用 HTTPS 而不是 SSH
git clone https://github.com/user/repo.git
# 而不是
git clone git@github.com:user/repo.git

# 或者配置 SSH 密钥
# 1. 生成密钥: ssh-keygen -t ed25519
# 2. 复制公钥: cat ~/.ssh/id_ed25519.pub
# 3. 添加到: https://github.com/settings/keys
```

---

## 📱 移动端/其他设备

### 使用手机下载
1. 手机访问: https://github.com/vercel-labs/skills
2. 点击 "Code" → "Download ZIP"
3. 传输到电脑
4. 解压并安装:
   ```bash
   unzip skills.zip
   cd skills-main
   npm install
   npm run build
   npm install -g .
   ```

### 使用云服务器
```bash
# 在可访问 GitHub 的服务器上
git clone https://github.com/vercel-labs/skills.git
cd skills
npm run build

# 打包
tar czf skills.tar.gz skills/

# 下载到本地
# 在本地解压并安装
```

---

## 📞 仍需帮助？

### 检查清单
- [ ] Git 已安装并添加到 PATH
- [ ] Node.js 和 npm 已安装
- [ ] 代理服务器正常运行（如果使用代理）
- [ ] SSH 密钥已添加到 GitHub（如果使用 SSH）
- [ ] 防火墙允许 git/npm 访问网络

### 联系方式
- 网络管理员：确认代理设置和防火墙规则
- IT 部门：请求添加 GitHub 到白名单
- GitHub 支持：https://github.com/contact

---

## 📚 相关文档

- [完整修复指南](network_fix_complete_guide.md) - 详细步骤
- [Skills 安装指南](skills_installation_guide.md) - 离线安装方法
- [PTO-ISA 自定义 Skills](../.claude/README.md) - 使用已配置的 skills

---

**版本**: 2.0
**更新**: 2025-02-10
**状态**: ✅ 完整修复方案已就绪
